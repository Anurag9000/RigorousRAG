"""Dataset-centric synchronized multi-model training runtime.

This module is the repository-neutral execution contract layered *under* the
literal OPF_ADP pressure scheduler and *above* repository-specific model/data
adapters.  The outer OPF scheduler still owns process admission, GPU assignment,
RAM/VRAM/swap pressure, retries and process lifecycle.  This layer owns the
in-process invariant requested for models that consume the same dataset:

* build one cohort per single dataset, ordered by descending number of distinct
  model families (then model count/name for deterministic ties);
* place every multi-dataset/overlap model in one final overlap cohort;
* force one physical batch cardinality inside a cohort;
* load each raw batch once, cache each distinct prepared/device view once and
  hand the same object/storage to every compatible model;
* step admitted models concurrently (CUDA streams for torch adapters when
  possible), synchronize, checkpoint one shared batch cursor, then advance;
* expose explicit CPU-only and GPU-first variants.  Auto mode prefers GPU and
  rebuilds the *whole* cohort on CPU after a CUDA/device OOM so model cursors
  cannot silently diverge;
* support NumPy/CuPy array substitution without making either dependency
  mandatory, while framework-specific adapters remain responsible for their
  explicit device placement.

A repository is not considered migrated merely because this module exists.  Its
scientific authority must provide DatasetAdapter/ModelAdapter implementations and
must fail closed if a retained training job is missing from the cohort manifest.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import enum
import hashlib
import importlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping, Protocol, Sequence

SCHEMA = "opf-dataset-cohort-runtime/v1"
PLAN_SCHEMA = "opf-dataset-cohort-plan/v1"
STATE_SCHEMA = "opf-dataset-cohort-state/v1"
OVERLAP_DATASET_KEY = "__overlap__"


class CohortError(RuntimeError):
    pass


class CohortOOM(CohortError):
    pass


class BackendUnavailable(CohortError):
    pass


class BackendMode(str, enum.Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


@dataclasses.dataclass(frozen=True)
class BackendProbe:
    mode: BackendMode
    gpu_available: bool
    selected_device: str
    torch_cuda: bool
    cupy_cuda: bool
    jax_gpu: bool
    tensorflow_gpu: bool
    nvidia_smi: bool
    reasons: tuple[str, ...]


def _safe_import(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _nvidia_smi() -> bool:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return False
    try:
        run = subprocess.run(
            [binary, "--query-gpu=index", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
        return run.returncode == 0 and bool(run.stdout.strip())
    except Exception:
        return False


def detect_backend(mode: str | BackendMode = BackendMode.AUTO) -> BackendProbe:
    selected = mode if isinstance(mode, BackendMode) else BackendMode(str(mode))
    if _truthy(os.environ.get("CPU_ONLY")) or _truthy(os.environ.get("TRAINING_CONTROL_CPU_ONLY")):
        selected = BackendMode.CPU
    hidden = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch_cuda = cupy_cuda = jax_gpu = tensorflow_gpu = False
    torch = _safe_import("torch")
    if torch is not None:
        with contextlib.suppress(Exception):
            torch_cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    cupy = _safe_import("cupy")
    if cupy is not None:
        with contextlib.suppress(Exception):
            cupy_cuda = int(cupy.cuda.runtime.getDeviceCount()) > 0
    jax = _safe_import("jax")
    if jax is not None:
        with contextlib.suppress(Exception):
            jax_gpu = any(str(device.platform).lower() in {"gpu", "cuda", "rocm"} for device in jax.devices())
    tf = _safe_import("tensorflow")
    if tf is not None:
        with contextlib.suppress(Exception):
            tensorflow_gpu = bool(tf.config.list_physical_devices("GPU"))
    smi = _nvidia_smi()
    available = not hidden and (torch_cuda or cupy_cuda or jax_gpu or tensorflow_gpu or smi)
    if selected is BackendMode.GPU and not available:
        raise BackendUnavailable("GPU backend required but no usable GPU was detected")
    use_gpu = available and selected is not BackendMode.CPU
    index = os.environ.get("TRAINING_CONTROL_GPU_INDEX", os.environ.get("GPU_DEVICE_INDEX", "0")).strip() or "0"
    reasons = tuple(
        name
        for name, enabled in (
            ("torch.cuda", torch_cuda),
            ("cupy.cuda", cupy_cuda),
            ("jax.gpu", jax_gpu),
            ("tensorflow.gpu", tensorflow_gpu),
            ("nvidia-smi", smi),
            ("CUDA_VISIBLE_DEVICES hides GPUs", hidden),
        )
        if enabled
    )
    return BackendProbe(selected, available, f"cuda:{index}" if use_gpu else "cpu", torch_cuda, cupy_cuda, jax_gpu, tensorflow_gpu, smi, reasons)


def subprocess_environment(
    mode: str | BackendMode,
    *,
    gpu_index: int | str | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a strict CPU-only or GPU-pinned child environment.

    In-process cohort execution never rewrites scheduler-owned CUDA visibility;
    this helper is for repository launchers that create a separate backend child.
    """
    selected = mode if isinstance(mode, BackendMode) else BackendMode(str(mode))
    env = dict(os.environ if base is None else base)
    if selected is BackendMode.CPU:
        env.update(
            CUDA_VISIBLE_DEVICES="",
            HIP_VISIBLE_DEVICES="",
            ROCR_VISIBLE_DEVICES="",
            JAX_PLATFORMS="cpu",
            JAX_PLATFORM_NAME="cpu",
            TRAINING_CONTROL_BACKEND="cpu",
            TRAINING_CONTROL_CPU_ONLY="1",
        )
        return env
    index = str(gpu_index if gpu_index is not None else env.get("TRAINING_CONTROL_GPU_INDEX", env.get("GPU_DEVICE_INDEX", "0"))).strip() or "0"
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = index
    env.pop("TRAINING_CONTROL_CPU_ONLY", None)
    env["TRAINING_CONTROL_BACKEND"] = "gpu" if selected is BackendMode.GPU else "auto"
    return env


def array_namespace(mode: str | BackendMode = BackendMode.AUTO) -> Any:
    probe = detect_backend(mode)
    if probe.selected_device.startswith("cuda"):
        cupy = _safe_import("cupy")
        if cupy is not None:
            return cupy
    numpy = _safe_import("numpy")
    if numpy is None:
        raise BackendUnavailable("NumPy is required by the CPU array backend")
    return numpy


def to_array(value: Any, *, mode: str | BackendMode = BackendMode.AUTO, dtype: Any = None) -> Any:
    return array_namespace(mode).asarray(value, dtype=dtype)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    job_id: str
    model_family: str
    datasets: tuple[str, ...]
    view_key: str
    batch_size: int
    adapter: str
    device_capable: bool = True
    cpu_capable: bool = True
    gpu_capable: bool = True
    exact_resume: bool = True
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_id.strip() or not self.model_family.strip() or not self.view_key.strip() or not self.adapter.strip():
            raise ValueError("job_id/model_family/view_key/adapter must be non-empty")
        if not self.datasets or any(not str(value).strip() for value in self.datasets):
            raise ValueError(f"{self.job_id}: at least one dataset is required")
        if self.batch_size <= 0:
            raise ValueError(f"{self.job_id}: batch_size must be positive")
        if not self.cpu_capable:
            raise ValueError(f"{self.job_id}: CPU variant is mandatory")
        if self.device_capable and not self.gpu_capable:
            raise ValueError(f"{self.job_id}: device-capable job lacks a GPU variant")
        if not self.exact_resume:
            raise ValueError(f"{self.job_id}: cohort jobs must support exact resume")

    @property
    def normalized_datasets(self) -> tuple[str, ...]:
        return tuple(sorted(dict.fromkeys(str(value).strip() for value in self.datasets)))


@dataclasses.dataclass(frozen=True)
class Cohort:
    index: int
    dataset_key: str
    overlap: bool
    models: tuple[ModelSpec, ...]
    uniform_batch_size: int
    distinct_model_families: int
    datasets: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class CohortPlan:
    cohorts: tuple[Cohort, ...]
    spec_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PLAN_SCHEMA,
            "spec_digest": self.spec_digest,
            "cohorts": [
                {
                    "index": cohort.index,
                    "dataset_key": cohort.dataset_key,
                    "overlap": cohort.overlap,
                    "uniform_batch_size": cohort.uniform_batch_size,
                    "distinct_model_families": cohort.distinct_model_families,
                    "datasets": list(cohort.datasets),
                    "models": [dataclasses.asdict(model) for model in cohort.models],
                }
                for cohort in self.cohorts
            ],
        }


def build_cohort_plan(specs: Sequence[ModelSpec], *, batch_size_override: int | None = None) -> CohortPlan:
    if not specs:
        return CohortPlan((), _digest([]))
    seen: set[str] = set()
    singles: dict[str, list[ModelSpec]] = defaultdict(list)
    overlap: list[ModelSpec] = []
    for spec in specs:
        if spec.job_id in seen:
            raise ValueError(f"duplicate cohort job_id: {spec.job_id}")
        seen.add(spec.job_id)
        datasets = spec.normalized_datasets
        (singles[datasets[0]] if len(datasets) == 1 else overlap).append(spec)

    def uniform(models: Sequence[ModelSpec]) -> int:
        if batch_size_override is not None:
            if batch_size_override <= 0:
                raise ValueError("batch_size_override must be positive")
            return int(batch_size_override)
        return min(model.batch_size for model in models)

    def rank(item: tuple[str, list[ModelSpec]]) -> tuple[int, int, str]:
        dataset, models = item
        return (-len({m.model_family for m in models}), -len(models), dataset)

    cohorts: list[Cohort] = []
    for index, (dataset, models) in enumerate(sorted(singles.items(), key=rank), start=1):
        ordered = tuple(sorted(models, key=lambda m: (m.model_family, m.job_id)))
        cohorts.append(Cohort(index, dataset, False, ordered, uniform(ordered), len({m.model_family for m in ordered}), (dataset,)))
    if overlap:
        ordered = tuple(sorted(overlap, key=lambda m: (m.model_family, m.job_id)))
        datasets = tuple(sorted({d for model in ordered for d in model.normalized_datasets}))
        cohorts.append(Cohort(len(cohorts) + 1, OVERLAP_DATASET_KEY, True, ordered, uniform(ordered), len({m.model_family for m in ordered}), datasets))
    serializable = [
        {
            "job_id": m.job_id,
            "model_family": m.model_family,
            "datasets": list(m.normalized_datasets),
            "view_key": m.view_key,
            "batch_size": m.batch_size,
            "adapter": m.adapter,
            "device_capable": m.device_capable,
            "cpu_capable": m.cpu_capable,
            "gpu_capable": m.gpu_capable,
            "exact_resume": m.exact_resume,
            "metadata": dict(m.metadata),
        }
        for m in sorted(specs, key=lambda row: row.job_id)
    ]
    return CohortPlan(tuple(cohorts), _digest(serializable))


class DatasetAdapter(Protocol):
    dataset_key: str
    def iter_raw_batches(self, *, batch_size: int, start_batch: int, seed: int) -> Iterator[Any]: ...
    def state_dict(self) -> Mapping[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class ModelAdapter(Protocol):
    framework: str
    view_key: str
    def build(self, *, device: str, batch_size: int) -> None: ...
    def prepare_view(self, raw_batch: Any, *, device: str) -> Any: ...
    def train_step(self, batch: Any, *, batch_index: int) -> Mapping[str, Any] | None: ...
    def save_checkpoint(self, path: Path) -> Mapping[str, Any] | None: ...
    def load_checkpoint(self, path: Path, metadata: Mapping[str, Any]) -> None: ...
    def close(self) -> None: ...


@dataclasses.dataclass
class SharedBatch:
    raw: Any
    views: MutableMapping[tuple[str, str], Any] = dataclasses.field(default_factory=dict)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def get_view(self, adapter: ModelAdapter, *, device: str) -> Any:
        key = (str(adapter.view_key), device)
        with self.lock:
            if key not in self.views:
                self.views[key] = adapter.prepare_view(self.raw, device=device)
            return self.views[key]

    def clear(self) -> None:
        with self.lock:
            self.views.clear()
            self.raw = None


@dataclasses.dataclass(frozen=True)
class PressureSnapshot:
    ram_percent: float | None
    swap_percent: float | None
    gpu_used_percent: float | None
    gpu_free_bytes: int | None


class PressureGuard:
    def __init__(self) -> None:
        self.ram_stop = float(os.environ.get("COHORT_RAM_STOP_PERCENT", 90.0))
        self.swap_stop = float(os.environ.get("COHORT_SWAP_STOP_PERCENT", 70.0))
        self.gpu_stop = float(os.environ.get("COHORT_GPU_STOP_PERCENT", 90.0))
        self.poll = float(os.environ.get("COHORT_PRESSURE_POLL_SECONDS", 0.5))

    def snapshot(self, device: str) -> PressureSnapshot:
        ram = swap = gpu = None
        free_bytes = None
        psutil = _safe_import("psutil")
        if psutil is not None:
            with contextlib.suppress(Exception):
                ram, swap = float(psutil.virtual_memory().percent), float(psutil.swap_memory().percent)
        if device.startswith("cuda"):
            torch = _safe_import("torch")
            if torch is not None:
                with contextlib.suppress(Exception):
                    free, total = torch.cuda.mem_get_info()
                    free_bytes, gpu = int(free), 100.0 * (1.0 - float(free) / float(total))
            if gpu is None:
                cupy = _safe_import("cupy")
                if cupy is not None:
                    with contextlib.suppress(Exception):
                        free, total = cupy.cuda.runtime.memGetInfo()
                        free_bytes, gpu = int(free), 100.0 * (1.0 - float(free) / float(total))
        return PressureSnapshot(ram, swap, gpu, free_bytes)

    def wait_until_safe(self, device: str) -> PressureSnapshot:
        while True:
            snap = self.snapshot(device)
            if not (
                (snap.ram_percent is not None and snap.ram_percent >= self.ram_stop)
                or (snap.swap_percent is not None and snap.swap_percent >= self.swap_stop)
                or (snap.gpu_used_percent is not None and snap.gpu_used_percent >= self.gpu_stop)
            ):
                return snap
            time.sleep(max(0.05, self.poll))


class AdapterRegistry:
    def __init__(self) -> None:
        self.models: dict[str, Callable[[ModelSpec], ModelAdapter]] = {}
        self.datasets: dict[str, Callable[[Cohort], DatasetAdapter]] = {}

    def register_model(self, name: str, factory: Callable[[ModelSpec], ModelAdapter]) -> None:
        if not name or name in self.models:
            raise ValueError(f"duplicate/invalid model adapter {name!r}")
        self.models[name] = factory

    def register_dataset(self, name: str, factory: Callable[[Cohort], DatasetAdapter]) -> None:
        if not name or name in self.datasets:
            raise ValueError(f"duplicate/invalid dataset adapter {name!r}")
        self.datasets[name] = factory

    def model(self, spec: ModelSpec) -> ModelAdapter:
        if spec.adapter not in self.models:
            raise CohortError(f"missing model adapter {spec.adapter!r} for {spec.job_id}")
        return self.models[spec.adapter](spec)

    def dataset(self, cohort: Cohort) -> DatasetAdapter:
        if cohort.dataset_key not in self.datasets:
            raise CohortError(f"missing dataset adapter for {cohort.dataset_key!r}")
        return self.datasets[cohort.dataset_key](cohort)

    def audit(self, specs: Sequence[ModelSpec], plan: CohortPlan) -> dict[str, Any]:
        errors = [f"{spec.job_id}: missing adapter {spec.adapter}" for spec in specs if spec.adapter not in self.models]
        errors += [f"cohort {cohort.index}: missing dataset adapter {cohort.dataset_key}" for cohort in plan.cohorts if cohort.dataset_key not in self.datasets]
        return {"schema": SCHEMA, "pass": not errors, "errors": errors, "plan": plan.to_dict()}


def _oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(mark in text for mark in ("cuda out of memory", "outofmemoryerror", "resourceexhaustederror", "hip out of memory", "std::bad_alloc"))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class CohortExecutor:
    def __init__(self, registry: AdapterRegistry, *, state_root: str | Path = ".training_control/cohorts") -> None:
        self.registry = registry
        self.root = Path(state_root)
        self.pressure = PressureGuard()
        self.max_parallel = int(os.environ.get("COHORT_MAX_PARALLEL_MODELS", 0))
        self.checkpoint_every = int(os.environ.get("COHORT_CHECKPOINT_EVERY_BATCHES", 1))
        if self.max_parallel < 0 or self.checkpoint_every <= 0:
            raise ValueError("invalid cohort concurrency/checkpoint settings")

    def _state_path(self, cohort: Cohort) -> Path:
        return self.root / f"cohort-{cohort.index:03d}" / "state.json"

    def _model_dir(self, cohort: Cohort, spec: ModelSpec) -> Path:
        return self.root / f"cohort-{cohort.index:03d}" / "models" / spec.job_id.replace("/", "_").replace(":", "__")

    def _load_state(self, cohort: Cohort, digest: str) -> Mapping[str, Any]:
        path = self._state_path(cohort)
        if not path.is_file():
            return {"next_batch": 0, "dataset_state": {}, "model_checkpoints": {}}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") != STATE_SCHEMA or raw.get("plan_digest") != digest or raw.get("dataset_key") != cohort.dataset_key:
            raise CohortError(f"stale/incompatible state for cohort {cohort.index}")
        return raw

    def _save(self, cohort: Cohort, digest: str, next_batch: int, dataset: DatasetAdapter, adapters: Sequence[tuple[ModelSpec, ModelAdapter, Any]], device: str) -> None:
        checkpoints: dict[str, Mapping[str, Any]] = {}
        for spec, adapter, _ in adapters:
            directory = self._model_dir(cohort, spec)
            directory.mkdir(parents=True, exist_ok=True)
            checkpoints[spec.job_id] = dict(adapter.save_checkpoint(directory) or {})
        _atomic_json(
            self._state_path(cohort),
            {
                "schema": STATE_SCHEMA,
                "plan_digest": digest,
                "dataset_key": cohort.dataset_key,
                "uniform_batch_size": cohort.uniform_batch_size,
                "next_batch": next_batch,
                "dataset_state": dict(dataset.state_dict()),
                "model_checkpoints": checkpoints,
                "backend": device,
                "python_rng_digest": _digest(repr(random.getstate())),
            },
        )

    def _build(self, cohort: Cohort, state: Mapping[str, Any], device: str) -> list[tuple[ModelSpec, ModelAdapter, Any]]:
        torch = _safe_import("torch")
        built: list[tuple[ModelSpec, ModelAdapter, Any]] = []
        try:
            for spec in cohort.models:
                self.pressure.wait_until_safe(device)
                adapter = self.registry.model(spec)
                if str(adapter.view_key) != spec.view_key:
                    raise CohortError(f"{spec.job_id}: adapter view_key drift")
                adapter.build(device=device, batch_size=cohort.uniform_batch_size)
                previous = state.get("model_checkpoints", {}).get(spec.job_id)
                if isinstance(previous, Mapping):
                    adapter.load_checkpoint(self._model_dir(cohort, spec), previous)
                stream = None
                if torch is not None and device.startswith("cuda") and str(getattr(adapter, "framework", "")).lower() == "torch":
                    with contextlib.suppress(Exception):
                        stream = torch.cuda.Stream(device=device)
                built.append((spec, adapter, stream))
            return built
        except BaseException:
            for _, adapter, _ in built:
                with contextlib.suppress(Exception):
                    adapter.close()
            raise

    def _step(self, rows: Sequence[tuple[ModelSpec, ModelAdapter, Any]], shared: SharedBatch, device: str, batch_index: int) -> None:
        torch = _safe_import("torch")
        workers = len(rows) if self.max_parallel == 0 else min(len(rows), self.max_parallel)
        workers = max(1, workers)

        def one(row: tuple[ModelSpec, ModelAdapter, Any]) -> None:
            spec, adapter, stream = row
            view = shared.get_view(adapter, device=device)
            try:
                if torch is not None and stream is not None:
                    with torch.cuda.stream(stream):
                        adapter.train_step(view, batch_index=batch_index)
                else:
                    adapter.train_step(view, batch_index=batch_index)
            except BaseException as exc:
                if _oom(exc):
                    raise CohortOOM(f"{spec.job_id}: {exc}") from exc
                raise

        if workers == 1:
            for row in rows:
                one(row)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cohort-model") as pool:
                futures = [pool.submit(one, row) for row in rows]
                for future in futures:
                    future.result()
        if device.startswith("cuda") and torch is not None:
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(device=device)

    def _run_once(self, cohort: Cohort, digest: str, *, device: str, seed: int) -> dict[str, Any]:
        state = self._load_state(cohort, digest)
        dataset = self.registry.dataset(cohort)
        if isinstance(state.get("dataset_state"), Mapping) and state["dataset_state"]:
            dataset.load_state_dict(state["dataset_state"])
        start = int(state.get("next_batch", 0))
        rows = self._build(cohort, state, device)
        completed = start
        try:
            for batch_index, raw in enumerate(dataset.iter_raw_batches(batch_size=cohort.uniform_batch_size, start_batch=start, seed=seed), start=start):
                self.pressure.wait_until_safe(device)
                shared = SharedBatch(raw)
                try:
                    self._step(rows, shared, device, batch_index)
                finally:
                    shared.clear()
                completed = batch_index + 1
                if completed % self.checkpoint_every == 0:
                    self._save(cohort, digest, completed, dataset, rows, device)
            self._save(cohort, digest, completed, dataset, rows, device)
            return {"schema": SCHEMA, "cohort": cohort.index, "dataset": cohort.dataset_key, "backend": device, "batches_completed": completed, "uniform_batch_size": cohort.uniform_batch_size, "models": [spec.job_id for spec, _, _ in rows]}
        finally:
            for _, adapter, _ in rows:
                with contextlib.suppress(Exception):
                    adapter.close()

    def run_cohort(self, cohort: Cohort, *, plan_digest: str, backend: str | BackendMode = BackendMode.AUTO, seed: int = 0) -> dict[str, Any]:
        selected = backend if isinstance(backend, BackendMode) else BackendMode(str(backend))
        if selected is BackendMode.CPU:
            return self._run_once(cohort, plan_digest, device="cpu", seed=seed)
        probe = detect_backend(selected)
        if probe.selected_device.startswith("cuda"):
            try:
                return self._run_once(cohort, plan_digest, device=probe.selected_device, seed=seed)
            except CohortOOM:
                if selected is BackendMode.GPU:
                    raise
                return self._run_once(cohort, plan_digest, device="cpu", seed=seed)
        return self._run_once(cohort, plan_digest, device="cpu", seed=seed)

    def run_plan(self, plan: CohortPlan, *, backend: str | BackendMode = BackendMode.AUTO, seed: int = 0) -> list[dict[str, Any]]:
        return [self.run_cohort(cohort, plan_digest=plan.spec_digest, backend=backend, seed=seed + cohort.index * 1_000_003) for cohort in plan.cohorts]


__all__ = [
    "AdapterRegistry", "BackendMode", "BackendProbe", "BackendUnavailable",
    "Cohort", "CohortError", "CohortExecutor", "CohortOOM", "CohortPlan",
    "DatasetAdapter", "ModelAdapter", "ModelSpec", "OVERLAP_DATASET_KEY",
    "PLAN_SCHEMA", "PressureGuard", "PressureSnapshot", "SCHEMA", "STATE_SCHEMA",
    "SharedBatch", "array_namespace", "build_cohort_plan", "detect_backend",
    "subprocess_environment", "to_array",
]
