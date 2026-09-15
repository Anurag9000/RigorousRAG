"""Transactional completion-aware extension of the canonical dataset-cohort runtime.

v2 deliberately layers on the immutable v1 runtime rather than forking its backend,
plan, pressure or shared-view semantics.  It adds two estate-wide execution rules:

* completed members are removed from view preparation and optimizer stepping as soon
  as an adapter reports ``done`` (or ``is_complete()``/``done``), and the completion
  set is persisted in the shared cohort state;
* very large cohorts may opt into pressure-admitted model residency windows.  A raw
  batch remains owned once by the cohort, while compatible prepared views remain
  cached on that SharedBatch.  Model checkpoints are written to immutable staged
  batch directories and the shared cursor is committed only after every residency
  window succeeds, preventing a partial-window OOM from double-stepping early models.

Residency-window mode is fail-closed.  Every model must declare
``metadata['transactional_batch_safe']=True`` and its adapter must expose
``transactional_batch_safe=True``.  Existing adapters therefore keep v1-style full
residency until they are explicitly audited for deferred/final artifact safety.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

BASE_REPOSITORY = "Anurag9000/RigorousRAG"
BASE_COMMIT = "0c50adb23e5ba58b7c49b18401950f0bf7e5b736"
BASE_PATH = "training/dataset_cohort_runtime.py"
BASE_URL = f"https://raw.githubusercontent.com/{BASE_REPOSITORY}/{BASE_COMMIT}/{BASE_PATH}"
STATE_SCHEMA_V2 = "opf-dataset-cohort-state/v2"
RUNTIME_SCHEMA_V2 = "opf-dataset-cohort-runtime/v2"


def _load_base():
    name = f"_opf_dataset_cohort_runtime_{BASE_COMMIT[:12]}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    cache = Path(os.environ.get("TRAINING_CONTROL_REPO_ROOT") or Path.cwd()).resolve()
    cache = cache / ".training_control" / "dataset_cohort" / BASE_COMMIT / "dataset_cohort_runtime.py"
    if not cache.is_file():
        request = urllib.request.Request(BASE_URL, headers={"User-Agent": "opf-dataset-cohort-runtime-v2/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(cache.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, cache)
    spec = importlib.util.spec_from_file_location(name, cache)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base dataset-cohort runtime: {cache}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()

# Re-export the stable v1 public surface.  CohortExecutor below intentionally
# replaces the one imported from v1.
AdapterRegistry = base.AdapterRegistry
BackendMode = base.BackendMode
BackendProbe = base.BackendProbe
BackendUnavailable = base.BackendUnavailable
Cohort = base.Cohort
CohortError = base.CohortError
CohortOOM = base.CohortOOM
CohortPlan = base.CohortPlan
DatasetAdapter = base.DatasetAdapter
ModelAdapter = base.ModelAdapter
ModelSpec = base.ModelSpec
OVERLAP_DATASET_KEY = base.OVERLAP_DATASET_KEY
PLAN_SCHEMA = base.PLAN_SCHEMA
PressureGuard = base.PressureGuard
PressureSnapshot = base.PressureSnapshot
SCHEMA = RUNTIME_SCHEMA_V2
STATE_SCHEMA = STATE_SCHEMA_V2
SharedBatch = base.SharedBatch
array_namespace = base.array_namespace
build_cohort_plan = base.build_cohort_plan
detect_backend = base.detect_backend
subprocess_environment = base.subprocess_environment
to_array = base.to_array


def _adapter_complete(adapter: Any, result: Mapping[str, Any] | None = None) -> bool:
    if isinstance(result, Mapping) and result.get("done") is True:
        return True
    method = getattr(adapter, "is_complete", None)
    if callable(method):
        with contextlib.suppress(Exception):
            return bool(method())
    return bool(getattr(adapter, "done", False))


def _safe_id(job_id: str) -> str:
    return job_id.replace("/", "_").replace(":", "__")


class CohortExecutor(base.CohortExecutor):
    """v1 executor plus completion pruning and optional transactional residency windows."""

    def __init__(self, registry: AdapterRegistry, *, state_root: str | Path = ".training_control/cohorts") -> None:
        super().__init__(registry, state_root=state_root)
        self.max_resident = int(os.environ.get("COHORT_MAX_RESIDENT_MODELS", "0"))
        if self.max_resident < 0:
            raise ValueError("COHORT_MAX_RESIDENT_MODELS must be >= 0")

    def _load_state(self, cohort: Cohort, digest: str) -> Mapping[str, Any]:
        path = self._state_path(cohort)
        if not path.is_file():
            return {
                "schema": STATE_SCHEMA_V2,
                "next_batch": 0,
                "dataset_state": {},
                "model_checkpoints": {},
                "completed_models": [],
            }
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema") not in {base.STATE_SCHEMA, STATE_SCHEMA_V2}:
            raise CohortError(f"unknown cohort state schema for cohort {cohort.index}: {raw.get('schema')!r}")
        if raw.get("plan_digest") != digest or raw.get("dataset_key") != cohort.dataset_key:
            raise CohortError(f"stale/incompatible state for cohort {cohort.index}")
        migrated = dict(raw)
        migrated.setdefault("completed_models", [])
        return migrated

    def _checkpoint_directory(self, cohort: Cohort, spec: ModelSpec, next_batch: int) -> Path:
        return (
            self.root
            / f"cohort-{cohort.index:03d}"
            / "versions"
            / f"batch-{next_batch:012d}"
            / _safe_id(spec.job_id)
        )

    def _previous_directory(self, cohort: Cohort, spec: ModelSpec, metadata: Mapping[str, Any]) -> Path:
        raw = metadata.get("checkpoint_dir") if isinstance(metadata, Mapping) else None
        if raw:
            path = Path(str(raw))
            return path if path.is_absolute() else self.root / path
        return self._model_dir(cohort, spec)

    def _write_state(
        self,
        cohort: Cohort,
        digest: str,
        *,
        next_batch: int,
        dataset: DatasetAdapter,
        checkpoints: Mapping[str, Mapping[str, Any]],
        completed: set[str],
        device: str,
    ) -> None:
        payload = {
            "schema": STATE_SCHEMA_V2,
            "plan_digest": digest,
            "dataset_key": cohort.dataset_key,
            "uniform_batch_size": cohort.uniform_batch_size,
            "next_batch": int(next_batch),
            "dataset_state": dict(dataset.state_dict()),
            "model_checkpoints": {key: dict(value) for key, value in checkpoints.items()},
            "completed_models": sorted(completed),
            "backend": device,
            "transactional_batch_commit": True,
        }
        base._atomic_json(self._state_path(cohort), payload)

    def _build_specs(
        self,
        cohort: Cohort,
        specs: Sequence[ModelSpec],
        checkpoints: Mapping[str, Mapping[str, Any]],
        device: str,
    ) -> list[tuple[ModelSpec, Any, Any]]:
        torch = base._safe_import("torch")
        built: list[tuple[ModelSpec, Any, Any]] = []
        try:
            for spec in specs:
                self.pressure.wait_until_safe(device)
                adapter = self.registry.model(spec)
                if str(adapter.view_key) != spec.view_key:
                    raise CohortError(f"{spec.job_id}: adapter view_key drift")
                adapter.build(device=device, batch_size=cohort.uniform_batch_size)
                previous = checkpoints.get(spec.job_id)
                if isinstance(previous, Mapping) and previous:
                    adapter.load_checkpoint(self._previous_directory(cohort, spec, previous), previous)
                stream = None
                if (
                    torch is not None
                    and device.startswith("cuda")
                    and str(getattr(adapter, "framework", "")).lower() == "torch"
                ):
                    with contextlib.suppress(Exception):
                        stream = torch.cuda.Stream(device=device)
                built.append((spec, adapter, stream))
            return built
        except BaseException:
            for _, adapter, _ in built:
                with contextlib.suppress(Exception):
                    adapter.close()
            raise

    def _step_rows(
        self,
        rows: Sequence[tuple[ModelSpec, Any, Any]],
        shared: SharedBatch,
        device: str,
        batch_index: int,
    ) -> set[str]:
        torch = base._safe_import("torch")
        active = [row for row in rows if not _adapter_complete(row[1])]
        if not active:
            return {spec.job_id for spec, adapter, _ in rows if _adapter_complete(adapter)}
        workers = len(active) if self.max_parallel == 0 else min(len(active), self.max_parallel)
        workers = max(1, workers)
        completed: set[str] = set()

        def one(row: tuple[ModelSpec, Any, Any]) -> tuple[str, bool]:
            spec, adapter, stream = row
            view = shared.get_view(adapter, device=device)
            try:
                if torch is not None and stream is not None:
                    with torch.cuda.stream(stream):
                        result = adapter.train_step(view, batch_index=batch_index)
                else:
                    result = adapter.train_step(view, batch_index=batch_index)
            except BaseException as exc:
                if base._oom(exc):
                    raise CohortOOM(f"{spec.job_id}: {exc}") from exc
                raise
            return spec.job_id, _adapter_complete(adapter, result)

        if workers == 1:
            results = [one(row) for row in active]
        else:
            with base.concurrent.futures.ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="cohort-model"
            ) as pool:
                futures = [pool.submit(one, row) for row in active]
                results = [future.result() for future in futures]
        completed.update(job_id for job_id, done in results if done)
        if device.startswith("cuda") and torch is not None:
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(device=device)
        return completed

    def _stage_rows(
        self,
        cohort: Cohort,
        rows: Sequence[tuple[ModelSpec, Any, Any]],
        *,
        next_batch: int,
        prior: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        staged = {key: dict(value) for key, value in prior.items()}
        for spec, adapter, _ in rows:
            directory = self._checkpoint_directory(cohort, spec, next_batch)
            directory.mkdir(parents=True, exist_ok=True)
            metadata = dict(adapter.save_checkpoint(directory) or {})
            try:
                relative = directory.relative_to(self.root)
                metadata["checkpoint_dir"] = relative.as_posix()
            except ValueError:
                metadata["checkpoint_dir"] = str(directory)
            metadata["committed_batch"] = int(next_batch)
            staged[spec.job_id] = metadata
        return staged

    def _full_resident(
        self,
        cohort: Cohort,
        digest: str,
        *,
        device: str,
        seed: int,
        state: Mapping[str, Any],
        dataset: DatasetAdapter,
    ) -> dict[str, Any]:
        start = int(state.get("next_batch", 0))
        checkpoints = {
            str(key): dict(value)
            for key, value in dict(state.get("model_checkpoints") or {}).items()
            if isinstance(value, Mapping)
        }
        completed_ids = {str(value) for value in state.get("completed_models", [])}
        specs = [spec for spec in cohort.models if spec.job_id not in completed_ids]
        rows = self._build_specs(cohort, specs, checkpoints, device)
        completed = start
        try:
            completed_ids.update(spec.job_id for spec, adapter, _ in rows if _adapter_complete(adapter))
            for batch_index, raw in enumerate(
                dataset.iter_raw_batches(
                    batch_size=cohort.uniform_batch_size,
                    start_batch=start,
                    seed=seed,
                ),
                start=start,
            ):
                active = [row for row in rows if row[0].job_id not in completed_ids and not _adapter_complete(row[1])]
                if not active:
                    break
                self.pressure.wait_until_safe(device)
                shared = SharedBatch(raw)
                try:
                    completed_ids.update(self._step_rows(active, shared, device, batch_index))
                finally:
                    shared.clear()
                next_batch = batch_index + 1
                # Immutable versioned checkpoints make state-file replacement the
                # only commit point for this shared batch.
                checkpoints = self._stage_rows(
                    cohort,
                    active,
                    next_batch=next_batch,
                    prior=checkpoints,
                )
                completed = next_batch
                if completed % self.checkpoint_every == 0:
                    self._write_state(
                        cohort,
                        digest,
                        next_batch=completed,
                        dataset=dataset,
                        checkpoints=checkpoints,
                        completed=completed_ids,
                        device=device,
                    )
            self._write_state(
                cohort,
                digest,
                next_batch=completed,
                dataset=dataset,
                checkpoints=checkpoints,
                completed=completed_ids,
                device=device,
            )
            return {
                "schema": RUNTIME_SCHEMA_V2,
                "cohort": cohort.index,
                "dataset": cohort.dataset_key,
                "backend": device,
                "batches_completed": completed,
                "uniform_batch_size": cohort.uniform_batch_size,
                "models": [spec.job_id for spec in cohort.models],
                "completed_models": sorted(completed_ids),
                "residency_mode": "full",
            }
        finally:
            for _, adapter, _ in rows:
                with contextlib.suppress(Exception):
                    adapter.close()

    def _require_window_safe(self, cohort: Cohort) -> None:
        errors: list[str] = []
        for spec in cohort.models:
            if spec.metadata.get("transactional_batch_safe") is not True:
                errors.append(f"{spec.job_id}: spec metadata lacks transactional_batch_safe=true")
                continue
            adapter = self.registry.model(spec)
            try:
                if getattr(adapter, "transactional_batch_safe", False) is not True:
                    errors.append(f"{spec.job_id}: adapter lacks transactional_batch_safe=true")
            finally:
                with contextlib.suppress(Exception):
                    adapter.close()
        if errors:
            raise CohortError(
                "residency-window mode is not source-proven safe:\n" + "\n".join(errors)
            )

    def _windowed(
        self,
        cohort: Cohort,
        digest: str,
        *,
        device: str,
        seed: int,
        state: Mapping[str, Any],
        dataset: DatasetAdapter,
    ) -> dict[str, Any]:
        self._require_window_safe(cohort)
        start = int(state.get("next_batch", 0))
        checkpoints = {
            str(key): dict(value)
            for key, value in dict(state.get("model_checkpoints") or {}).items()
            if isinstance(value, Mapping)
        }
        completed_ids = {str(value) for value in state.get("completed_models", [])}
        completed = start
        width = max(1, self.max_resident)
        for batch_index, raw in enumerate(
            dataset.iter_raw_batches(
                batch_size=cohort.uniform_batch_size,
                start_batch=start,
                seed=seed,
            ),
            start=start,
        ):
            active_specs = [spec for spec in cohort.models if spec.job_id not in completed_ids]
            if not active_specs:
                break
            self.pressure.wait_until_safe(device)
            shared = SharedBatch(raw)
            batch_checkpoints = {key: dict(value) for key, value in checkpoints.items()}
            batch_completed = set(completed_ids)
            try:
                for offset in range(0, len(active_specs), width):
                    window = active_specs[offset : offset + width]
                    rows = self._build_specs(cohort, window, checkpoints, device)
                    try:
                        batch_completed.update(self._step_rows(rows, shared, device, batch_index))
                        batch_checkpoints = self._stage_rows(
                            cohort,
                            rows,
                            next_batch=batch_index + 1,
                            prior=batch_checkpoints,
                        )
                    finally:
                        for _, adapter, _ in rows:
                            with contextlib.suppress(Exception):
                                adapter.close()
            finally:
                shared.clear()
            # Commit only after every window completed and staged successfully.
            completed = batch_index + 1
            checkpoints = batch_checkpoints
            completed_ids = batch_completed
            self._write_state(
                cohort,
                digest,
                next_batch=completed,
                dataset=dataset,
                checkpoints=checkpoints,
                completed=completed_ids,
                device=device,
            )
        self._write_state(
            cohort,
            digest,
            next_batch=completed,
            dataset=dataset,
            checkpoints=checkpoints,
            completed=completed_ids,
            device=device,
        )
        return {
            "schema": RUNTIME_SCHEMA_V2,
            "cohort": cohort.index,
            "dataset": cohort.dataset_key,
            "backend": device,
            "batches_completed": completed,
            "uniform_batch_size": cohort.uniform_batch_size,
            "models": [spec.job_id for spec in cohort.models],
            "completed_models": sorted(completed_ids),
            "residency_mode": "windowed",
            "max_resident_models": width,
            "transactional_batch_commit": True,
        }

    def _run_once(self, cohort: Cohort, digest: str, *, device: str, seed: int) -> dict[str, Any]:
        state = self._load_state(cohort, digest)
        dataset = self.registry.dataset(cohort)
        if isinstance(state.get("dataset_state"), Mapping) and state["dataset_state"]:
            dataset.load_state_dict(state["dataset_state"])
        use_windows = self.max_resident > 0 and self.max_resident < len(cohort.models)
        if use_windows:
            return self._windowed(
                cohort,
                digest,
                device=device,
                seed=seed,
                state=state,
                dataset=dataset,
            )
        return self._full_resident(
            cohort,
            digest,
            device=device,
            seed=seed,
            state=state,
            dataset=dataset,
        )


__all__ = [
    "AdapterRegistry",
    "BackendMode",
    "BackendProbe",
    "BackendUnavailable",
    "Cohort",
    "CohortError",
    "CohortExecutor",
    "CohortOOM",
    "CohortPlan",
    "DatasetAdapter",
    "ModelAdapter",
    "ModelSpec",
    "OVERLAP_DATASET_KEY",
    "PLAN_SCHEMA",
    "PressureGuard",
    "PressureSnapshot",
    "RUNTIME_SCHEMA_V2",
    "SCHEMA",
    "STATE_SCHEMA",
    "STATE_SCHEMA_V2",
    "SharedBatch",
    "array_namespace",
    "build_cohort_plan",
    "detect_backend",
    "subprocess_environment",
    "to_array",
]
