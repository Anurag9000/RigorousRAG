"""Atomic, content-verified PyTorch training checkpoints and exact resume state.

A checkpoint captures more than model weights: optimizer, scheduler, AMP scaler,
trainer/stage cursor, Python/PyTorch/CUDA RNG, sampler state and collator state are bound
into one manifest.  Model tensors are stored with safetensors by default.  Optimizer and
scheduler state use ``torch.save`` because PyTorch exposes those states as nested tensor
objects; loading uses ``weights_only=True`` and is restricted to a caller-selected local
checkpoint root.

No checkpoint is created or loaded merely by importing this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
except Exception:  # pragma: no cover - optional training dependency.
    torch = None  # type: ignore[assignment]

_HEX = frozenset("0123456789abcdef")
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024 * 1024


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("checkpoint execution requires the optional PyTorch dependency")


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in _HEX for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def _git_commit(value: Any) -> str:
    commit = _identifier(value, "source_commit", 64).lower()
    if len(commit) not in {40, 64} or any(ch not in _HEX for ch in commit):
        raise ValueError("source_commit must be a full SHA-1 or SHA-256 Git object id")
    return commit


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    selected = Path(path).resolve(strict=True)
    if not selected.is_file() or selected.is_symlink():
        raise ValueError("checkpoint artifact must be a regular non-symlink file")
    if selected.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("checkpoint artifact exceeds safety bound")
    digest = hashlib.sha256()
    with selected.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_tuple(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__tuple__": [_json_tuple(item) for item in value]}
    if isinstance(value, list):
        return [_json_tuple(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_tuple(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported RNG JSON type: {type(value).__name__}")


def _restore_tuple(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"__tuple__"}:
        return tuple(_restore_tuple(item) for item in value["__tuple__"])
    if isinstance(value, list):
        return [_restore_tuple(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_tuple(item) for key, item in value.items()}
    return value


def capture_rng_state() -> dict[str, Any]:
    _require_torch()
    result: dict[str, Any] = {
        "python": _json_tuple(random.getstate()),
        "torch_cpu": base64.b64encode(bytes(torch.get_rng_state().tolist())).decode("ascii"),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = [
            base64.b64encode(bytes(state.tolist())).decode("ascii") for state in torch.cuda.get_rng_state_all()
        ]
    try:
        import numpy as np

        np_state = np.random.get_state()
        result["numpy"] = {
            "bit_generator": np_state[0],
            "state": np_state[1].tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        }
    except Exception:
        result["numpy"] = None
    return result


def restore_rng_state(state: Mapping[str, Any]) -> None:
    _require_torch()
    random.setstate(_restore_tuple(state["python"]))
    cpu_bytes = base64.b64decode(state["torch_cpu"], validate=True)
    torch.set_rng_state(torch.tensor(list(cpu_bytes), dtype=torch.uint8))
    cuda_states = state.get("torch_cuda") or []
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        decoded = [
            torch.tensor(list(base64.b64decode(value, validate=True)), dtype=torch.uint8, device="cpu")
            for value in cuda_states
        ]
        if len(decoded) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from checkpoint RNG state")
        torch.cuda.set_rng_state_all(decoded)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("checkpoint contains NumPy RNG state but NumPy is unavailable") from exc
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype="uint32"),
                int(numpy_state["pos"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )


@dataclass(frozen=True)
class TrainerCursor:
    stage_index: int
    epoch: int
    batch_in_epoch: int
    global_step: int
    optimizer_step: int
    examples_seen: int
    tokens_seen: int = 0

    def __post_init__(self) -> None:
        for name in (
            "stage_index",
            "epoch",
            "batch_in_epoch",
            "global_step",
            "optimizer_step",
            "examples_seen",
            "tokens_seen",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class TrainerState:
    run_id: str
    cursor: TrainerCursor
    best_metric: float | None = None
    best_checkpoint_digest: str | None = None
    early_stopping_bad_steps: int = 0
    stage_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        if not isinstance(self.cursor, TrainerCursor):
            raise ValueError("cursor must be TrainerCursor")
        if self.best_metric is not None:
            value = float(self.best_metric)
            if value != value or abs(value) == float("inf"):
                raise ValueError("best_metric must be finite")
            object.__setattr__(self, "best_metric", value)
        if self.best_checkpoint_digest is not None:
            object.__setattr__(
                self,
                "best_checkpoint_digest",
                _sha256(self.best_checkpoint_digest, "best_checkpoint_digest"),
            )
        if isinstance(self.early_stopping_bad_steps, bool) or not isinstance(self.early_stopping_bad_steps, int) or self.early_stopping_bad_steps < 0:
            raise ValueError("early_stopping_bad_steps must be non-negative")
        if self.stage_name is not None:
            object.__setattr__(self, "stage_name", _identifier(self.stage_name, "stage_name", 500))


@dataclass(frozen=True)
class CheckpointArtifact:
    filename: str
    sha256: str
    byte_size: int
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "filename", _identifier(self.filename, "filename", 500))
        if "/" in self.filename or "\\" in self.filename or self.filename in {".", ".."}:
            raise ValueError("checkpoint artifact filename must be a basename")
        object.__setattr__(self, "sha256", _sha256(self.sha256, "artifact sha256"))
        if isinstance(self.byte_size, bool) or not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("artifact byte_size must be non-negative")
        object.__setattr__(self, "kind", _identifier(self.kind, "artifact kind", 200))


@dataclass(frozen=True)
class TensorCheckpointManifest:
    version: int
    run_id: str
    source_commit: str
    training_config_digest: str
    dataset_manifest_digest: str
    model_architecture: str
    trainer_state_digest: str
    artifacts: tuple[CheckpointArtifact, ...]
    parent_checkpoint_digest: str | None = None
    stage_boundary: bool = False
    metric_snapshot: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported tensor checkpoint manifest version")
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        object.__setattr__(self, "source_commit", _git_commit(self.source_commit))
        object.__setattr__(
            self,
            "training_config_digest",
            _sha256(self.training_config_digest, "training_config_digest"),
        )
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            _sha256(self.dataset_manifest_digest, "dataset_manifest_digest"),
        )
        object.__setattr__(self, "model_architecture", _identifier(self.model_architecture, "model_architecture"))
        object.__setattr__(self, "trainer_state_digest", _sha256(self.trainer_state_digest, "trainer_state_digest"))
        if not self.artifacts or len(self.artifacts) > 100:
            raise ValueError("checkpoint manifest requires bounded artifacts")
        if len({artifact.filename for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("checkpoint artifact filenames must be unique")
        if self.parent_checkpoint_digest is not None:
            object.__setattr__(
                self,
                "parent_checkpoint_digest",
                _sha256(self.parent_checkpoint_digest, "parent_checkpoint_digest"),
            )
        if not isinstance(self.stage_boundary, bool):
            raise ValueError("stage_boundary must be boolean")
        metrics: dict[str, float] = {}
        for key, raw in self.metric_snapshot.items():
            value = float(raw)
            if value != value or abs(value) == float("inf"):
                raise ValueError("checkpoint metrics must be finite")
            metrics[_identifier(key, "metric name", 300)] = value
        object.__setattr__(self, "metric_snapshot", metrics)

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


@dataclass(frozen=True)
class LoadedCheckpoint:
    path: Path
    manifest: TensorCheckpointManifest
    trainer_state: TrainerState
    sampler_state: Mapping[str, Any]
    collator_state: Mapping[str, Any]


@dataclass(frozen=True)
class CheckpointRetentionPolicy:
    keep_last: int = 3
    keep_every_steps: int | None = None
    keep_stage_boundaries: bool = True
    keep_best: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.keep_last, bool) or not isinstance(self.keep_last, int) or self.keep_last < 0:
            raise ValueError("keep_last must be non-negative")
        if self.keep_every_steps is not None and (
            isinstance(self.keep_every_steps, bool)
            or not isinstance(self.keep_every_steps, int)
            or self.keep_every_steps <= 0
        ):
            raise ValueError("keep_every_steps must be positive or None")
        if not isinstance(self.keep_stage_boundaries, bool) or not isinstance(self.keep_best, bool):
            raise ValueError("retention flags must be boolean")


class CheckpointManager:
    def __init__(self, root: str | Path) -> None:
        selected = Path(root).expanduser().resolve()
        selected.mkdir(parents=True, exist_ok=True)
        if selected.is_symlink():
            raise ValueError("checkpoint root may not be a symlink")
        self.root = selected

    @staticmethod
    def _save_model_safetensors(model: Any, path: Path) -> None:
        _require_torch()
        try:
            from safetensors.torch import save_file
        except Exception as exc:
            raise RuntimeError(
                "safe model checkpointing requires safetensors; install optional training dependencies"
            ) from exc
        module = model.module if hasattr(model, "module") else model
        state = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in module.state_dict().items()
            if torch.is_tensor(value)
        }
        save_file(state, str(path), metadata={"format": "pt"})

    @staticmethod
    def _load_model_safetensors(model: Any, path: Path, *, strict: bool = True) -> None:
        _require_torch()
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("loading model checkpoints requires safetensors") from exc
        module = model.module if hasattr(model, "module") else model
        state = load_file(str(path), device="cpu")
        incompatible = module.load_state_dict(state, strict=strict)
        if strict and (getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None)):
            raise RuntimeError("strict model checkpoint restoration reported incompatible keys")

    @staticmethod
    def _torch_save_state(value: Any, path: Path) -> None:
        _require_torch()
        torch.save(value, path)

    @staticmethod
    def _torch_load_state(path: Path) -> Any:
        _require_torch()
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "safe optimizer/scheduler checkpoint loading requires a PyTorch version supporting weights_only=True"
            ) from exc

    def save(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        trainer_state: TrainerState,
        sampler_state: Mapping[str, Any],
        collator_state: Mapping[str, Any],
        source_commit: str,
        training_config_digest: str,
        dataset_manifest_digest: str,
        model_architecture: str,
        parent_checkpoint_digest: str | None = None,
        stage_boundary: bool = False,
        metric_snapshot: Mapping[str, float] | None = None,
    ) -> TensorCheckpointManifest:
        _require_torch()
        if not isinstance(trainer_state, TrainerState):
            raise ValueError("trainer_state must be TrainerState")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.root))
        try:
            model_path = temporary / "model.safetensors"
            optimizer_path = temporary / "optimizer.pt"
            scheduler_path = temporary / "scheduler.pt"
            scaler_path = temporary / "scaler.pt"
            trainer_path = temporary / "trainer_state.json"
            sampler_path = temporary / "sampler_state.json"
            collator_path = temporary / "collator_state.json"
            rng_path = temporary / "rng_state.json"

            self._save_model_safetensors(model, model_path)
            self._torch_save_state(optimizer.state_dict(), optimizer_path)
            if scheduler is not None:
                self._torch_save_state(scheduler.state_dict(), scheduler_path)
            if scaler is not None:
                self._torch_save_state(scaler.state_dict(), scaler_path)
            _atomic_json(trainer_path, asdict(trainer_state))
            _atomic_json(sampler_path, dict(sampler_state))
            _atomic_json(collator_path, dict(collator_state))
            _atomic_json(rng_path, capture_rng_state())

            paths = [
                (model_path, "model"),
                (optimizer_path, "optimizer"),
                (trainer_path, "trainer_state"),
                (sampler_path, "sampler_state"),
                (collator_path, "collator_state"),
                (rng_path, "rng_state"),
            ]
            if scheduler is not None:
                paths.append((scheduler_path, "scheduler"))
            if scaler is not None:
                paths.append((scaler_path, "amp_scaler"))
            artifacts = tuple(
                CheckpointArtifact(path.name, sha256_file(path), path.stat().st_size, kind) for path, kind in paths
            )
            manifest = TensorCheckpointManifest(
                version=1,
                run_id=trainer_state.run_id,
                source_commit=source_commit,
                training_config_digest=training_config_digest,
                dataset_manifest_digest=dataset_manifest_digest,
                model_architecture=model_architecture,
                trainer_state_digest=canonical_digest(asdict(trainer_state)),
                artifacts=artifacts,
                parent_checkpoint_digest=parent_checkpoint_digest,
                stage_boundary=stage_boundary,
                metric_snapshot=metric_snapshot or {},
            )
            _atomic_json(temporary / "manifest.json", asdict(manifest))
            destination = self.root / manifest.digest
            if destination.exists():
                shutil.rmtree(temporary)
                existing = self.read_manifest(destination)
                if existing.digest != manifest.digest:
                    raise RuntimeError("content-addressed checkpoint directory collision")
            else:
                os.replace(temporary, destination)
            _atomic_json(
                self.root / "latest.json",
                {"checkpoint_digest": manifest.digest, "global_step": trainer_state.cursor.global_step},
            )
            if stage_boundary:
                _atomic_json(
                    self.root / f"stage-{trainer_state.cursor.stage_index:04d}.json",
                    {"checkpoint_digest": manifest.digest},
                )
            if trainer_state.best_checkpoint_digest == manifest.digest:
                _atomic_json(self.root / "best.json", {"checkpoint_digest": manifest.digest})
            return manifest
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    @staticmethod
    def _decode_trainer_state(value: Mapping[str, Any]) -> TrainerState:
        cursor_value = value["cursor"]
        return TrainerState(
            run_id=value["run_id"],
            cursor=TrainerCursor(**cursor_value),
            best_metric=value.get("best_metric"),
            best_checkpoint_digest=value.get("best_checkpoint_digest"),
            early_stopping_bad_steps=int(value.get("early_stopping_bad_steps", 0)),
            stage_name=value.get("stage_name"),
        )

    def read_manifest(self, checkpoint_path: str | Path) -> TensorCheckpointManifest:
        path = Path(checkpoint_path).resolve(strict=True)
        if path.parent != self.root or not path.is_dir() or path.is_symlink():
            raise ValueError("checkpoint path must be an immediate non-symlink child of configured root")
        payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        artifacts = tuple(CheckpointArtifact(**value) for value in payload["artifacts"])
        return TensorCheckpointManifest(
            version=int(payload["version"]),
            run_id=payload["run_id"],
            source_commit=payload["source_commit"],
            training_config_digest=payload["training_config_digest"],
            dataset_manifest_digest=payload["dataset_manifest_digest"],
            model_architecture=payload["model_architecture"],
            trainer_state_digest=payload["trainer_state_digest"],
            artifacts=artifacts,
            parent_checkpoint_digest=payload.get("parent_checkpoint_digest"),
            stage_boundary=bool(payload.get("stage_boundary", False)),
            metric_snapshot=payload.get("metric_snapshot") or {},
        )

    def verify(self, checkpoint_digest: str) -> tuple[Path, TensorCheckpointManifest]:
        selected_digest = _sha256(checkpoint_digest, "checkpoint_digest")
        path = (self.root / selected_digest).resolve(strict=True)
        manifest = self.read_manifest(path)
        if manifest.digest != selected_digest:
            raise RuntimeError("checkpoint directory name does not match manifest digest")
        for artifact in manifest.artifacts:
            artifact_path = (path / artifact.filename).resolve(strict=True)
            if artifact_path.parent != path:
                raise RuntimeError("checkpoint artifact escaped checkpoint directory")
            if artifact_path.stat().st_size != artifact.byte_size:
                raise RuntimeError(f"checkpoint artifact size mismatch: {artifact.filename}")
            if sha256_file(artifact_path) != artifact.sha256:
                raise RuntimeError(f"checkpoint artifact digest mismatch: {artifact.filename}")
        return path, manifest

    def load(
        self,
        checkpoint_digest: str,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        expected_source_commit: str,
        expected_training_config_digest: str,
        expected_dataset_manifest_digest: str,
        expected_model_architecture: str,
        strict_model: bool = True,
        restore_rng: bool = True,
    ) -> LoadedCheckpoint:
        path, manifest = self.verify(checkpoint_digest)
        if manifest.source_commit != _git_commit(expected_source_commit):
            raise ValueError("checkpoint source commit does not match requested resume source")
        if manifest.training_config_digest != _sha256(expected_training_config_digest, "training config digest"):
            raise ValueError("checkpoint training config differs")
        if manifest.dataset_manifest_digest != _sha256(expected_dataset_manifest_digest, "dataset manifest digest"):
            raise ValueError("checkpoint dataset manifest differs")
        if manifest.model_architecture != _identifier(expected_model_architecture, "model architecture"):
            raise ValueError("checkpoint model architecture differs")
        self._load_model_safetensors(model, path / "model.safetensors", strict=strict_model)
        optimizer.load_state_dict(self._torch_load_state(path / "optimizer.pt"))
        if scheduler is not None and (path / "scheduler.pt").exists():
            scheduler.load_state_dict(self._torch_load_state(path / "scheduler.pt"))
        if scaler is not None and (path / "scaler.pt").exists():
            scaler.load_state_dict(self._torch_load_state(path / "scaler.pt"))
        trainer_payload = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        trainer_state = self._decode_trainer_state(trainer_payload)
        if canonical_digest(asdict(trainer_state)) != manifest.trainer_state_digest:
            raise RuntimeError("trainer state digest does not match checkpoint manifest")
        sampler_state = json.loads((path / "sampler_state.json").read_text(encoding="utf-8"))
        collator_state = json.loads((path / "collator_state.json").read_text(encoding="utf-8"))
        if restore_rng:
            restore_rng_state(json.loads((path / "rng_state.json").read_text(encoding="utf-8")))
        return LoadedCheckpoint(path, manifest, trainer_state, sampler_state, collator_state)

    def resolve_pointer(self, pointer: str = "latest") -> str:
        selected = _identifier(pointer, "checkpoint pointer", 200)
        path = self.root / f"{selected}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _sha256(payload["checkpoint_digest"], "checkpoint_digest")

    def prune(self, policy: CheckpointRetentionPolicy) -> tuple[str, ...]:
        """Delete only unpinned content-addressed checkpoints under an explicit policy."""

        if not isinstance(policy, CheckpointRetentionPolicy):
            raise ValueError("policy must be CheckpointRetentionPolicy")
        checkpoints: list[tuple[str, TensorCheckpointManifest, int]] = []
        for child in self.root.iterdir():
            if not child.is_dir() or child.is_symlink() or len(child.name) != 64 or any(ch not in _HEX for ch in child.name):
                continue
            manifest = self.read_manifest(child)
            trainer_payload = json.loads((child / "trainer_state.json").read_text(encoding="utf-8"))
            step = int(trainer_payload["cursor"]["global_step"])
            checkpoints.append((child.name, manifest, step))
        checkpoints.sort(key=lambda item: (item[2], item[0]))
        keep: set[str] = {value[0] for value in checkpoints[-policy.keep_last :]} if policy.keep_last else set()
        if policy.keep_stage_boundaries:
            keep.update(digest for digest, manifest, _ in checkpoints if manifest.stage_boundary)
        if policy.keep_every_steps is not None:
            keep.update(digest for digest, _, step in checkpoints if step % policy.keep_every_steps == 0)
        if policy.keep_best and (self.root / "best.json").exists():
            keep.add(self.resolve_pointer("best"))
        if (self.root / "latest.json").exists():
            keep.add(self.resolve_pointer("latest"))
        deleted: list[str] = []
        for digest, _, _ in checkpoints:
            if digest in keep:
                continue
            shutil.rmtree(self.root / digest)
            deleted.append(digest)
        return tuple(deleted)


__all__ = [
    "CheckpointArtifact",
    "CheckpointManager",
    "CheckpointRetentionPolicy",
    "LoadedCheckpoint",
    "TensorCheckpointManifest",
    "TrainerCursor",
    "TrainerState",
    "canonical_digest",
    "capture_rng_state",
    "restore_rng_state",
    "sha256_file",
]
