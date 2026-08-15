"""Provider-neutral training lineage, replay identity, and continual-learning metrics."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def _text(value: Any, label: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split())
    if not cleaned:
        raise ValueError(f"{label} must be non-empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return cleaned


@dataclass(frozen=True)
class TrainingRequest:
    run_id: str
    parent_artifact_sha256: str
    dataset_sha256: str
    code_revision: str
    seed: int
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "parent_artifact_sha256", _digest(self.parent_artifact_sha256, "parent_artifact_sha256"))
        object.__setattr__(self, "dataset_sha256", _digest(self.dataset_sha256, "dataset_sha256"))
        object.__setattr__(self, "code_revision", _text(self.code_revision, "code_revision"))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.config, Mapping):
            raise ValueError("config must be a mapping")
        _canonical_json(dict(self.config))

    @property
    def config_sha256(self) -> str:
        return _sha256_hex(_canonical_json(dict(self.config)))

    @property
    def request_sha256(self) -> str:
        payload = {
            "run_id": self.run_id,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "dataset_sha256": self.dataset_sha256,
            "code_revision": self.code_revision,
            "seed": self.seed,
            "config_sha256": self.config_sha256,
        }
        return _sha256_hex(_canonical_json(payload))


@dataclass(frozen=True)
class TrainingOutcome:
    output_artifact_sha256: str
    evaluation_sha256: tuple[str, ...] = ()
    provider_run_ref: str = "local"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_artifact_sha256", _digest(self.output_artifact_sha256, "output_artifact_sha256"))
        object.__setattr__(self, "evaluation_sha256", tuple(_digest(item, "evaluation_sha256") for item in self.evaluation_sha256))
        object.__setattr__(self, "provider_run_ref", _text(self.provider_run_ref, "provider_run_ref"))


class TrainingProvider(Protocol):
    def train(self, request: TrainingRequest) -> TrainingOutcome: ...


@dataclass(frozen=True)
class TrainingLineage:
    request: TrainingRequest
    outcome: TrainingOutcome
    lineage_sha256: str

    @classmethod
    def bind(cls, request: TrainingRequest, outcome: TrainingOutcome) -> "TrainingLineage":
        payload = {
            "request_sha256": request.request_sha256,
            "output_artifact_sha256": outcome.output_artifact_sha256,
            "evaluation_sha256": list(outcome.evaluation_sha256),
            "provider_run_ref": outcome.provider_run_ref,
        }
        return cls(request, outcome, _sha256_hex(_canonical_json(payload)))


@dataclass(frozen=True)
class ReplayManifest:
    owner_id: str
    example_digests: tuple[str, ...]
    manifest_sha256: str


def build_privacy_safe_replay_manifest(
    *, owner_id: str, example_ids: Sequence[str], secret_salt: bytes
) -> ReplayManifest:
    """Bind replay membership without storing raw example IDs or the HMAC key."""

    owner = _text(owner_id, "owner_id")
    if not isinstance(secret_salt, bytes) or len(secret_salt) < 16:
        raise ValueError("secret_salt must contain at least 16 bytes")
    digests: list[str] = []
    for example_id in example_ids:
        item = _text(example_id, "example_id", maximum=2048)
        message = f"{owner}\x00{item}".encode("utf-8")
        digests.append(hmac.new(secret_salt, message, hashlib.sha256).hexdigest())
    if len(set(digests)) != len(digests):
        raise ValueError("duplicate replay example")
    ordered = tuple(sorted(digests))
    manifest = _sha256_hex(_canonical_json({"owner_id": owner, "example_digests": ordered}))
    return ReplayManifest(owner, ordered, manifest)


@dataclass(frozen=True)
class ContinualTransferReport:
    per_task_forgetting: tuple[float, ...]
    average_forgetting: float
    per_task_forward_transfer: tuple[float, ...]
    average_forward_transfer: float


def continual_transfer_metrics(
    accuracy_matrix: Sequence[Sequence[float]], *, baseline: Sequence[float]
) -> ContinualTransferReport:
    """Compute final forgetting and pre-learning forward transfer from a stage×task matrix."""

    rows = [tuple(float(value) for value in row) for row in accuracy_matrix]
    if not rows:
        raise ValueError("accuracy_matrix must be non-empty")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("accuracy_matrix must be rectangular and non-empty")
    if len(rows) < width:
        raise ValueError("accuracy_matrix needs at least one training stage per task")
    if len(baseline) != width:
        raise ValueError("baseline width must match task count")
    values = [value for row in rows for value in row] + [float(value) for value in baseline]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("metrics must be finite")

    final = rows[-1]
    forgetting: list[float] = []
    for task in range(width - 1):
        best_after_learning = max(rows[stage][task] for stage in range(task, len(rows)))
        forgetting.append(best_after_learning - final[task])

    forward: list[float] = []
    baseline_values = [float(value) for value in baseline]
    for task in range(1, width):
        forward.append(rows[task - 1][task] - baseline_values[task])

    avg_forgetting = sum(forgetting) / len(forgetting) if forgetting else 0.0
    avg_forward = sum(forward) / len(forward) if forward else 0.0
    return ContinualTransferReport(tuple(forgetting), avg_forgetting, tuple(forward), avg_forward)
