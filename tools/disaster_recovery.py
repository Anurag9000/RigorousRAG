"""Backup integrity, recovery-objective, canary and rollback control primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import Enum

from tools.supply_chain import sha256_bytes


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, order=True)
class BackupArtifact:
    name: str
    sha256: str
    size_bytes: int
    created_at: float


@dataclass(frozen=True)
class BackupManifest:
    artifacts: tuple[BackupArtifact, ...]
    manifest_sha256: str


def build_backup_manifest(files: Mapping[str, bytes], *, created_at: float) -> BackupManifest:
    artifacts = tuple(
        sorted(
            (
                BackupArtifact(name, sha256_bytes(content), len(content), float(created_at))
                for name, content in files.items()
            ),
            key=lambda item: item.name,
        )
    )
    digest = sha256_bytes(_canonical_json([asdict(item) for item in artifacts]))
    return BackupManifest(artifacts, digest)


def verify_backup_manifest(manifest: BackupManifest, files: Mapping[str, bytes]) -> bool:
    expected = build_backup_manifest(
        files,
        created_at=min((item.created_at for item in manifest.artifacts), default=0.0),
    )
    return expected == manifest


@dataclass(frozen=True)
class RecoveryObjective:
    max_rpo_seconds: float
    max_rto_seconds: float

    def __post_init__(self) -> None:
        if self.max_rpo_seconds < 0.0 or self.max_rto_seconds < 0.0:
            raise ValueError("recovery objectives must not be negative")


@dataclass(frozen=True)
class RestoreRehearsal:
    incident_at: float
    backup_at: float
    restore_started_at: float
    restore_completed_at: float
    required_artifacts: tuple[str, ...]
    restored_artifacts: tuple[str, ...]
    integrity_ok: bool


@dataclass(frozen=True)
class RecoveryDecision:
    ready: bool
    rpo_seconds: float
    rto_seconds: float
    reason_codes: tuple[str, ...]


def evaluate_recovery(
    rehearsal: RestoreRehearsal,
    objective: RecoveryObjective,
) -> RecoveryDecision:
    rpo = max(0.0, rehearsal.incident_at - rehearsal.backup_at)
    rto = max(0.0, rehearsal.restore_completed_at - rehearsal.incident_at)
    reasons: list[str] = []
    if rehearsal.restore_started_at < rehearsal.incident_at:
        reasons.append("restore_started_before_incident")
    if rehearsal.restore_completed_at < rehearsal.restore_started_at:
        reasons.append("restore_completion_precedes_start")
    if rpo > objective.max_rpo_seconds:
        reasons.append("rpo_exceeded")
    if rto > objective.max_rto_seconds:
        reasons.append("rto_exceeded")
    missing = set(rehearsal.required_artifacts) - set(rehearsal.restored_artifacts)
    if missing:
        reasons.append("restore_artifacts_missing")
    if not rehearsal.integrity_ok:
        reasons.append("restore_integrity_failed")
    return RecoveryDecision(not reasons, rpo, rto, tuple(reasons))


@dataclass(frozen=True)
class CanaryThresholds:
    min_samples: int = 100
    max_error_rate: float = 0.01
    max_p95_latency_ms: float = 1000.0
    min_quality_score: float = 0.0

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0.0 <= self.max_error_rate <= 1.0:
            raise ValueError("max_error_rate must be between zero and one")
        if self.max_p95_latency_ms <= 0.0:
            raise ValueError("max_p95_latency_ms must be positive")


@dataclass(frozen=True)
class CanaryObservation:
    samples: int
    error_rate: float
    p95_latency_ms: float
    quality_score: float


class CanaryAction(str, Enum):
    HOLD = "hold"
    PROMOTE = "promote"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class CanaryDecision:
    action: CanaryAction
    reason_codes: tuple[str, ...]


def evaluate_canary(
    observation: CanaryObservation,
    thresholds: CanaryThresholds | None = None,
) -> CanaryDecision:
    selected = thresholds or CanaryThresholds()
    if observation.samples < selected.min_samples:
        return CanaryDecision(CanaryAction.HOLD, ("insufficient_samples",))
    reasons: list[str] = []
    if observation.error_rate > selected.max_error_rate:
        reasons.append("error_rate_exceeded")
    if observation.p95_latency_ms > selected.max_p95_latency_ms:
        reasons.append("latency_budget_exceeded")
    if observation.quality_score < selected.min_quality_score:
        reasons.append("quality_floor_missed")
    if reasons:
        return CanaryDecision(CanaryAction.ROLLBACK, tuple(reasons))
    return CanaryDecision(CanaryAction.PROMOTE, ())


class RollbackState(str, Enum):
    PREPARED = "prepared"
    APPLIED = "applied"
    COMPLETED = "completed"


@dataclass(frozen=True)
class RollbackAction:
    action_id: str
    resource: str
    from_version: str
    to_version: str
    state: RollbackState = RollbackState.PREPARED


def prepare_rollback(resource: str, from_version: str, to_version: str) -> RollbackAction:
    if not resource.strip() or not from_version.strip() or not to_version.strip():
        raise ValueError("rollback identifiers must be non-empty")
    payload = {
        "resource": resource.strip(),
        "from_version": from_version.strip(),
        "to_version": to_version.strip(),
    }
    action_id = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return RollbackAction(action_id, payload["resource"], payload["from_version"], payload["to_version"])


def advance_rollback(action: RollbackAction, next_state: RollbackState) -> RollbackAction:
    order: Sequence[RollbackState] = (
        RollbackState.PREPARED,
        RollbackState.APPLIED,
        RollbackState.COMPLETED,
    )
    current_index = order.index(action.state)
    target_index = order.index(next_state)
    if target_index == current_index:
        return action
    if target_index != current_index + 1:
        raise ValueError("rollback state transitions must be sequential")
    return replace(action, state=next_state)


__all__ = [
    "BackupArtifact",
    "BackupManifest",
    "CanaryAction",
    "CanaryDecision",
    "CanaryObservation",
    "CanaryThresholds",
    "RecoveryDecision",
    "RecoveryObjective",
    "RestoreRehearsal",
    "RollbackAction",
    "RollbackState",
    "advance_rollback",
    "build_backup_manifest",
    "evaluate_canary",
    "evaluate_recovery",
    "prepare_rollback",
    "verify_backup_manifest",
]
