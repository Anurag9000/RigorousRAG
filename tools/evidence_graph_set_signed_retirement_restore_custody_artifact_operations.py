"""Read-only custody artifact audit and conservative retention planning."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
    deterministic_signed_retirement_restore_id,
)
from tools.security import normalize_owner_id

_STATES = frozenset(
    {"planned", "running", "completed", "orphaned", "failed", "cancelled"}
)
_CLASSIFICATIONS = frozenset(
    {
        "planned_ready",
        "running_active",
        "running_expired_reclaimable",
        "running_expired_exhausted",
        "failed_retryable",
        "failed_exhausted",
        "completed_pair",
        "orphan_backup_without_receipt",
        "orphan_receipt_without_backup",
        "orphan_artifact_collision",
        "cancelled",
    }
)
_TERMINAL = frozenset({"completed", "orphaned", "cancelled"})
_MAX_LIMIT = 10_000
_MAX_HOLDS = 100_000


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _optional_identifier(value: Any, label: str, maximum: int = 200) -> str | None:
    return None if value is None else _identifier(value, label, maximum)


def _holds(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("held_restore_ids must be an iterable.")
    result: set[str] = set()
    for value in values:
        result.add(_digest(value, "held_restore_id"))
        if len(result) > _MAX_HOLDS:
            raise ValueError("held restore IDs exceed the limit.")
    return frozenset(result)


def _restore_id(value: Any) -> str:
    return deterministic_signed_retirement_restore_id(
        owner_id=value.owner_id,
        snapshot_digest=value.snapshot_digest,
        target_path_digest=value.target_path_digest,
    )


def _classification(value: Any, *, now: float) -> tuple[str, bool, bool]:
    if value.state == "planned":
        return "planned_ready", False, False
    if value.state == "running":
        active = bool(
            value.lease_expires_at is not None and value.lease_expires_at > now
        )
        if active:
            return "running_active", True, False
        return (
            "running_expired_reclaimable"
            if value.attempt_count < value.max_attempts
            else "running_expired_exhausted",
            False,
            True,
        )
    if value.state == "failed":
        return (
            "failed_retryable"
            if value.attempt_count < value.max_attempts
            else "failed_exhausted",
            False,
            False,
        )
    if value.state == "completed":
        return "completed_pair", False, False
    if value.state == "orphaned":
        mapping = {
            "backup_without_receipt": "orphan_backup_without_receipt",
            "receipt_without_backup": "orphan_receipt_without_backup",
            "artifact_collision": "orphan_artifact_collision",
        }
        try:
            return mapping[value.disposition], False, False
        except KeyError as exc:
            raise RuntimeError("orphaned artifact has unsupported disposition.") from exc
    if value.state == "cancelled":
        return "cancelled", False, False
    raise RuntimeError("artifact journal returned an unsupported state.")


@dataclass(frozen=True)
class RestoreCustodyArtifactOperationalItem:
    artifact_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    backup_path_digest: str
    receipt_path_digest: str
    state: str
    phase: str
    classification: str
    attempt_count: int
    max_attempts: int
    lease_owner_present: bool
    lease_expires_at: float | None
    lease_active: bool
    lease_expired: bool
    disposition: str | None
    backup_sha256: str | None
    backup_size_bytes: int | None
    receipt_digest: str | None
    failure_type: str | None
    updated_at: float
    completed_at: float | None

    def __post_init__(self) -> None:
        for field in (
            "artifact_id",
            "restore_id",
            "snapshot_digest",
            "target_path_digest",
            "backup_path_digest",
            "receipt_path_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        state = _identifier(self.state, "state", 30)
        phase = _identifier(self.phase, "phase", 40)
        classification = _identifier(self.classification, "classification", 80)
        if state not in _STATES or classification not in _CLASSIFICATIONS:
            raise ValueError("artifact operational state or classification is unsupported.")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "classification", classification)
        object.__setattr__(
            self,
            "attempt_count",
            _integer(self.attempt_count, "attempt_count", 0, 1_000_000),
        )
        object.__setattr__(
            self,
            "max_attempts",
            _integer(self.max_attempts, "max_attempts", 1, 1_000_000),
        )
        for field in ("lease_owner_present", "lease_active", "lease_expired"):
            object.__setattr__(self, field, _boolean(getattr(self, field), field))
        if self.lease_active and self.lease_expired:
            raise ValueError("artifact lease cannot be active and expired.")
        lease = (
            None
            if self.lease_expires_at is None
            else _timestamp(self.lease_expires_at, "lease_expires_at")
        )
        object.__setattr__(self, "lease_expires_at", lease)
        object.__setattr__(
            self,
            "disposition",
            _optional_identifier(self.disposition, "disposition", 80),
        )
        object.__setattr__(
            self,
            "backup_sha256",
            _optional_digest(self.backup_sha256, "backup_sha256"),
        )
        size = (
            None
            if self.backup_size_bytes is None
            else _integer(
                self.backup_size_bytes,
                "backup_size_bytes",
                1,
                1024 * 1024 * 1024 * 1024,
            )
        )
        object.__setattr__(self, "backup_size_bytes", size)
        object.__setattr__(
            self,
            "receipt_digest",
            _optional_digest(self.receipt_digest, "receipt_digest"),
        )
        object.__setattr__(
            self,
            "failure_type",
            _optional_identifier(self.failure_type, "failure_type", 200),
        )
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated_at"))
        completed = (
            None
            if self.completed_at is None
            else _timestamp(self.completed_at, "completed_at")
        )
        object.__setattr__(self, "completed_at", completed)


@dataclass(frozen=True)
class RestoreCustodyArtifactOperationalReport:
    owner_id: str
    restore_id: str | None
    state: str | None
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[RestoreCustodyArtifactOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    artifact_deletion_performed: bool = False
    artifact_overwrite_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        restore = _optional_digest(self.restore_id, "restore_id")
        state = None if self.state is None else _identifier(self.state, "state", 30)
        if state is not None and state not in _STATES:
            raise ValueError("artifact state is unsupported.")
        generated = _timestamp(self.generated_at, "generated_at")
        count = _integer(self.item_count, "item_count", 0, _MAX_LIMIT)
        if count != len(self.items):
            raise ValueError("artifact report count differs from items.")
        expected_counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
        seen: set[str] = set()
        for item in self.items:
            if not isinstance(item, RestoreCustodyArtifactOperationalItem):
                raise ValueError("artifact operational item is invalid.")
            if item.artifact_id in seen:
                raise ValueError("artifact report contains duplicate IDs.")
            seen.add(item.artifact_id)
            expected_counts[item.classification] += 1
        if dict(self.classification_counts) != expected_counts:
            raise ValueError("artifact classification counts differ from items.")
        for field in (
            "mutation_performed",
            "artifact_deletion_performed",
            "artifact_overwrite_performed",
            "source_text_returned",
            "raw_path_returned",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"{field} must be false.")
        stable = {
            "scope": "rigorousrag-restore-custody-artifact-audit-v1",
            "owner_id": owner,
            "restore_id": restore,
            "state": state,
            "generated_at": generated,
            "item_count": count,
            "classification_counts": expected_counts,
            "items": [asdict(item) for item in self.items],
        }
        digest = _digest(self.report_digest, "report_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("report_digest differs from artifact report.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "restore_id", restore)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "item_count", count)
        object.__setattr__(self, "classification_counts", expected_counts)
        object.__setattr__(self, "report_digest", digest)


@dataclass(frozen=True)
class RestoreCustodyArtifactRetentionItem:
    artifact_id: str
    restore_id: str
    target_path_digest: str
    state: str
    disposition: str | None
    completed_at: float
    age_seconds: float
    held: bool
    protected_as_latest: bool
    retention_candidate: bool
    reason: str

    def __post_init__(self) -> None:
        for field in ("artifact_id", "restore_id", "target_path_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        state = _identifier(self.state, "state", 30)
        if state not in _TERMINAL:
            raise ValueError("retention item must be terminal.")
        object.__setattr__(self, "state", state)
        object.__setattr__(
            self,
            "disposition",
            _optional_identifier(self.disposition, "disposition", 80),
        )
        object.__setattr__(
            self,
            "completed_at",
            _timestamp(self.completed_at, "completed_at"),
        )
        object.__setattr__(
            self,
            "age_seconds",
            _timestamp(self.age_seconds, "age_seconds"),
        )
        for field in ("held", "protected_as_latest", "retention_candidate"):
            object.__setattr__(self, field, _boolean(getattr(self, field), field))
        reason = _identifier(self.reason, "reason", 100)
        object.__setattr__(self, "reason", reason)
        if self.retention_candidate and (
            state == "orphaned" or self.held or self.protected_as_latest
        ):
            raise ValueError("invalid artifact retention candidate.")


@dataclass(frozen=True)
class RestoreCustodyArtifactRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_target: int
    include_completed: bool
    candidate_count: int
    items: tuple[RestoreCustodyArtifactRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    artifact_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_path_returned: bool = False

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        generated = _timestamp(self.generated_at, "generated_at")
        minimum_age = _timestamp(self.minimum_age_seconds, "minimum_age_seconds")
        latest = _integer(self.retain_latest_per_target, "retain_latest_per_target", 1, 100)
        include = _boolean(self.include_completed, "include_completed")
        candidates = _integer(self.candidate_count, "candidate_count", 0, _MAX_LIMIT)
        if candidates != sum(item.retention_candidate for item in self.items):
            raise ValueError("artifact candidate count differs from items.")
        seen: set[str] = set()
        for item in self.items:
            if not isinstance(item, RestoreCustodyArtifactRetentionItem):
                raise ValueError("artifact retention item is invalid.")
            if item.artifact_id in seen:
                raise ValueError("artifact retention plan contains duplicate IDs.")
            seen.add(item.artifact_id)
        for field in (
            "deletion_performed",
            "artifact_mutation_performed",
            "source_text_returned",
            "raw_path_returned",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"{field} must be false.")
        stable = {
            "scope": "rigorousrag-restore-custody-artifact-retention-v1",
            "owner_id": owner,
            "generated_at": generated,
            "minimum_age_seconds": minimum_age,
            "retain_latest_per_target": latest,
            "include_completed": include,
            "candidate_count": candidates,
            "items": [asdict(item) for item in self.items],
        }
        digest = _digest(self.plan_digest, "plan_digest")
        if digest != _canonical_digest(stable):
            raise ValueError("plan_digest differs from artifact retention plan.")
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "generated_at", generated)
        object.__setattr__(self, "minimum_age_seconds", minimum_age)
        object.__setattr__(self, "retain_latest_per_target", latest)
        object.__setattr__(self, "include_completed", include)
        object.__setattr__(self, "candidate_count", candidates)
        object.__setattr__(self, "plan_digest", digest)


def audit_restore_custody_artifacts(
    *,
    owner_id: str,
    journal: Any,
    restore_id: str | None = None,
    state: str | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> RestoreCustodyArtifactOperationalReport:
    owner = normalize_owner_id(owner_id)
    selected_restore = _optional_digest(restore_id, "restore_id")
    selected_state = None if state is None else _identifier(state, "state", 30)
    if selected_state is not None and selected_state not in _STATES:
        raise ValueError("artifact state is unsupported.")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("artifact journal lacks the required read boundary.")
    values = tuple(journal.list(owner_id=owner, state=selected_state, limit=count))
    if len(values) >= count:
        raise RuntimeError("artifact audit reached the bounded result limit.")
    rendered: list[RestoreCustodyArtifactOperationalItem] = []
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    seen: set[str] = set()
    for value in values:
        artifact_id = _digest(value.artifact_id, "artifact_id")
        if artifact_id in seen:
            raise RuntimeError("artifact journal returned duplicate IDs.")
        seen.add(artifact_id)
        derived_restore = _restore_id(value)
        if selected_restore is not None and derived_restore != selected_restore:
            continue
        classification, active, expired = _classification(value, now=timestamp)
        counts[classification] += 1
        rendered.append(
            RestoreCustodyArtifactOperationalItem(
                artifact_id=artifact_id,
                restore_id=derived_restore,
                snapshot_digest=value.snapshot_digest,
                target_path_digest=value.target_path_digest,
                backup_path_digest=value.backup_path_digest,
                receipt_path_digest=value.receipt_path_digest,
                state=value.state,
                phase=value.phase,
                classification=classification,
                attempt_count=value.attempt_count,
                max_attempts=value.max_attempts,
                lease_owner_present=value.lease_owner is not None,
                lease_expires_at=value.lease_expires_at,
                lease_active=active,
                lease_expired=expired,
                disposition=value.disposition,
                backup_sha256=value.backup_sha256,
                backup_size_bytes=value.backup_size_bytes,
                receipt_digest=value.receipt_digest,
                failure_type=value.failure_type,
                updated_at=value.updated_at,
                completed_at=value.completed_at,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.artifact_id))
    stable = {
        "scope": "rigorousrag-restore-custody-artifact-audit-v1",
        "owner_id": owner,
        "restore_id": selected_restore,
        "state": selected_state,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return RestoreCustodyArtifactOperationalReport(
        owner_id=owner,
        restore_id=selected_restore,
        state=selected_state,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_restore_custody_artifact_retention(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    minimum_age_seconds: float = 365 * 24 * 60 * 60,
    retain_latest_per_target: int = 1,
    include_completed: bool = False,
    held_restore_ids: Iterable[str] | None = None,
    limit: int = 10_000,
) -> RestoreCustodyArtifactRetentionPlan:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _timestamp(minimum_age_seconds, "minimum_age_seconds")
    latest_count = _integer(retain_latest_per_target, "retain_latest_per_target", 1, 100)
    include = _boolean(include_completed, "include_completed")
    held = _holds(held_restore_ids)
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("artifact retention plan reached the bounded result limit.")
    seen: set[str] = set()
    terminal_by_target: dict[str, list[Any]] = {}
    for value in values:
        artifact_id = _digest(value.artifact_id, "artifact_id")
        if artifact_id in seen:
            raise RuntimeError("artifact journal returned duplicate IDs.")
        seen.add(artifact_id)
        if value.state in _TERMINAL:
            terminal_by_target.setdefault(value.target_path_digest, []).append(value)
    protected: set[str] = set()
    for target_values in terminal_by_target.values():
        ordered = sorted(
            target_values,
            key=lambda value: (-float(value.completed_at or 0.0), value.artifact_id),
        )
        protected.update(value.artifact_id for value in ordered[:latest_count])
    rendered: list[RestoreCustodyArtifactRetentionItem] = []
    for value in values:
        if value.state not in _TERMINAL or value.completed_at is None:
            continue
        restore = _restore_id(value)
        completed = _timestamp(value.completed_at, "completed_at")
        age = max(0.0, timestamp - completed)
        is_held = restore in held
        is_latest = value.artifact_id in protected
        eligible = value.state == "cancelled" or (
            value.state == "completed" and include
        )
        candidate = bool(
            eligible and age >= minimum_age and not is_held and not is_latest
        )
        if value.state == "orphaned":
            reason = "orphan_evidence_never_candidate"
        elif is_held:
            reason = "legal_hold"
        elif is_latest:
            reason = "latest_terminal_for_target"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif value.state == "completed" and not include:
            reason = "completed_pairs_retained_by_default"
        elif candidate:
            reason = "old_terminal_duplicate_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            RestoreCustodyArtifactRetentionItem(
                artifact_id=value.artifact_id,
                restore_id=restore,
                target_path_digest=value.target_path_digest,
                state=value.state,
                disposition=value.disposition,
                completed_at=completed,
                age_seconds=age,
                held=is_held,
                protected_as_latest=is_latest,
                retention_candidate=candidate,
                reason=reason,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.artifact_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-restore-custody-artifact-retention-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_target": latest_count,
        "include_completed": include,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return RestoreCustodyArtifactRetentionPlan(
        owner_id=owner,
        generated_at=timestamp,
        minimum_age_seconds=minimum_age,
        retain_latest_per_target=latest_count,
        include_completed=include,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "RestoreCustodyArtifactOperationalItem",
    "RestoreCustodyArtifactOperationalReport",
    "RestoreCustodyArtifactRetentionItem",
    "RestoreCustodyArtifactRetentionPlan",
    "audit_restore_custody_artifacts",
    "plan_restore_custody_artifact_retention",
]
