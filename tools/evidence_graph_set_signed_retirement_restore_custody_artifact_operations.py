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
    selected_restore = (
        None if restore_id is None else _digest(restore_id, "restore_id")
    )
    selected_state = None if state is None else _identifier(state, "state", 30)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("artifact journal lacks the required read boundary.")
    values = tuple(
        journal.list(owner_id=owner, state=selected_state, limit=count)
    )
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
    latest_count = _integer(
        retain_latest_per_target,
        "retain_latest_per_target",
        1,
        100,
    )
    include = _boolean(include_completed, "include_completed")
    held = _holds(held_restore_ids)
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("artifact retention plan reached the bounded result limit.")
    seen: set[str] = set()
    terminal_by_target: dict[str, list[Any]] = {}
    for value in values:
        if value.artifact_id in seen:
            raise RuntimeError("artifact journal returned duplicate IDs.")
        seen.add(value.artifact_id)
        if value.state in _TERMINAL:
            terminal_by_target.setdefault(value.target_path_digest, []).append(value)
    protected: set[str] = set()
    for target_values in terminal_by_target.values():
        ordered = sorted(
            target_values,
            key=lambda value: (
                -float(value.completed_at or 0.0),
                value.artifact_id,
            ),
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
        eligible_state = value.state == "cancelled" or (
            value.state == "completed" and include
        )
        candidate = bool(
            eligible_state
            and age >= minimum_age
            and not is_held
            and not is_latest
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
