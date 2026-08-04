"""Read-only operational audit and conservative retention planning for deletions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _integer,
    _timestamp,
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
        "completed",
        "cancelled",
    }
)
_TERMINAL_STATES = frozenset({"completed", "cancelled"})
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


def _holds(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("held_deletion_ids must be an iterable.")
    rendered: set[str] = set()
    for value in values:
        rendered.add(_digest(value, "held_deletion_id"))
        if len(rendered) > _MAX_HOLDS:
            raise ValueError("held deletion IDs exceed the limit.")
    return frozenset(rendered)


def _classification(value: Any, *, now: float) -> tuple[str, bool, bool]:
    if value.state == "planned":
        return "planned_ready", False, False
    if value.state == "running":
        active = bool(
            value.lease_expires_at is not None and value.lease_expires_at > now
        )
        if active:
            return "running_active", True, False
        reclaimable = value.attempt_count < value.max_attempts
        return (
            "running_expired_reclaimable"
            if reclaimable
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
        return "completed", False, False
    if value.state == "cancelled":
        return "cancelled", False, False
    raise RuntimeError("deletion journal returned an unsupported state.")


@dataclass(frozen=True)
class RestoreDeletionOperationalItem:
    deletion_id: str
    authorization_id: str
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner_present: bool
    lease_expires_at: float | None
    lease_active: bool
    lease_expired: bool
    marker_digest: str | None
    tombstone_digest: str | None
    custody_manifest_digest: str | None
    failure_type: str | None
    updated_at: float
    completed_at: float | None
    classification: str


@dataclass(frozen=True)
class RestoreDeletionOperationalReport:
    owner_id: str
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[RestoreDeletionOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    restore_row_deleted: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


@dataclass(frozen=True)
class RestoreDeletionRetentionItem:
    deletion_id: str
    authorization_id: str
    restore_id: str
    state: str
    phase: str
    completed_at: float
    age_seconds: float
    held: bool
    protected_as_latest: bool
    retention_candidate: bool
    reason: str


@dataclass(frozen=True)
class RestoreDeletionRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_restore: int
    include_completed: bool
    candidate_count: int
    items: tuple[RestoreDeletionRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    compaction_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


def audit_restore_deletion_operations(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    limit: int = 1_000,
) -> RestoreDeletionOperationalReport:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal lacks the required read boundary.")
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError(
            "deletion operational audit reached the bounded result limit."
        )
    seen: set[str] = set()
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    rendered: list[RestoreDeletionOperationalItem] = []
    for value in values:
        deletion_id = _digest(value.deletion_id, "deletion_id")
        if deletion_id in seen:
            raise RuntimeError("deletion journal returned duplicate IDs.")
        seen.add(deletion_id)
        classification, active, expired = _classification(
            value, now=timestamp
        )
        counts[classification] += 1
        rendered.append(
            RestoreDeletionOperationalItem(
                deletion_id=deletion_id,
                authorization_id=_digest(
                    value.authorization_id, "authorization_id"
                ),
                restore_id=_digest(value.restore_id, "restore_id"),
                snapshot_digest=_digest(
                    value.snapshot_digest, "snapshot_digest"
                ),
                target_path_digest=_digest(
                    value.target_path_digest, "target_path_digest"
                ),
                state=value.state,
                phase=value.phase,
                attempt_count=value.attempt_count,
                max_attempts=value.max_attempts,
                lease_owner_present=value.lease_owner is not None,
                lease_expires_at=value.lease_expires_at,
                lease_active=active,
                lease_expired=expired,
                marker_digest=value.marker_digest,
                tombstone_digest=value.tombstone_digest,
                custody_manifest_digest=value.custody_manifest_digest,
                failure_type=value.failure_type,
                updated_at=value.updated_at,
                completed_at=value.completed_at,
                classification=classification,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.deletion_id))
    stable = {
        "scope": "rigorousrag-restore-deletion-operational-audit-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return RestoreDeletionOperationalReport(
        owner_id=owner,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_restore_deletion_retention(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    minimum_age_seconds: float = 365 * 24 * 60 * 60,
    retain_latest_per_restore: int = 1,
    include_completed: bool = False,
    held_deletion_ids: Iterable[str] | None = None,
    limit: int = 10_000,
) -> RestoreDeletionRetentionPlan:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _timestamp(
        minimum_age_seconds, "minimum_age_seconds"
    )
    latest_count = _integer(
        retain_latest_per_restore,
        "retain_latest_per_restore",
        1,
        100,
    )
    if not isinstance(include_completed, bool):
        raise ValueError("include_completed must be boolean.")
    held = _holds(held_deletion_ids)
    count = _integer(limit, "limit", 1, 10_000)
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("deletion retention plan reached the bounded limit.")
    seen: set[str] = set()
    terminal_by_restore: dict[str, list[Any]] = {}
    for value in values:
        deletion_id = _digest(value.deletion_id, "deletion_id")
        if deletion_id in seen:
            raise RuntimeError("deletion journal returned duplicate IDs.")
        seen.add(deletion_id)
        if value.state in _TERMINAL_STATES:
            terminal_by_restore.setdefault(value.restore_id, []).append(value)
    protected: set[str] = set()
    for restore_values in terminal_by_restore.values():
        ordered = sorted(
            restore_values,
            key=lambda value: (
                -float(value.completed_at or 0.0),
                value.deletion_id,
            ),
        )
        protected.update(
            value.deletion_id for value in ordered[:latest_count]
        )
    rendered: list[RestoreDeletionRetentionItem] = []
    for value in values:
        if value.state not in _TERMINAL_STATES or value.completed_at is None:
            continue
        completed = _timestamp(value.completed_at, "completed_at")
        age = max(0.0, timestamp - completed)
        is_held = value.deletion_id in held
        is_latest = value.deletion_id in protected
        eligible_state = value.state == "cancelled" or include_completed
        candidate = bool(
            age >= minimum_age
            and not is_held
            and not is_latest
            and eligible_state
        )
        if is_held:
            reason = "operator_hold"
        elif is_latest:
            reason = "latest_terminal_for_restore"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif value.state == "completed" and not include_completed:
            reason = "completed_deletions_retained_by_default"
        elif candidate:
            reason = "old_terminal_duplicate_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            RestoreDeletionRetentionItem(
                deletion_id=value.deletion_id,
                authorization_id=value.authorization_id,
                restore_id=value.restore_id,
                state=value.state,
                phase=value.phase,
                completed_at=completed,
                age_seconds=age,
                held=is_held,
                protected_as_latest=is_latest,
                retention_candidate=candidate,
                reason=reason,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.deletion_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-restore-deletion-retention-plan-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_restore": latest_count,
        "include_completed": include_completed,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return RestoreDeletionRetentionPlan(
        owner_id=owner,
        generated_at=timestamp,
        minimum_age_seconds=minimum_age,
        retain_latest_per_restore=latest_count,
        include_completed=include_completed,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "RestoreDeletionOperationalItem",
    "RestoreDeletionOperationalReport",
    "RestoreDeletionRetentionItem",
    "RestoreDeletionRetentionPlan",
    "audit_restore_deletion_operations",
    "plan_restore_deletion_retention",
]
