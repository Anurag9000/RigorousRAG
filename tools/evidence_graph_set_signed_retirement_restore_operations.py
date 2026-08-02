"""Read-only audit and conservative retention planning for restore intents."""

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
_MAX_LIMIT = 10_000


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


def _optional_digest(value: Any, label: str) -> str | None:
    return None if value is None else _digest(value, label)


def _holds(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("held_restore_ids must be an iterable.")
    rendered: set[str] = set()
    for value in values:
        rendered.add(_digest(value, "held_restore_id"))
        if len(rendered) > _MAX_HOLDS:
            raise ValueError("held restore IDs exceed the limit.")
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
        return "completed", False, False
    if value.state == "cancelled":
        return "cancelled", False, False
    raise RuntimeError("restore journal returned an unsupported state.")


@dataclass(frozen=True)
class SignedRetirementRestoreOperationalItem:
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    snapshot_record_count: int
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner_present: bool
    lease_expires_at: float | None
    lease_active: bool
    lease_expired: bool
    target_verification_digest: str | None
    failure_type: str | None
    updated_at: float
    completed_at: float | None
    classification: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "restore_id", _digest(self.restore_id, "restore_id"))
        object.__setattr__(
            self,
            "snapshot_digest",
            _digest(self.snapshot_digest, "snapshot_digest"),
        )
        object.__setattr__(
            self,
            "target_path_digest",
            _digest(self.target_path_digest, "target_path_digest"),
        )
        object.__setattr__(
            self,
            "snapshot_record_count",
            _integer(
                self.snapshot_record_count,
                "snapshot_record_count",
                1,
                _MAX_LIMIT,
            ),
        )
        object.__setattr__(self, "state", _identifier(self.state, "state", 30))
        object.__setattr__(self, "phase", _identifier(self.phase, "phase", 30))
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
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count exceeds max_attempts.")
        if not isinstance(self.lease_owner_present, bool):
            raise ValueError("lease_owner_present must be boolean.")
        lease_expires = (
            None
            if self.lease_expires_at is None
            else _timestamp(self.lease_expires_at, "lease_expires_at")
        )
        object.__setattr__(self, "lease_expires_at", lease_expires)
        if not isinstance(self.lease_active, bool) or not isinstance(
            self.lease_expired, bool
        ):
            raise ValueError("lease flags must be boolean.")
        if self.lease_active and self.lease_expired:
            raise ValueError("lease cannot be active and expired.")
        verification = _optional_digest(
            self.target_verification_digest,
            "target_verification_digest",
        )
        object.__setattr__(self, "target_verification_digest", verification)
        failure = (
            None
            if self.failure_type is None
            else _identifier(self.failure_type, "failure_type", 200)
        )
        object.__setattr__(self, "failure_type", failure)
        object.__setattr__(
            self,
            "updated_at",
            _timestamp(self.updated_at, "updated_at"),
        )
        completed = (
            None
            if self.completed_at is None
            else _timestamp(self.completed_at, "completed_at")
        )
        object.__setattr__(self, "completed_at", completed)
        classification = _identifier(self.classification, "classification", 100)
        if classification not in _CLASSIFICATIONS:
            raise ValueError("restore classification is unsupported.")
        object.__setattr__(self, "classification", classification)


@dataclass(frozen=True)
class SignedRetirementRestoreOperationalReport:
    owner_id: str
    state_filter: str | None
    snapshot_digest_filter: str | None
    target_path_digest_filter: str | None
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[SignedRetirementRestoreOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


@dataclass(frozen=True)
class SignedRetirementRestoreRetentionItem:
    restore_id: str
    snapshot_digest: str
    target_path_digest: str
    state: str
    phase: str
    completed_at: float
    age_seconds: float
    held: bool
    protected_as_latest_for_target: bool
    retention_candidate: bool
    reason: str


@dataclass(frozen=True)
class SignedRetirementRestoreRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_target: int
    include_completed: bool
    candidate_count: int
    items: tuple[SignedRetirementRestoreRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    journal_mutation_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


def audit_signed_retirement_restore_operations(
    *,
    owner_id: str,
    journal: Any,
    state: str | None = None,
    snapshot_digest: str | None = None,
    target_path_digest: str | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> SignedRetirementRestoreOperationalReport:
    owner = normalize_owner_id(owner_id)
    state_filter = None if state is None else _identifier(state, "state", 30)
    snapshot_filter = _optional_digest(snapshot_digest, "snapshot_digest")
    target_filter = _optional_digest(target_path_digest, "target_path_digest")
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal lacks the required read boundary.")
    values = tuple(journal.list(owner_id=owner, state=state_filter, limit=count))
    if len(values) >= count:
        raise RuntimeError(
            "restore operational audit reached the bounded result limit."
        )
    rendered: list[SignedRetirementRestoreOperationalItem] = []
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    seen: set[str] = set()
    for value in values:
        restore_id = _digest(value.restore_id, "restore_id")
        if restore_id in seen:
            raise RuntimeError("restore journal returned duplicate IDs.")
        seen.add(restore_id)
        if snapshot_filter is not None and value.snapshot_digest != snapshot_filter:
            continue
        if target_filter is not None and value.target_path_digest != target_filter:
            continue
        classification, active, expired = _classification(value, now=timestamp)
        counts[classification] += 1
        rendered.append(
            SignedRetirementRestoreOperationalItem(
                restore_id=restore_id,
                snapshot_digest=value.snapshot_digest,
                target_path_digest=value.target_path_digest,
                snapshot_record_count=value.snapshot_record_count,
                state=value.state,
                phase=value.phase,
                attempt_count=value.attempt_count,
                max_attempts=value.max_attempts,
                lease_owner_present=value.lease_owner is not None,
                lease_expires_at=value.lease_expires_at,
                lease_active=active,
                lease_expired=expired,
                target_verification_digest=value.target_verification_digest,
                failure_type=value.failure_type,
                updated_at=value.updated_at,
                completed_at=value.completed_at,
                classification=classification,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.restore_id))
    stable = {
        "scope": "rigorousrag-signed-retirement-restore-operational-audit-v1",
        "owner_id": owner,
        "state_filter": state_filter,
        "snapshot_digest_filter": snapshot_filter,
        "target_path_digest_filter": target_filter,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return SignedRetirementRestoreOperationalReport(
        owner_id=owner,
        state_filter=state_filter,
        snapshot_digest_filter=snapshot_filter,
        target_path_digest_filter=target_filter,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_signed_retirement_restore_retention(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    minimum_age_seconds: float = 180 * 24 * 60 * 60,
    retain_latest_per_target: int = 1,
    include_completed: bool = False,
    held_restore_ids: Iterable[str] | None = None,
    limit: int = _MAX_LIMIT,
) -> SignedRetirementRestoreRetentionPlan:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _timestamp(minimum_age_seconds, "minimum_age_seconds")
    latest_count = _integer(
        retain_latest_per_target,
        "retain_latest_per_target",
        1,
        100,
    )
    if not isinstance(include_completed, bool):
        raise ValueError("include_completed must be boolean.")
    held = _holds(held_restore_ids)
    count = _integer(limit, "limit", 1, _MAX_LIMIT)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal lacks the required read boundary.")
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("restore retention plan reached the bounded result limit.")
    seen: set[str] = set()
    terminal_by_target: dict[str, list[Any]] = {}
    for value in values:
        restore_id = _digest(value.restore_id, "restore_id")
        if restore_id in seen:
            raise RuntimeError("restore journal returned duplicate IDs.")
        seen.add(restore_id)
        if value.state in _TERMINAL_STATES:
            terminal_by_target.setdefault(value.target_path_digest, []).append(value)
    protected: set[str] = set()
    for target_values in terminal_by_target.values():
        ordered = sorted(
            target_values,
            key=lambda value: (
                -float(value.completed_at or 0.0),
                value.restore_id,
            ),
        )
        protected.update(
            value.restore_id for value in ordered[:latest_count]
        )

    rendered: list[SignedRetirementRestoreRetentionItem] = []
    for value in values:
        if value.state not in _TERMINAL_STATES or value.completed_at is None:
            continue
        completed = _timestamp(value.completed_at, "completed_at")
        age = max(0.0, timestamp - completed)
        is_held = value.restore_id in held
        is_latest = value.restore_id in protected
        eligible_state = value.state == "cancelled" or include_completed
        candidate = bool(
            age >= minimum_age
            and not is_held
            and not is_latest
            and eligible_state
        )
        if is_held:
            reason = "legal_hold"
        elif is_latest:
            reason = "latest_terminal_for_target"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif value.state == "completed" and not include_completed:
            reason = "completed_restores_retained_by_default"
        elif candidate:
            reason = "old_terminal_target_history_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            SignedRetirementRestoreRetentionItem(
                restore_id=value.restore_id,
                snapshot_digest=value.snapshot_digest,
                target_path_digest=value.target_path_digest,
                state=value.state,
                phase=value.phase,
                completed_at=completed,
                age_seconds=age,
                held=is_held,
                protected_as_latest_for_target=is_latest,
                retention_candidate=candidate,
                reason=reason,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.restore_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-signed-retirement-restore-retention-plan-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_target": latest_count,
        "include_completed": include_completed,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return SignedRetirementRestoreRetentionPlan(
        owner_id=owner,
        generated_at=timestamp,
        minimum_age_seconds=minimum_age,
        retain_latest_per_target=latest_count,
        include_completed=include_completed,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "SignedRetirementRestoreOperationalItem",
    "SignedRetirementRestoreOperationalReport",
    "SignedRetirementRestoreRetentionItem",
    "SignedRetirementRestoreRetentionPlan",
    "audit_signed_retirement_restore_operations",
    "plan_signed_retirement_restore_retention",
]
