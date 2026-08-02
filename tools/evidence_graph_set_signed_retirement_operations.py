"""Read-only operational audit and conservative retention planning for retirements."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_set_signed_retirement_contracts import (
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


def _nonnegative(value: Any, label: str) -> float:
    selected = _timestamp(value, label)
    return selected


def _holds(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("held_retirement_ids must be an iterable.")
    rendered: set[str] = set()
    for value in values:
        rendered.add(_digest(value, "held_retirement_id"))
        if len(rendered) > _MAX_HOLDS:
            raise ValueError("held retirement IDs exceed the limit.")
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
    raise RuntimeError("retirement journal returned an unsupported state.")


@dataclass(frozen=True)
class SignedRetirementOperationalItem:
    retirement_id: str
    publication_operation_id: str
    graph_set_key: str
    state: str
    phase: str
    attempt_count: int
    max_attempts: int
    lease_owner_present: bool
    lease_expires_at: float | None
    lease_active: bool
    lease_expired: bool
    updated_at: float
    completed_at: float | None
    classification: str
    failure_type: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "retirement_id", _digest(self.retirement_id, "retirement_id"))
        object.__setattr__(
            self,
            "publication_operation_id",
            _digest(self.publication_operation_id, "publication_operation_id"),
        )
        object.__setattr__(
            self,
            "graph_set_key",
            _identifier(self.graph_set_key, "graph_set_key", 500),
        )
        object.__setattr__(self, "state", _identifier(self.state, "state", 30))
        object.__setattr__(self, "phase", _identifier(self.phase, "phase", 40))
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
        if not isinstance(self.lease_owner_present, bool):
            raise ValueError("lease_owner_present must be boolean.")
        lease_expires = None if self.lease_expires_at is None else _timestamp(
            self.lease_expires_at, "lease_expires_at"
        )
        object.__setattr__(self, "lease_expires_at", lease_expires)
        if not isinstance(self.lease_active, bool) or not isinstance(
            self.lease_expired, bool
        ):
            raise ValueError("lease flags must be boolean.")
        if self.lease_active and self.lease_expired:
            raise ValueError("lease cannot be active and expired.")
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, "updated_at"))
        completed = None if self.completed_at is None else _timestamp(
            self.completed_at, "completed_at"
        )
        object.__setattr__(self, "completed_at", completed)
        classification = _identifier(self.classification, "classification", 100)
        if classification not in _CLASSIFICATIONS:
            raise ValueError("operational classification is unsupported.")
        object.__setattr__(self, "classification", classification)
        failure = None if self.failure_type is None else _identifier(
            self.failure_type, "failure_type", 200
        )
        object.__setattr__(self, "failure_type", failure)


@dataclass(frozen=True)
class SignedRetirementOperationalReport:
    owner_id: str
    publication_operation_id: str | None
    generated_at: float
    item_count: int
    classification_counts: dict[str, int]
    items: tuple[SignedRetirementOperationalItem, ...]
    report_digest: str
    mutation_performed: bool = False
    source_text_returned: bool = False


@dataclass(frozen=True)
class SignedRetirementRetentionItem:
    retirement_id: str
    publication_operation_id: str
    state: str
    phase: str
    completed_at: float
    age_seconds: float
    held: bool
    protected_as_latest: bool
    retention_candidate: bool
    reason: str


@dataclass(frozen=True)
class SignedRetirementRetentionPlan:
    owner_id: str
    generated_at: float
    minimum_age_seconds: float
    retain_latest_per_operation: int
    include_completed: bool
    candidate_count: int
    items: tuple[SignedRetirementRetentionItem, ...]
    plan_digest: str
    deletion_performed: bool = False
    source_text_returned: bool = False


def audit_signed_retirement_operations(
    *,
    owner_id: str,
    journal: Any,
    publication_operation_id: str | None = None,
    now: float | None = None,
    limit: int = 1_000,
) -> SignedRetirementOperationalReport:
    owner = normalize_owner_id(owner_id)
    operation = None if publication_operation_id is None else _digest(
        publication_operation_id, "publication_operation_id"
    )
    timestamp = _timestamp(time.time() if now is None else now, "now")
    count = _integer(limit, "limit", 1, 10_000)
    if not callable(getattr(journal, "list", None)):
        raise ValueError("journal lacks the required read boundary.")
    values = tuple(
        journal.list(
            owner_id=owner,
            publication_operation_id=operation,
            limit=count,
        )
    )
    if len(values) >= count:
        raise RuntimeError(
            "retirement operational audit reached the bounded result limit."
        )
    rendered: list[SignedRetirementOperationalItem] = []
    counts = {name: 0 for name in sorted(_CLASSIFICATIONS)}
    seen: set[str] = set()
    for value in values:
        retirement_id = _digest(value.retirement_id, "retirement_id")
        if retirement_id in seen:
            raise RuntimeError("retirement journal returned duplicate IDs.")
        seen.add(retirement_id)
        classification, active, expired = _classification(value, now=timestamp)
        counts[classification] += 1
        rendered.append(
            SignedRetirementOperationalItem(
                retirement_id=retirement_id,
                publication_operation_id=value.publication_operation_id,
                graph_set_key=value.graph_set_key,
                state=value.state,
                phase=value.phase,
                attempt_count=value.attempt_count,
                max_attempts=value.max_attempts,
                lease_owner_present=value.lease_owner is not None,
                lease_expires_at=value.lease_expires_at,
                lease_active=active,
                lease_expired=expired,
                updated_at=value.updated_at,
                completed_at=value.completed_at,
                classification=classification,
                failure_type=value.failure_type,
            )
        )
    items = tuple(sorted(rendered, key=lambda item: item.retirement_id))
    stable = {
        "scope": "rigorousrag-signed-retirement-operational-audit-v1",
        "owner_id": owner,
        "publication_operation_id": operation,
        "generated_at": timestamp,
        "item_count": len(items),
        "classification_counts": counts,
        "items": [asdict(item) for item in items],
    }
    return SignedRetirementOperationalReport(
        owner_id=owner,
        publication_operation_id=operation,
        generated_at=timestamp,
        item_count=len(items),
        classification_counts=counts,
        items=items,
        report_digest=_canonical_digest(stable),
    )


def plan_signed_retirement_retention(
    *,
    owner_id: str,
    journal: Any,
    now: float | None = None,
    minimum_age_seconds: float = 180 * 24 * 60 * 60,
    retain_latest_per_operation: int = 1,
    include_completed: bool = False,
    held_retirement_ids: Iterable[str] | None = None,
    limit: int = 10_000,
) -> SignedRetirementRetentionPlan:
    owner = normalize_owner_id(owner_id)
    timestamp = _timestamp(time.time() if now is None else now, "now")
    minimum_age = _nonnegative(minimum_age_seconds, "minimum_age_seconds")
    latest_count = _integer(
        retain_latest_per_operation,
        "retain_latest_per_operation",
        1,
        100,
    )
    if not isinstance(include_completed, bool):
        raise ValueError("include_completed must be boolean.")
    held = _holds(held_retirement_ids)
    count = _integer(limit, "limit", 1, 10_000)
    values = tuple(journal.list(owner_id=owner, limit=count))
    if len(values) >= count:
        raise RuntimeError("retention plan reached the bounded result limit.")
    terminal_by_operation: dict[str, list[Any]] = {}
    for value in values:
        if value.state in _TERMINAL_STATES:
            terminal_by_operation.setdefault(
                value.publication_operation_id, []
            ).append(value)
    protected: set[str] = set()
    for operation_values in terminal_by_operation.values():
        ordered = sorted(
            operation_values,
            key=lambda value: (
                -float(value.completed_at or 0.0),
                value.retirement_id,
            ),
        )
        protected.update(
            value.retirement_id for value in ordered[:latest_count]
        )

    rendered: list[SignedRetirementRetentionItem] = []
    for value in values:
        if value.state not in _TERMINAL_STATES or value.completed_at is None:
            continue
        completed = _timestamp(value.completed_at, "completed_at")
        age = max(0.0, timestamp - completed)
        is_held = value.retirement_id in held
        is_latest = value.retirement_id in protected
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
            reason = "latest_terminal_for_operation"
        elif age < minimum_age:
            reason = "younger_than_minimum_age"
        elif value.state == "completed" and not include_completed:
            reason = "completed_retirements_retained_by_default"
        elif candidate:
            reason = "old_terminal_duplicate_candidate"
        else:
            reason = "not_retention_candidate"
        rendered.append(
            SignedRetirementRetentionItem(
                retirement_id=value.retirement_id,
                publication_operation_id=value.publication_operation_id,
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
    items = tuple(sorted(rendered, key=lambda item: item.retirement_id))
    candidate_count = sum(item.retention_candidate for item in items)
    stable = {
        "scope": "rigorousrag-signed-retirement-retention-plan-v1",
        "owner_id": owner,
        "generated_at": timestamp,
        "minimum_age_seconds": minimum_age,
        "retain_latest_per_operation": latest_count,
        "include_completed": include_completed,
        "candidate_count": candidate_count,
        "items": [asdict(item) for item in items],
    }
    return SignedRetirementRetentionPlan(
        owner_id=owner,
        generated_at=timestamp,
        minimum_age_seconds=minimum_age,
        retain_latest_per_operation=latest_count,
        include_completed=include_completed,
        candidate_count=candidate_count,
        items=items,
        plan_digest=_canonical_digest(stable),
    )


__all__ = [
    "SignedRetirementOperationalItem",
    "SignedRetirementOperationalReport",
    "SignedRetirementRetentionItem",
    "SignedRetirementRetentionPlan",
    "audit_signed_retirement_operations",
    "plan_signed_retirement_retention",
]
