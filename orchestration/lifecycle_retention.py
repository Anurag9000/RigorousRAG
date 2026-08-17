"""Store-neutral lifecycle retention, compaction and privacy-safe correlation export.

This module closes the operational lifecycle gap without coupling to one journal schema.
Concrete lifecycle/job stores implement the small protocols below.  Planning is separate
from mutation, legal holds are mandatory inputs, compaction is optimistic-revision
guarded, and exported rows contain identifiers/state/timing/digests only—never source
paths, queries or document text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence


def _id(value: Any, label: str, limit: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > limit or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{label} is invalid")
    return value


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()).hexdigest()


class TerminalState(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TerminalOperation:
    operation_id: str
    state: TerminalState
    revision: int
    created_at: datetime
    finished_at: datetime
    owner_scope_digest: str
    job_id: str | None = None
    generation_id: str | None = None
    error_type: str | None = None
    correlation_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation_id", _id(self.operation_id, "operation_id"))
        if not isinstance(self.state, TerminalState):
            object.__setattr__(self, "state", TerminalState(self.state))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError("revision must be positive")
        created, finished = _utc(self.created_at, "created_at"), _utc(self.finished_at, "finished_at")
        if finished < created:
            raise ValueError("finished_at precedes created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "finished_at", finished)
        for name in ("owner_scope_digest", "correlation_digest"):
            value = getattr(self, name)
            if value is not None:
                value = _id(value, name, 64).lower()
                if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"{name} must be SHA-256")
                object.__setattr__(self, name, value)
        for name in ("job_id", "generation_id", "error_type"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _id(value, name))


class LifecycleRetentionBackend(Protocol):
    def terminal_operations_before(self, before: datetime, *, limit: int) -> Sequence[TerminalOperation]: ...
    def is_under_legal_hold(self, operation_id: str) -> bool: ...
    def compact(self, operation_id: str, *, expected_revision: int, tombstone_digest: str) -> bool: ...


@dataclass(frozen=True)
class RetentionPolicy:
    completed_days: int = 90
    failed_days: int = 180
    cancelled_days: int = 90
    batch_limit: int = 1000

    def __post_init__(self) -> None:
        for name in ("completed_days", "failed_days", "cancelled_days"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.batch_limit, bool) or not isinstance(self.batch_limit, int) or not 1 <= self.batch_limit <= 100000:
            raise ValueError("batch_limit is invalid")

    def age_for(self, state: TerminalState) -> timedelta:
        days = {TerminalState.COMPLETED: self.completed_days, TerminalState.FAILED: self.failed_days, TerminalState.CANCELLED: self.cancelled_days}[state]
        return timedelta(days=days)


@dataclass(frozen=True)
class CompactionCandidate:
    operation_id: str
    expected_revision: int
    finished_at: datetime
    state: TerminalState
    tombstone_digest: str


@dataclass(frozen=True)
class CompactionPlan:
    as_of: datetime
    candidates: tuple[CompactionCandidate, ...]
    held_operation_count: int
    retained_by_age_count: int

    @property
    def digest(self) -> str:
        return _digest(asdict(self))


def plan_compaction(backend: LifecycleRetentionBackend, policy: RetentionPolicy, *, now: datetime) -> CompactionPlan:
    instant = _utc(now, "now")
    oldest_window = max(policy.completed_days, policy.failed_days, policy.cancelled_days)
    rows = backend.terminal_operations_before(instant - timedelta(days=oldest_window), limit=policy.batch_limit)
    candidates: list[CompactionCandidate] = []
    held = retained = 0
    for row in rows:
        if not isinstance(row, TerminalOperation):
            raise ValueError("backend returned a non-TerminalOperation")
        if instant - row.finished_at < policy.age_for(row.state):
            retained += 1
            continue
        if backend.is_under_legal_hold(row.operation_id):
            held += 1
            continue
        tombstone = _digest({"operation_id": row.operation_id, "revision": row.revision, "state": row.state.value, "finished_at": row.finished_at.isoformat(), "owner_scope_digest": row.owner_scope_digest})
        candidates.append(CompactionCandidate(row.operation_id, row.revision, row.finished_at, row.state, tombstone))
    return CompactionPlan(instant, tuple(candidates), held, retained)


def apply_compaction(backend: LifecycleRetentionBackend, plan: CompactionPlan) -> tuple[str, ...]:
    compacted: list[str] = []
    for candidate in plan.candidates:
        if backend.is_under_legal_hold(candidate.operation_id):
            continue
        if backend.compact(candidate.operation_id, expected_revision=candidate.expected_revision, tombstone_digest=candidate.tombstone_digest):
            compacted.append(candidate.operation_id)
    return tuple(compacted)


@dataclass(frozen=True)
class CorrelationRow:
    operation_id: str
    state: str
    created_at: str
    finished_at: str
    duration_ms: int
    owner_scope_digest: str
    job_id: str | None
    generation_id: str | None
    error_type: str | None
    correlation_digest: str


def privacy_safe_correlation_rows(operations: Sequence[TerminalOperation]) -> tuple[CorrelationRow, ...]:
    rows: list[CorrelationRow] = []
    for operation in operations:
        correlation = operation.correlation_digest or _digest({"operation_id": operation.operation_id, "job_id": operation.job_id, "generation_id": operation.generation_id})
        rows.append(CorrelationRow(
            operation.operation_id,
            operation.state.value,
            operation.created_at.isoformat(),
            operation.finished_at.isoformat(),
            max(0, int((operation.finished_at - operation.created_at).total_seconds() * 1000)),
            operation.owner_scope_digest,
            operation.job_id,
            operation.generation_id,
            operation.error_type,
            correlation,
        ))
    return tuple(rows)


def correlation_export_manifest(rows: Sequence[CorrelationRow]) -> Mapping[str, Any]:
    payload = [asdict(row) for row in rows]
    return {"schema_version": 1, "row_count": len(payload), "content_sha256": _digest(payload)}


__all__ = ["CompactionCandidate", "CompactionPlan", "CorrelationRow", "LifecycleRetentionBackend", "RetentionPolicy", "TerminalOperation", "TerminalState", "apply_compaction", "correlation_export_manifest", "plan_compaction", "privacy_safe_correlation_rows"]
