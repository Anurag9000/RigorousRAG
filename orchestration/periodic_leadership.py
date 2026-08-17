"""Fenced cross-process leadership contracts for periodic reconciliation jobs.

Existing RigorousRAG lease providers can implement ``LeadershipLeaseBackend``.  This
module adds the missing scheduler semantics on top: stable job identities, monotonic
fencing tokens, bounded lease durations, due-time calculation, renewal, and explicit
completion/failure recording.  No thread, process, database or timer starts on import.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Protocol, Sequence

_MAX_SECONDS = 365 * 24 * 60 * 60


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class PeriodicJobSpec:
    job_id: str
    interval_seconds: int
    lease_seconds: int
    max_runtime_seconds: int
    jitter_seconds: int = 0
    enabled: bool = True
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _identifier(self.job_id, "job_id"))
        for name, minimum in (
            ("interval_seconds", 1),
            ("lease_seconds", 1),
            ("max_runtime_seconds", 1),
            ("jitter_seconds", 0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _MAX_SECONDS:
                raise ValueError(f"{name} is invalid")
        if self.lease_seconds > self.max_runtime_seconds:
            raise ValueError("lease_seconds may not exceed max_runtime_seconds")
        if self.jitter_seconds >= self.interval_seconds:
            raise ValueError("jitter_seconds must be smaller than interval_seconds")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        metadata = self.metadata or {}
        if len(metadata) > 1_000:
            raise ValueError("metadata is too large")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in metadata.items()
            },
        )


@dataclass(frozen=True)
class LeadershipLease:
    job_id: str
    holder_id: str
    fencing_token: int
    acquired_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        for name in ("job_id", "holder_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token < 1:
            raise ValueError("fencing_token must be a positive integer")
        acquired = _utc(self.acquired_at, "acquired_at")
        expires = _utc(self.expires_at, "expires_at")
        if expires <= acquired:
            raise ValueError("lease expiry must be after acquisition")
        object.__setattr__(self, "acquired_at", acquired)
        object.__setattr__(self, "expires_at", expires)

    def is_valid_at(self, now: datetime) -> bool:
        instant = _utc(now, "now")
        return self.acquired_at <= instant < self.expires_at


@dataclass(frozen=True)
class JobRunRecord:
    job_id: str
    fencing_token: int
    scheduled_at: datetime
    started_at: datetime
    completed_at: datetime | None
    success: bool | None
    error_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _identifier(self.job_id, "job_id"))
        if isinstance(self.fencing_token, bool) or not isinstance(self.fencing_token, int) or self.fencing_token < 1:
            raise ValueError("fencing_token must be positive")
        scheduled = _utc(self.scheduled_at, "scheduled_at")
        started = _utc(self.started_at, "started_at")
        if started < scheduled:
            raise ValueError("started_at may not precede scheduled_at")
        object.__setattr__(self, "scheduled_at", scheduled)
        object.__setattr__(self, "started_at", started)
        if self.completed_at is not None:
            completed = _utc(self.completed_at, "completed_at")
            if completed < started:
                raise ValueError("completed_at may not precede started_at")
            object.__setattr__(self, "completed_at", completed)
        if self.completed_at is None and self.success is not None:
            raise ValueError("unfinished run may not have success state")
        if self.completed_at is not None and self.success is None:
            raise ValueError("finished run requires success state")
        if self.error_type is not None:
            object.__setattr__(self, "error_type", _identifier(self.error_type, "error_type", 500))
        if self.success is True and self.error_type is not None:
            raise ValueError("successful run may not have error_type")


class LeadershipLeaseBackend(Protocol):
    """Database/distributed lease backend; implementations must fence stale holders."""

    def try_acquire(
        self,
        job_id: str,
        holder_id: str,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> LeadershipLease | None: ...

    def renew(
        self,
        lease: LeadershipLease,
        *,
        now: datetime,
        lease_seconds: int,
    ) -> LeadershipLease: ...

    def release(self, lease: LeadershipLease, *, now: datetime) -> None: ...

    def assert_fence(self, job_id: str, fencing_token: int) -> None: ...


class JobRunStore(Protocol):
    def latest(self, job_id: str) -> JobRunRecord | None: ...

    def record_started(self, record: JobRunRecord) -> None: ...

    def record_finished(self, record: JobRunRecord) -> None: ...


@dataclass(frozen=True)
class DueJob:
    spec: PeriodicJobSpec
    due_at: datetime


def deterministic_jitter_seconds(job_id: str, interval_anchor: datetime, maximum: int) -> int:
    """Stable non-random jitter derived from job id and UTC interval anchor."""

    if maximum <= 0:
        return 0
    import hashlib

    anchor = _utc(interval_anchor, "interval_anchor").isoformat()
    digest = hashlib.sha256(f"{job_id}\0{anchor}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (maximum + 1)


def next_due_time(spec: PeriodicJobSpec, latest: JobRunRecord | None, *, now: datetime) -> datetime:
    instant = _utc(now, "now")
    if latest is None:
        anchor = instant
    else:
        anchor = latest.completed_at or latest.started_at
        anchor = _utc(anchor, "latest run anchor") + timedelta(seconds=spec.interval_seconds)
    jitter = deterministic_jitter_seconds(spec.job_id, anchor, spec.jitter_seconds)
    return anchor + timedelta(seconds=jitter)


def due_jobs(
    specs: Sequence[PeriodicJobSpec],
    store: JobRunStore,
    *,
    now: datetime,
) -> tuple[DueJob, ...]:
    instant = _utc(now, "now")
    if len(specs) > 100_000:
        raise ValueError("too many periodic job specs")
    seen: set[str] = set()
    due: list[DueJob] = []
    for spec in specs:
        if spec.job_id in seen:
            raise ValueError("periodic job ids must be unique")
        seen.add(spec.job_id)
        if not spec.enabled:
            continue
        due_at = next_due_time(spec, store.latest(spec.job_id), now=instant)
        if due_at <= instant:
            due.append(DueJob(spec, due_at))
    return tuple(sorted(due, key=lambda value: (value.due_at, value.spec.job_id)))


@dataclass(frozen=True)
class AcquiredJob:
    due: DueJob
    lease: LeadershipLease
    started_record: JobRunRecord


def acquire_due_job(
    due: DueJob,
    backend: LeadershipLeaseBackend,
    store: JobRunStore,
    *,
    holder_id: str,
    now: datetime,
) -> AcquiredJob | None:
    instant = _utc(now, "now")
    lease = backend.try_acquire(
        due.spec.job_id,
        _identifier(holder_id, "holder_id"),
        now=instant,
        lease_seconds=due.spec.lease_seconds,
    )
    if lease is None:
        return None
    backend.assert_fence(lease.job_id, lease.fencing_token)
    started = JobRunRecord(
        job_id=lease.job_id,
        fencing_token=lease.fencing_token,
        scheduled_at=due.due_at,
        started_at=instant,
        completed_at=None,
        success=None,
    )
    store.record_started(started)
    return AcquiredJob(due, lease, started)


def finish_acquired_job(
    acquired: AcquiredJob,
    backend: LeadershipLeaseBackend,
    store: JobRunStore,
    *,
    now: datetime,
    success: bool,
    error_type: str | None = None,
) -> JobRunRecord:
    instant = _utc(now, "now")
    backend.assert_fence(acquired.lease.job_id, acquired.lease.fencing_token)
    finished = JobRunRecord(
        job_id=acquired.lease.job_id,
        fencing_token=acquired.lease.fencing_token,
        scheduled_at=acquired.started_record.scheduled_at,
        started_at=acquired.started_record.started_at,
        completed_at=instant,
        success=success,
        error_type=error_type,
    )
    store.record_finished(finished)
    backend.release(acquired.lease, now=instant)
    return finished


__all__ = [
    "AcquiredJob",
    "DueJob",
    "JobRunRecord",
    "JobRunStore",
    "LeadershipLease",
    "LeadershipLeaseBackend",
    "PeriodicJobSpec",
    "acquire_due_job",
    "deterministic_jitter_seconds",
    "due_jobs",
    "finish_acquired_job",
    "next_due_time",
]
