"""One-shot fenced periodic reconciliation orchestration.

This module intentionally does not start timers or background threads.  A process-level
scheduler invokes ``run_due_reconciliations`` periodically.  Due-time computation and
leadership fencing come from :mod:`orchestration.periodic_leadership`; job callbacks are
bounded, explicitly registered, re-fence before/after mutation, renew long leases, and
record terminal success/failure without leaking exception text into durable state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Protocol, Sequence

from orchestration.periodic_leadership import (
    AcquiredJob,
    JobRunStore,
    LeadershipLeaseBackend,
    PeriodicJobSpec,
    acquire_due_job,
    due_jobs,
    finish_acquired_job,
)


def _id(value: Any, label: str, limit: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > limit or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{label} is invalid")
    return value


@dataclass(frozen=True)
class ReconciliationResult:
    examined: int = 0
    repaired: int = 0
    unchanged: int = 0
    failed_items: int = 0
    continuation_token: str | None = None

    def __post_init__(self) -> None:
        for name in ("examined", "repaired", "unchanged", "failed_items"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.continuation_token is not None:
            object.__setattr__(self, "continuation_token", _id(self.continuation_token, "continuation_token", 4000))


class ReconciliationJob(Protocol):
    def __call__(self, *, fencing_token: int, continuation_token: str | None) -> ReconciliationResult: ...


@dataclass(frozen=True)
class ReconciliationBinding:
    spec: PeriodicJobSpec
    callback: ReconciliationJob
    renewal_margin_seconds: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.spec, PeriodicJobSpec):
            raise ValueError("spec must be PeriodicJobSpec")
        if not callable(self.callback):
            raise ValueError("callback must be callable")
        if isinstance(self.renewal_margin_seconds, bool) or not isinstance(self.renewal_margin_seconds, int) or self.renewal_margin_seconds < 1:
            raise ValueError("renewal_margin_seconds must be positive")
        if self.renewal_margin_seconds >= self.spec.lease_seconds:
            raise ValueError("renewal margin must be smaller than lease duration")


class ContinuationStore(Protocol):
    def get(self, job_id: str) -> str | None: ...
    def compare_and_set(self, job_id: str, *, fencing_token: int, value: str | None) -> None: ...


@dataclass(frozen=True)
class PeriodicReconciliationReport:
    job_id: str
    acquired: bool
    success: bool | None
    result: ReconciliationResult | None = None
    failure_type: str | None = None


def _renew_if_needed(acquired: AcquiredJob, backend: LeadershipLeaseBackend, binding: ReconciliationBinding, *, now: datetime) -> AcquiredJob:
    remaining = acquired.lease.expires_at - now.astimezone(acquired.lease.expires_at.tzinfo)
    if remaining > timedelta(seconds=binding.renewal_margin_seconds):
        return acquired
    renewed = backend.renew(
        acquired.lease,
        now=now,
        lease_seconds=binding.spec.lease_seconds,
    )
    backend.assert_fence(renewed.job_id, renewed.fencing_token)
    return AcquiredJob(acquired.due, renewed, acquired.started_record)


def run_due_reconciliations(
    bindings: Sequence[ReconciliationBinding],
    *,
    backend: LeadershipLeaseBackend,
    run_store: JobRunStore,
    continuation_store: ContinuationStore,
    holder_id: str,
    now: datetime,
    after_job_now: Callable[[], datetime] | None = None,
) -> tuple[PeriodicReconciliationReport, ...]:
    """Claim and execute each due job at most once for this scheduler invocation."""

    by_id: dict[str, ReconciliationBinding] = {}
    for binding in bindings:
        if binding.spec.job_id in by_id:
            raise ValueError("periodic reconciliation job ids must be unique")
        by_id[binding.spec.job_id] = binding
    reports: list[PeriodicReconciliationReport] = []
    for due in due_jobs(tuple(binding.spec for binding in bindings), run_store, now=now):
        binding = by_id[due.spec.job_id]
        acquired = acquire_due_job(due, backend, run_store, holder_id=holder_id, now=now)
        if acquired is None:
            reports.append(PeriodicReconciliationReport(due.spec.job_id, False, None))
            continue
        try:
            backend.assert_fence(acquired.lease.job_id, acquired.lease.fencing_token)
            continuation = continuation_store.get(due.spec.job_id)
            result = binding.callback(
                fencing_token=acquired.lease.fencing_token,
                continuation_token=continuation,
            )
            if not isinstance(result, ReconciliationResult):
                raise TypeError("reconciliation callback returned an invalid result")
            completed_at = after_job_now() if after_job_now is not None else now
            acquired = _renew_if_needed(acquired, backend, binding, now=completed_at)
            backend.assert_fence(acquired.lease.job_id, acquired.lease.fencing_token)
            continuation_store.compare_and_set(
                due.spec.job_id,
                fencing_token=acquired.lease.fencing_token,
                value=result.continuation_token,
            )
            finish_acquired_job(acquired, backend, run_store, now=completed_at, success=True)
            reports.append(PeriodicReconciliationReport(due.spec.job_id, True, True, result=result))
        except Exception as exc:
            failure_type = type(exc).__name__
            completed_at = after_job_now() if after_job_now is not None else now
            try:
                backend.assert_fence(acquired.lease.job_id, acquired.lease.fencing_token)
                finish_acquired_job(
                    acquired,
                    backend,
                    run_store,
                    now=completed_at,
                    success=False,
                    error_type=failure_type,
                )
            finally:
                reports.append(PeriodicReconciliationReport(due.spec.job_id, True, False, failure_type=failure_type))
    return tuple(reports)


__all__ = ["ContinuationStore", "PeriodicReconciliationReport", "ReconciliationBinding", "ReconciliationJob", "ReconciliationResult", "run_due_reconciliations"]
