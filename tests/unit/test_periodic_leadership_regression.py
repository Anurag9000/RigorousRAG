from __future__ import annotations

from datetime import datetime, timedelta, timezone

from orchestration.periodic_leadership import JobRunRecord, PeriodicJobSpec, due_jobs, next_due_time


class _Store:
    def __init__(self, latest: JobRunRecord | None = None) -> None:
        self.value = latest

    def latest(self, job_id: str) -> JobRunRecord | None:
        del job_id
        return self.value


def _spec(*, jitter_seconds: int = 17) -> PeriodicJobSpec:
    return PeriodicJobSpec(
        job_id="population-reconciliation",
        interval_seconds=300,
        lease_seconds=30,
        max_runtime_seconds=60,
        jitter_seconds=jitter_seconds,
    )


def test_never_run_job_is_due_immediately_even_with_jitter() -> None:
    now = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    spec = _spec()

    assert next_due_time(spec, None, now=now) == now
    assert due_jobs((spec,), _Store(), now=now)[0].due_at == now


def test_repeated_polling_cannot_push_first_due_time_forward() -> None:
    spec = _spec()
    first_poll = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    later_poll = first_poll + timedelta(seconds=5)

    assert next_due_time(spec, None, now=first_poll) == first_poll
    assert next_due_time(spec, None, now=later_poll) == later_poll
    assert len(due_jobs((spec,), _Store(), now=first_poll)) == 1
    assert len(due_jobs((spec,), _Store(), now=later_poll)) == 1


def test_completed_job_keeps_deterministic_interval_jitter() -> None:
    completed = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
    latest = JobRunRecord(
        job_id="population-reconciliation",
        fencing_token=9,
        scheduled_at=completed - timedelta(seconds=10),
        started_at=completed - timedelta(seconds=5),
        completed_at=completed,
        success=True,
    )
    spec = _spec()
    poll_a = completed + timedelta(seconds=1)
    poll_b = completed + timedelta(seconds=120)

    assert next_due_time(spec, latest, now=poll_a) == next_due_time(spec, latest, now=poll_b)
