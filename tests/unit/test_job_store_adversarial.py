import math

import pytest

import tools.job_store as job_store
from tools.job_store import JobStore


def test_constructor_rejects_boolean_fractional_and_invalid_ttl(tmp_path):
    for value in (True, 60.5, 0, -1, 31_536_001):
        with pytest.raises(ValueError, match="ttl_seconds"):
            JobStore(tmp_path / f"jobs-{value}.sqlite3", ttl_seconds=value)

    floored = JobStore(tmp_path / "jobs-floored.sqlite3", ttl_seconds=59)
    assert floored.ttl_seconds == 60


def test_unknown_update_fields_are_rejected_before_persistence(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match="Unsupported job update field"):
        store.update(
            "job-1",
            "alice",
            status="queued",
            filename="paper.txt",
            source_path="/tmp/paper.txt",
            unexpected="value",
        )

    assert store.get("job-1", "alice") is None


def test_nonfinite_clock_cannot_enter_durable_state(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.sqlite3")
    monkeypatch.setattr(job_store.time, "time", lambda: float("nan"))

    with pytest.raises(ValueError, match="current time"):
        store.update(
            "job-1",
            "alice",
            status="queued",
            filename="paper.txt",
            source_path="/tmp/paper.txt",
        )

    assert store.get("job-1", "alice") is None


def test_boolean_deadline_and_attempt_limits_are_rejected(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match="next_attempt_at"):
        store.update(
            "job-1",
            "alice",
            status="queued",
            filename="paper.txt",
            source_path="/tmp/paper.txt",
            next_attempt_at=True,
        )

    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path="/tmp/paper.txt",
    )
    with pytest.raises(ValueError, match="max_attempts"):
        store.claim("job-1", "alice", True)
    with pytest.raises(ValueError, match="attempts"):
        store.retry_delay_seconds(1.5)


def test_invalid_source_path_type_is_rejected(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match="filesystem path"):
        store.update(
            "job-1",
            "alice",
            status="queued",
            filename="paper.txt",
            source_path=object(),
        )

    assert store.get("job-1", "alice") is None


def test_retry_deadlines_remain_finite_and_bounded(tmp_path, monkeypatch):
    monkeypatch.setenv("INGEST_RETRY_BASE_SECONDS", "2")
    monkeypatch.setenv("INGEST_RETRY_MAX_SECONDS", "30")
    store = JobStore(tmp_path / "jobs.sqlite3")

    deadline = store.retry_deadline(1_000_000, now=100.0)

    assert math.isfinite(deadline)
    assert deadline == 130.0


def test_nonfinite_prune_clock_fails_without_deleting_rows(tmp_path, monkeypatch):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update(
        "job-1",
        "alice",
        status="failed",
        filename="paper.txt",
        source_path="",
        message="failed",
    )
    monkeypatch.setattr(job_store.time, "time", lambda: float("inf"))

    with pytest.raises(ValueError, match="current time"):
        store.prune()

    assert store.get("job-1", "alice")["status"] == "failed"
