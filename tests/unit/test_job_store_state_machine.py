import math

import pytest

from tools.job_store import JobStore


def test_terminal_jobs_cannot_be_resurrected(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update("job-success", "alice", status="success", filename="paper.txt")
    store.update("job-failed", "alice", status="failed", filename="paper.txt")

    with pytest.raises(ValueError, match="success to queued"):
        store.update("job-success", "alice", status="queued", filename="paper.txt")
    with pytest.raises(ValueError, match="failed to processing"):
        store.update("job-failed", "alice", status="processing", filename="paper.txt")

    assert store.get_internal("job-success", "alice")["status"] == "success"
    assert store.get_internal("job-failed", "alice")["status"] == "failed"


def test_invalid_status_is_rejected_before_persistence(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    with pytest.raises(ValueError, match="status must be one of"):
        store.update("job-1", "alice", status="cancelled", filename="paper.txt")

    assert store.get_internal("job-1", "alice") is None


def test_processing_and_finalizing_recovery_transitions_remain_valid(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update("job-1", "alice", status="processing", filename="paper.txt")
    store.update("job-1", "alice", status="queued", filename="paper.txt")
    assert store.get_internal("job-1", "alice")["status"] == "queued"

    assert store.claim("job-1", "alice", max_attempts=3, now=10_000) is True
    store.update("job-1", "alice", status="finalizing", filename="paper.txt")
    store.update("job-1", "alice", status="success", filename="paper.txt")
    assert store.get_internal("job-1", "alice")["status"] == "success"


def test_retry_environment_values_are_finite(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_RETRY_BASE_SECONDS", "nan")
    monkeypatch.setenv("INGEST_RETRY_MAX_SECONDS", "inf")

    store = JobStore(tmp_path / "jobs.sqlite3")

    assert math.isfinite(store.retry_base_seconds)
    assert math.isfinite(store.retry_max_seconds)
    assert store.retry_base_seconds > 0
    assert store.retry_max_seconds >= store.retry_base_seconds


def test_public_filename_never_becomes_empty(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update("job-1", "alice", status="queued", filename="   ")

    assert store.get("job-1", "alice")["filename"] == "upload"


def test_source_path_is_stored_lexically_without_following_symlink(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.txt",
        source_path=link,
    )

    internal = store.get_internal("job-1", "alice")
    assert internal["source_path"] == str(link.absolute())
    assert internal["source_path"] != str(target.resolve())
