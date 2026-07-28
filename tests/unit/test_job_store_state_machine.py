import math
import sqlite3
import time

import pytest

from tools.job_store import JobStore


def test_terminal_jobs_cannot_be_resurrected(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update(
        "job-success",
        "alice",
        status="success",
        filename="paper.txt",
        doc_id="doc-1",
    )
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
    queued = store.get_internal("job-1", "alice")
    assert queued["status"] == "queued"
    assert queued["doc_id"] is None

    assert store.claim(
        "job-1",
        "alice",
        max_attempts=3,
        now=time.time() + store.retry_max_seconds + 1,
    ) is True
    store.update(
        "job-1",
        "alice",
        status="finalizing",
        filename="paper.txt",
        doc_id="doc-1",
    )
    store.update(
        "job-1",
        "alice",
        status="success",
        filename="paper.txt",
        doc_id="doc-1",
    )
    assert store.get_internal("job-1", "alice")["status"] == "success"


def test_finalizing_replay_clears_uncommitted_document_id(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update("job-1", "alice", status="processing", filename="paper.txt")
    store.update(
        "job-1",
        "alice",
        status="finalizing",
        filename="paper.txt",
        doc_id="doc-uncommitted",
    )
    assert store.get("job-1", "alice")["doc_id"] == "doc-uncommitted"

    store.update("job-1", "alice", status="queued", filename="paper.txt")

    public = store.get("job-1", "alice")
    assert public["status"] == "queued"
    assert public["doc_id"] is None


def test_direct_processing_or_queued_success_is_rejected(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update("queued", "alice", status="queued", filename="paper.txt")
    with pytest.raises(ValueError, match="queued to success"):
        store.update(
            "queued",
            "alice",
            status="success",
            filename="paper.txt",
            doc_id="doc-1",
        )

    store.update("processing", "alice", status="processing", filename="paper.txt")
    with pytest.raises(ValueError, match="processing to success"):
        store.update(
            "processing",
            "alice",
            status="success",
            filename="paper.txt",
            doc_id="doc-1",
        )


def test_retry_environment_values_are_finite(monkeypatch, tmp_path):
    monkeypatch.setenv("INGEST_RETRY_BASE_SECONDS", "nan")
    monkeypatch.setenv("INGEST_RETRY_MAX_SECONDS", "inf")

    store = JobStore(tmp_path / "jobs.sqlite3")

    assert math.isfinite(store.retry_base_seconds)
    assert math.isfinite(store.retry_max_seconds)
    assert store.retry_base_seconds > 0
    assert store.retry_max_seconds >= store.retry_base_seconds


def test_retry_delay_is_bounded_for_corrupt_attempt_count(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")

    delay = store._retry_delay(10**100)

    assert math.isfinite(delay)
    assert delay == store.retry_max_seconds


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


def test_symlinked_job_database_is_refused(tmp_path):
    target = tmp_path / "target.sqlite3"
    with sqlite3.connect(target) as connection:
        connection.execute("SELECT 1")
    link = tmp_path / "jobs.sqlite3"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable in this environment.")

    with pytest.raises(ValueError, match="JOB_DB_PATH"):
        JobStore(link)
