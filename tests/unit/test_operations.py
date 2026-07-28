import time
from pathlib import Path

import pytest

from tools.document_store import DocumentStore
from tools.job_store import JobStore
from tools.rate_limit import SlidingWindowRateLimiter


def test_job_store_is_owner_scoped_persistent_and_prunable(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    source = tmp_path / "upload.pdf"
    source.write_bytes(b"%PDF-test")
    first = JobStore(path, ttl_seconds=60)
    first.update(
        "job-1",
        "alice",
        status="queued",
        filename="paper.pdf",
        source_path=str(source),
    )
    second = JobStore(path, ttl_seconds=60)
    assert second.get("job-1", "alice")["filename"] == "paper.pdf"
    assert second.get("job-1", "bob") is None
    assert second.claim("job-1", "alice", max_attempts=3) is True
    assert first.claim("job-1", "alice", max_attempts=3) is False
    internal = second.get_internal("job-1", "alice")
    assert internal and internal["status"] == "processing"
    assert internal["attempts"] == 1
    second.update(
        "job-1",
        "alice",
        status="success",
        filename="paper.pdf",
        source_path="",
        doc_id="doc-1",
    )
    assert first.get("job-1", "alice")["doc_id"] == "doc-1"
    assert second.prune(now=time.time() + 120) == 1
    assert second.get("job-1", "alice") is None


def test_job_id_cannot_be_reassigned_across_owners(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.update("job-1", "alice", status="queued", filename="a.txt")
    with pytest.raises(PermissionError):
        store.update("job-1", "bob", status="queued", filename="b.txt")


def test_recoverable_includes_interrupted_jobs_and_not_completed_jobs(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    store.update(
        "queued",
        "alice",
        status="queued",
        filename="a.txt",
        source_path=str(source),
    )
    store.update(
        "done",
        "alice",
        status="success",
        filename="b.txt",
        source_path="",
    )
    assert [record["job_id"] for record in store.recoverable()] == ["queued"]


def test_atomic_claim_enforces_attempt_limit(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="a.txt",
        source_path=str(source),
    )
    assert store.claim("job-1", "alice", max_attempts=1) is True
    store.update(
        "job-1",
        "alice",
        status="queued",
        filename="a.txt",
        source_path=str(source),
    )
    assert store.claim("job-1", "alice", max_attempts=1) is False


def test_document_store_is_owner_scoped_and_keeps_paths_out_of_public_vectors(tmp_path):
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "source.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-test")
    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    previous = store.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )
    assert previous is None
    assert store.source_path(owner_id="alice", doc_id="doc-1") == source.resolve()
    assert store.source_path(owner_id="bob", doc_id="doc-1") is None
    record = store.delete(owner_id="alice", doc_id="doc-1")
    assert record and Path(record["source_path"]) == source.resolve()
    assert store.get(owner_id="alice", doc_id="doc-1") is None


def test_document_store_rejects_sources_outside_upload_root(tmp_path):
    upload_root = tmp_path / "uploads"
    upload_root.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-test")
    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    with pytest.raises(ValueError):
        store.register(
            owner_id="alice",
            doc_id="doc-1",
            filename="paper.pdf",
            mime_type="application/pdf",
            source_path=outside,
        )


def test_sliding_window_rate_limiter_reports_retry_time():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)
    assert limiter.retry_after("alice", now=0.0) == 0.0
    assert limiter.retry_after("alice", now=1.0) == 0.0
    retry = limiter.retry_after("alice", now=2.0)
    assert retry > 0
    assert limiter.retry_after("bob", now=2.0) == 0.0
    assert limiter.retry_after("alice", now=61.0) == 0.0
