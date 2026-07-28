import os
import time
from pathlib import Path

import pytest

from tools.document_store import DocumentStore
from tools.job_store import JobStore
from tools.rate_limit import SlidingWindowRateLimiter


@pytest.fixture(autouse=True)
def disable_automatic_orphan_cleanup(monkeypatch):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")


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
    assert internal["doc_id"] is None
    second.update(
        "job-1",
        "alice",
        status="finalizing",
        filename="paper.pdf",
        source_path=str(source),
        doc_id="doc-1",
    )
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


def test_recoverable_includes_queued_processing_and_finalizing_only(tmp_path):
    store = JobStore(tmp_path / "jobs.sqlite3")
    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    for job_id, status in (
        ("queued", "queued"),
        ("processing", "processing"),
        ("finalizing", "finalizing"),
        ("done", "success"),
        ("failed", "failed"),
    ):
        store.update(
            job_id,
            "alice",
            status=status,
            filename=f"{job_id}.txt",
            source_path=str(source) if status not in {"success", "failed"} else "",
            doc_id="doc-1" if status in {"finalizing", "success"} else None,
        )
    records = {record["job_id"]: record for record in store.recoverable()}
    assert set(records) == {"queued", "processing", "finalizing"}
    assert records["queued"]["doc_id"] is None
    assert records["processing"]["doc_id"] is None
    assert records["finalizing"]["doc_id"] == "doc-1"
    assert store.active_source_paths() == {source.resolve()}


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


def test_document_store_is_owner_scoped_and_keeps_paths_private(tmp_path):
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
    assert store.retained_source_path(owner_id="alice", doc_id="doc-1") == source.resolve()
    assert store.retained_source_path(owner_id="bob", doc_id="doc-1") is None
    assert store.source_path(owner_id="alice", doc_id="doc-1") is None
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


def test_document_store_rejects_symlinked_sources(tmp_path):
    upload_root = tmp_path / "uploads"
    target = upload_root / "alice" / "target.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-test")
    link = upload_root / "alice" / "link.pdf"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    with pytest.raises(ValueError, match="regular file"):
        store.register(
            owner_id="alice",
            doc_id="doc-1",
            filename="paper.pdf",
            mime_type="application/pdf",
            source_path=link,
        )


def test_copy_source_is_bounded_and_rolls_back_partial_output(tmp_path):
    source = tmp_path / "large.txt"
    source.write_bytes(b"0123456789")
    upload_root = tmp_path / "uploads"
    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    with pytest.raises(ValueError, match="limit"):
        store.copy_source(owner_id="alice", source_path=source, max_bytes=9)
    assert not list(upload_root.rglob("*.txt"))

    copied = store.copy_source(owner_id="alice", source_path=source, max_bytes=10)
    assert copied.read_bytes() == source.read_bytes()
    assert copied.parent == (upload_root / "alice").resolve()


def test_orphan_cleanup_preserves_referenced_recent_and_symlink_files(tmp_path):
    upload_root = tmp_path / "uploads"
    owner_dir = upload_root / "alice"
    owner_dir.mkdir(parents=True)
    retained = owner_dir / "retained.pdf"
    active = owner_dir / "active.txt"
    recent = owner_dir / "recent.txt"
    orphan = owner_dir / "orphan.txt"
    for path in (retained, active, recent, orphan):
        path.write_text(path.stem, encoding="utf-8")

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = owner_dir / "link.txt"
    try:
        os.symlink(outside, link)
    except OSError:
        link = None

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.orphan_grace_seconds = 100
    store.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="retained.pdf",
        mime_type="application/pdf",
        source_path=retained,
    )
    for path in (retained, active, orphan):
        os.utime(path, (0, 0))
    os.utime(recent, (9_950, 9_950))

    class ActiveJobs:
        @staticmethod
        def active_source_paths():
            return {active.resolve()}

    assert store.cleanup_orphans(now=10_000, job_store=ActiveJobs()) == 1
    assert retained.exists()
    assert active.exists()
    assert recent.exists()
    assert not orphan.exists()
    assert outside.exists()
    if link is not None:
        assert link.is_symlink()


def test_orphan_cleanup_fails_closed_when_references_cannot_be_read(tmp_path):
    upload_root = tmp_path / "uploads"
    orphan = upload_root / "alice" / "orphan.txt"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("orphan", encoding="utf-8")
    os.utime(orphan, (0, 0))
    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.orphan_grace_seconds = 1

    class BrokenJobs:
        @staticmethod
        def active_source_paths():
            raise RuntimeError("database unavailable")

    assert store.cleanup_orphans(now=10_000, job_store=BrokenJobs()) == 0
    assert orphan.exists()
    assert store.last_cleanup_errors == ["reference_lookup_failed:RuntimeError"]


def test_sliding_window_rate_limiter_reports_retry_time():
    limiter = SlidingWindowRateLimiter(requests_per_minute=2)
    assert limiter.retry_after("alice", now=0.0) == 0.0
    assert limiter.retry_after("alice", now=1.0) == 0.0
    retry = limiter.retry_after("alice", now=2.0)
    assert retry > 0
    assert limiter.retry_after("bob", now=2.0) == 0.0
    assert limiter.retry_after("alice", now=61.0) == 0.0
