import os
from types import SimpleNamespace

import fitz

from tools.document_store import DocumentStore


def _make_pdf(path, *, pages=1, width=612, height=792):
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=width, height=height)
        page.insert_text((72, 72), f"Figure {index + 1}")
    document.save(path)
    document.close()


def test_missing_retained_source_is_reported_as_text_only(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-test")
    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )
    record = store.get(owner_id="alice", doc_id="doc-1")
    assert record["source_retained"] == 1
    assert record["visual_source_available"] is False

    source.unlink()

    record = store.get(owner_id="alice", doc_id="doc-1")
    assert record is not None
    assert record["source_retained"] == 0
    assert record["source_path"] is None
    assert record["visual_source_available"] is False
    assert store.source_path(owner_id="alice", doc_id="doc-1") is None
    assert store.retained_source_paths() == set()


def test_visual_page_limit_does_not_turn_retained_source_into_orphan(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("VISUAL_MAX_PDF_PAGES", "1")
    monkeypatch.setenv("ORPHAN_GRACE_SECONDS", "60")
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source, pages=2)
    os.utime(source, (1, 1))

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    record = store.get(owner_id="alice", doc_id="doc-1")
    assert record and record["source_retained"] == 1
    assert record["visual_source_available"] is False
    assert store.source_path(owner_id="alice", doc_id="doc-1") is None
    assert store.retained_source_paths() == {source.resolve()}
    deleted = store.cleanup_orphans(
        now=10_000,
        job_store=SimpleNamespace(active_source_paths=lambda: set()),
    )
    assert deleted == 0
    assert source.exists()


def test_visual_render_pixel_limit_rejects_extreme_page_geometry(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("VISUAL_MAX_RENDER_PIXELS", "1000000")
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "wide.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source, width=1000, height=1000)

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id="doc-wide",
        filename="wide.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    record = store.get(owner_id="alice", doc_id="doc-wide")
    assert record and record["source_retained"] == 1
    assert record["visual_source_available"] is False
    assert store.source_path(owner_id="alice", doc_id="doc-wide") is None
