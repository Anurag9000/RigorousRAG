import hashlib
import os
import uuid
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


def _doc_id(path, owner="alice"):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, f"rigorousrag:{owner}:{digest}")
    )


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
    assert record["visual_source_available"] is True
    assert record["visual_source_verified"] is False
    assert store.source_path(owner_id="alice", doc_id="doc-1") is None

    source.unlink()

    record = store.get(owner_id="alice", doc_id="doc-1")
    assert record is not None
    assert record["source_retained"] == 0
    assert record["source_path"] is None
    assert record["visual_source_available"] is False
    assert record["visual_source_verified"] is False
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
    document_id = _doc_id(source)
    os.utime(source, (1, 1))

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    record = store.get(owner_id="alice", doc_id=document_id)
    assert record and record["source_retained"] == 1
    assert record["visual_source_available"] is True
    assert record["visual_source_verified"] is False
    verified = store.get(
        owner_id="alice",
        doc_id=document_id,
        verify_visual=True,
    )
    assert verified and verified["visual_source_available"] is False
    assert verified["visual_source_verified"] is True
    assert store.source_path(owner_id="alice", doc_id=document_id) is None
    assert store.retained_source_path(owner_id="alice", doc_id=document_id) == source.resolve()
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
    document_id = _doc_id(source)

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="wide.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    record = store.get(owner_id="alice", doc_id=document_id)
    assert record and record["source_retained"] == 1
    assert record["visual_source_available"] is True
    assert record["visual_source_verified"] is False
    verified = store.get(
        owner_id="alice",
        doc_id=document_id,
        verify_visual=True,
    )
    assert verified and verified["visual_source_available"] is False
    assert store.source_path(owner_id="alice", doc_id=document_id) is None


def test_low_clip_override_cannot_weaken_renderer_preflight(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    monkeypatch.setenv("VISUAL_MAX_RENDER_PIXELS", "1000000")
    monkeypatch.setenv("VISUAL_CLIP_HEIGHT_POINTS", "100")
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "wide.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source, width=1000, height=1000)
    document_id = _doc_id(source)

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="wide.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    assert store.visual_clip_height_points == 565.0
    assert store.source_path(owner_id="alice", doc_id=document_id) is None
    assert store.retained_source_path(owner_id="alice", doc_id=document_id) == source.resolve()


def test_safe_pdf_is_returned_only_after_verification(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "safe.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source)
    document_id = _doc_id(source)

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="safe.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    quick = store.get(owner_id="alice", doc_id=document_id)
    assert quick and quick["visual_source_available"] is True
    assert quick["visual_source_verified"] is False
    verified = store.get(
        owner_id="alice",
        doc_id=document_id,
        verify_visual=True,
    )
    assert verified and verified["visual_source_available"] is True
    assert verified["visual_source_verified"] is True
    assert store.source_path(owner_id="alice", doc_id=document_id) == source.resolve()


def test_mutated_retained_pdf_is_refused_but_remains_managed(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    upload_root = tmp_path / "uploads"
    source = upload_root / "alice" / "mutated.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source)
    document_id = _doc_id(source)

    store = DocumentStore(tmp_path / "documents.sqlite3", upload_root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="mutated.pdf",
        mime_type="application/pdf",
        source_path=source,
    )
    assert store.source_path(owner_id="alice", doc_id=document_id) == source.resolve()

    source.write_bytes(source.read_bytes() + b"\n% host-side mutation")

    quick = store.get(owner_id="alice", doc_id=document_id)
    assert quick and quick["source_retained"] == 1
    assert store.source_path(owner_id="alice", doc_id=document_id) is None
    assert store.retained_source_path(owner_id="alice", doc_id=document_id) == source.resolve()
    assert store.retained_source_paths() == {source.resolve()}
