import hashlib
import uuid

import fitz
import pytest

from tools.document_store import DocumentStore
from tools.upload_storage import read_owner_file


def _make_pdf(path):
    document = fitz.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((72, 72), "Figure 1")
    document.save(path)
    document.close()


def _doc_id(payload: bytes, owner="alice"):
    digest = hashlib.sha256(payload).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rigorousrag:{owner}:{digest}"))


def test_source_bytes_returns_identity_verified_preflighted_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    root = tmp_path / "uploads"
    source = root / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source)
    payload = source.read_bytes()
    document_id = _doc_id(payload)

    store = DocumentStore(tmp_path / "documents.sqlite3", root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )

    assert store.source_bytes(owner_id="alice", doc_id=document_id) == payload


def test_source_bytes_rejects_mutated_retained_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    root = tmp_path / "uploads"
    source = root / "alice" / "paper.pdf"
    source.parent.mkdir(parents=True)
    _make_pdf(source)
    document_id = _doc_id(source.read_bytes())

    store = DocumentStore(tmp_path / "documents.sqlite3", root)
    store.register(
        owner_id="alice",
        doc_id=document_id,
        filename="paper.pdf",
        mime_type="application/pdf",
        source_path=source,
    )
    source.write_bytes(source.read_bytes() + b"\n% mutation")

    assert store.source_bytes(owner_id="alice", doc_id=document_id) is None


def test_anchored_read_refuses_owner_directory_symlink_swap(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    owner = root / "alice"
    owner.mkdir()
    stored = owner / "paper.pdf"
    stored.write_bytes(b"inside")
    original_owner = root / "alice-original"
    owner.rename(original_owner)

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / stored.name
    outside_file.write_bytes(b"outside")
    try:
        owner.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        original_owner.rename(owner)
        pytest.skip("Symlinks are unavailable in this environment.")

    assert read_owner_file(root, stored, max_bytes=100) is None
    assert outside_file.read_bytes() == b"outside"
