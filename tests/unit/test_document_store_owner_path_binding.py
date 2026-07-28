import time

import pytest

from tools.document_store import DocumentStore


def _store(monkeypatch, tmp_path):
    monkeypatch.setenv("ORPHAN_CLEANUP_ON_STARTUP", "false")
    root = tmp_path / "uploads"
    root.mkdir()
    return DocumentStore(tmp_path / "documents.sqlite3", root), root


def test_register_refuses_another_owners_retained_path(monkeypatch, tmp_path):
    store, root = _store(monkeypatch, tmp_path)
    bob_source = root / "bob" / "paper.pdf"
    bob_source.parent.mkdir()
    bob_source.write_bytes(b"%PDF-test")

    with pytest.raises(ValueError, match="matching owner directory"):
        store.register(
            owner_id="alice",
            doc_id="doc-1",
            filename="paper.pdf",
            mime_type="application/pdf",
            source_path=bob_source,
        )

    assert store.get(owner_id="alice", doc_id="doc-1") is None
    assert bob_source.exists()


def test_corrupted_cross_owner_row_downgrades_to_no_source(monkeypatch, tmp_path):
    store, root = _store(monkeypatch, tmp_path)
    bob_source = root / "bob" / "paper.pdf"
    bob_source.parent.mkdir()
    bob_source.write_bytes(b"%PDF-test")
    now = time.time()
    with store._lock, store._connect() as connection:
        connection.execute(
            """
            INSERT INTO documents(
                owner_id, doc_id, filename, mime_type, source_path,
                source_retained, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "alice",
                "doc-1",
                "paper.pdf",
                "application/pdf",
                str(bob_source),
                1,
                now,
                now,
            ),
        )

    record = store.get(owner_id="alice", doc_id="doc-1")
    assert record["source_path"] is None
    assert record["source_retained"] == 0
    assert record["visual_source_available"] is False
    assert store.retained_source_paths() == set()

    deleted = store.delete(owner_id="alice", doc_id="doc-1")
    assert deleted["source_path"] is None
    assert deleted["source_retained"] == 0
    assert bob_source.read_bytes() == b"%PDF-test"


def test_prior_corrupted_path_is_never_returned_for_cleanup(monkeypatch, tmp_path):
    store, root = _store(monkeypatch, tmp_path)
    alice_source = root / "alice" / "new.pdf"
    bob_source = root / "bob" / "old.pdf"
    alice_source.parent.mkdir()
    bob_source.parent.mkdir()
    alice_source.write_bytes(b"alice")
    bob_source.write_bytes(b"bob")
    now = time.time()
    with store._lock, store._connect() as connection:
        connection.execute(
            """
            INSERT INTO documents(
                owner_id, doc_id, filename, mime_type, source_path,
                source_retained, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "alice",
                "doc-1",
                "old.pdf",
                "application/pdf",
                str(bob_source),
                1,
                now,
                now,
            ),
        )

    previous = store.register(
        owner_id="alice",
        doc_id="doc-1",
        filename="new.pdf",
        mime_type="application/pdf",
        source_path=alice_source,
    )

    assert previous is None
    assert bob_source.read_bytes() == b"bob"
