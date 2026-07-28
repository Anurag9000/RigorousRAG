from tools.document_store import DocumentStore


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
    assert store.get(owner_id="alice", doc_id="doc-1")["source_retained"] == 1

    source.unlink()

    record = store.get(owner_id="alice", doc_id="doc-1")
    assert record is not None
    assert record["source_retained"] == 0
    assert record["source_path"] is None
    assert store.source_path(owner_id="alice", doc_id="doc-1") is None
    assert store.retained_source_paths() == set()
