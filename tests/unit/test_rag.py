import threading
from unittest.mock import MagicMock

from tools.ingestion_models import DocumentSection
from tools.rag import RAGLayer


def make_rag():
    rag = object.__new__(RAGLayer)
    rag.collection = MagicMock()
    rag._write_lock = threading.RLock()
    return rag


def test_add_document_is_owner_scoped_and_retry_safe():
    rag = make_rag()
    count = rag.add_document(
        "doc-1",
        None,
        sections=[DocumentSection(title="Methods", content="alpha beta gamma " * 100)],
        metadata={"owner_id": "alice", "filename": "paper.pdf"},
        chunk_size=120,
        overlap=20,
    )
    assert count > 0
    delete_where = rag.collection.delete.call_args.kwargs["where"]
    assert {"owner_id": {"$eq": "alice"}} in delete_where["$and"]
    assert {"doc_id": {"$eq": "doc-1"}} in delete_where["$and"]
    assert rag.collection.upsert.called
    upsert = rag.collection.upsert.call_args.kwargs
    assert all(meta["owner_id"] == "alice" for meta in upsert["metadatas"])
    assert all(meta["section_title"] == "Methods" for meta in upsert["metadatas"])


def test_query_always_combines_owner_and_document_filters():
    rag = make_rag()
    rag.collection.query.return_value = {
        "ids": [["chunk-1"]],
        "documents": [["evidence"]],
        "metadatas": [[{"owner_id": "alice", "doc_id": "doc-1"}]],
        "distances": [[0.25]],
    }
    chunks = rag.query("question", owner_id="alice", doc_id="doc-1", n_results=3)
    assert len(chunks) == 1
    assert chunks[0].distance == 0.25
    assert chunks[0].score == 0.8
    where = rag.collection.query.call_args.kwargs["where"]
    assert {"owner_id": {"$eq": "alice"}} in where["$and"]
    assert {"doc_id": {"$eq": "doc-1"}} in where["$and"]


def test_empty_owner_is_rejected():
    rag = make_rag()
    try:
        rag.query("question", owner_id="")
    except ValueError as exc:
        assert "owner_id" in str(exc)
    else:
        raise AssertionError("An empty owner must never become an unscoped query.")


def test_delete_document_is_owner_scoped():
    rag = make_rag()
    rag.delete_document(owner_id="alice", doc_id="doc-1")
    where = rag.collection.delete.call_args.kwargs["where"]
    assert {"owner_id": {"$eq": "alice"}} in where["$and"]
    assert {"doc_id": {"$eq": "doc-1"}} in where["$and"]
