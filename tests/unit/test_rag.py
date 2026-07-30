import threading
from unittest.mock import MagicMock

import pytest

from tools.ingestion_models import DocumentSection
from tools.rag import RAGLayer


def make_rag():
    rag = object.__new__(RAGLayer)
    rag.collection = MagicMock()
    rag.collection.get.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }
    rag._write_lock = threading.RLock()
    return rag


def test_add_document_is_owner_scoped_and_retry_safe():
    rag = make_rag()
    rag.collection.get.return_value = {
        "ids": ["obsolete-chunk"],
        "documents": ["old evidence"],
        "metadatas": [{"owner_id": "alice", "doc_id": "doc-1"}],
    }
    count = rag.add_document(
        "doc-1",
        None,
        sections=[DocumentSection(title="Methods", content="alpha beta gamma " * 100)],
        metadata={"owner_id": "alice", "filename": "paper.pdf"},
        chunk_size=120,
        overlap=20,
    )
    assert count > 0
    get_where = rag.collection.get.call_args.kwargs["where"]
    assert {"owner_id": {"$eq": "alice"}} in get_where["$and"]
    assert {"doc_id": {"$eq": "doc-1"}} in get_where["$and"]
    assert rag.collection.upsert.called
    upsert = rag.collection.upsert.call_args.kwargs
    assert all(meta["owner_id"] == "alice" for meta in upsert["metadatas"])
    assert all(meta["section_title"] == "Methods" for meta in upsert["metadatas"])
    rag.collection.delete.assert_called_once_with(ids=["obsolete-chunk"])


def test_failed_batch_rolls_back_new_ids_and_restores_previous_chunks():
    rag = make_rag()
    rag.collection.get.return_value = {
        "ids": ["previous"],
        "documents": ["previous evidence"],
        "metadatas": [{"owner_id": "alice", "doc_id": "doc-1"}],
    }
    upsert_calls = {"count": 0}

    def fail_second_batch(**_kwargs):
        upsert_calls["count"] += 1
        if upsert_calls["count"] == 2:
            raise RuntimeError("embedding failed")

    rag.collection.upsert.side_effect = fail_second_batch
    with pytest.raises(RuntimeError, match="embedding failed"):
        rag.add_document(
            "doc-1",
            "evidence " * 5000,
            {"owner_id": "alice"},
            chunk_size=20,
            overlap=2,
        )

    deleted_ids = [
        chunk_id
        for call in rag.collection.delete.call_args_list
        for chunk_id in call.kwargs.get("ids", [])
    ]
    assert "previous" not in deleted_ids
    assert deleted_ids
    restore_call = rag.collection.upsert.call_args_list[-1]
    assert restore_call.kwargs["ids"] == ["previous"]
    assert restore_call.kwargs["documents"] == ["previous evidence"]


def test_non_replace_success_retains_old_non_overlapping_chunks():
    rag = make_rag()
    rag.collection.get.return_value = {
        "ids": ["legacy-chunk"],
        "documents": ["legacy evidence"],
        "metadatas": [{"owner_id": "alice", "doc_id": "doc-1"}],
    }

    count = rag.add_document(
        "doc-1",
        "new evidence " * 100,
        {"owner_id": "alice"},
        chunk_size=120,
        overlap=20,
        replace=False,
    )

    assert count > 0
    rag.collection.delete.assert_not_called()
    assert rag.collection.get.call_args.kwargs["where"]["$and"] == [
        {"owner_id": {"$eq": "alice"}},
        {"doc_id": {"$eq": "doc-1"}},
    ]


def test_non_replace_failure_restores_overwritten_deterministic_chunk():
    rag = make_rag()
    overlapping_id = "doc-1:s0:p0:c0"
    rag.collection.get.return_value = {
        "ids": [overlapping_id, "legacy-chunk"],
        "documents": ["old first chunk", "legacy evidence"],
        "metadatas": [
            {"owner_id": "alice", "doc_id": "doc-1", "version": "old"},
            {"owner_id": "alice", "doc_id": "doc-1", "version": "legacy"},
        ],
    }
    upsert_calls = {"count": 0}

    def fail_second_batch(**_kwargs):
        upsert_calls["count"] += 1
        if upsert_calls["count"] == 2:
            raise RuntimeError("embedding failed")

    rag.collection.upsert.side_effect = fail_second_batch
    with pytest.raises(RuntimeError, match="embedding failed"):
        rag.add_document(
            "doc-1",
            "new evidence " * 5000,
            {"owner_id": "alice"},
            chunk_size=20,
            overlap=2,
            replace=False,
        )

    deleted_ids = {
        chunk_id
        for call in rag.collection.delete.call_args_list
        for chunk_id in call.kwargs.get("ids", [])
    }
    assert overlapping_id not in deleted_ids
    assert "legacy-chunk" not in deleted_ids
    restore_call = rag.collection.upsert.call_args_list[-1]
    assert restore_call.kwargs["ids"] == [overlapping_id, "legacy-chunk"]
    assert restore_call.kwargs["documents"] == ["old first chunk", "legacy evidence"]
    assert restore_call.kwargs["metadatas"][0]["version"] == "old"


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


def test_total_vector_query_failure_is_not_reported_as_empty_evidence():
    rag = make_rag()
    rag.collection.query.side_effect = RuntimeError("database unavailable")
    with pytest.raises(RuntimeError, match="Vector retrieval is unavailable"):
        rag.query("question", owner_id="alice")


def test_document_listing_pages_past_many_chunks_from_one_document(monkeypatch):
    rag = make_rag()
    monkeypatch.setattr("tools.rag.LIST_SCAN_BATCH", 3)
    monkeypatch.setattr("tools.rag.MAX_LIST_SCAN_CHUNKS", 12)
    rag.collection.get.side_effect = [
        {
            "ids": ["a1", "a2", "a3"],
            "metadatas": [
                {
                    "owner_id": "alice",
                    "doc_id": "doc-a",
                    "filename": "a.pdf",
                    "created_at": "2026-01-01",
                },
                {
                    "owner_id": "alice",
                    "doc_id": "doc-a",
                    "filename": "a.pdf",
                    "created_at": "2026-01-01",
                },
                {
                    "owner_id": "alice",
                    "doc_id": "doc-a",
                    "filename": "a.pdf",
                    "created_at": "2026-01-01",
                },
            ],
        },
        {
            "ids": ["a4", "b1"],
            "metadatas": [
                {
                    "owner_id": "alice",
                    "doc_id": "doc-a",
                    "filename": "a.pdf",
                    "created_at": "2026-01-01",
                },
                {
                    "owner_id": "alice",
                    "doc_id": "doc-b",
                    "filename": "b.pdf",
                    "created_at": "2026-02-01",
                },
            ],
        },
    ]

    documents = rag.list_documents(owner_id="alice", limit=2)

    assert [item["doc_id"] for item in documents] == ["doc-b", "doc-a"]
    assert rag.collection.get.call_args_list[1].kwargs["offset"] == 3


def test_empty_owner_is_rejected():
    rag = make_rag()
    with pytest.raises(ValueError, match="Owner identifiers"):
        rag.query("question", owner_id="")


def test_delete_document_is_owner_scoped():
    rag = make_rag()
    rag.delete_document(owner_id="alice", doc_id="doc-1")
    where = rag.collection.delete.call_args.kwargs["where"]
    assert {"owner_id": {"$eq": "alice"}} in where["$and"]
    assert {"doc_id": {"$eq": "doc-1"}} in where["$and"]
