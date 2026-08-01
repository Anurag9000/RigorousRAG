from unittest.mock import MagicMock

import tools.authoritative_document_index as boundary
from tools.authoritative_document_index import install_authoritative_rag_deletion
from tools.rag import RAGLayer
from tools.raw_index_coordinator import RawDeleteIndexCoordinator


def test_public_rag_delete_is_idempotent_and_authoritative(monkeypatch):
    install_authoritative_rag_deletion()
    first = RAGLayer.delete_document
    install_authoritative_rag_deletion()
    assert RAGLayer.delete_document is first
    assert getattr(first, "_rigorousrag_authoritative_delete", False) is True
    assert hasattr(RAGLayer, "_authoritative_raw_delete_document")
    assert RAGLayer._authoritative_raw_delete_document is not first

    calls = []
    monkeypatch.setattr(
        boundary,
        "delete_authoritative_document",
        lambda **kwargs: calls.append(kwargs) or True,
    )
    instance = object.__new__(RAGLayer)
    assert RAGLayer.delete_document(
        instance,
        owner_id="alice",
        doc_id="doc-1",
    ) is True
    assert calls == [
        {
            "owner_id": "alice",
            "doc_id": "doc-1",
            "rag": instance,
            "audit_metadata": {"operation": "document_delete"},
        }
    ]


def test_raw_coordinator_deletes_collection_without_public_lifecycle():
    rag = MagicMock()
    sparse = MagicMock()
    coordinator = RawDeleteIndexCoordinator(rag=rag, sparse=sparse)
    prior = MagicMock()
    prior.vector.row_count = 1
    prior.sparse = MagicMock()
    coordinator.snapshot = MagicMock(return_value=prior)

    assert coordinator.delete_document(owner_id="alice", doc_id="doc-1") is True
    rag.collection.delete.assert_called_once()
    rag.delete_document.assert_not_called()
    sparse.delete_document.assert_called_once_with(
        owner_id="alice",
        doc_id="doc-1",
    )
