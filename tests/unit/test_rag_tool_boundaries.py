from types import SimpleNamespace

import pytest

import tools.rag_tool as rag_tool
from tools.rag import Chunk


def test_empty_query_does_not_initialize_vector_store(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "get_rag_layer",
        lambda: (_ for _ in ()).throw(AssertionError("RAG should not initialize")),
    )

    assert rag_tool.search_uploaded_docs("   ", owner_id="alice") == []


def test_oversized_query_and_doc_id_fail_before_vector_store(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "get_rag_layer",
        lambda: (_ for _ in ()).throw(AssertionError("RAG should not initialize")),
    )

    with pytest.raises(ValueError, match="10,000"):
        rag_tool.search_uploaded_docs("q" * 10001, owner_id="alice")
    with pytest.raises(ValueError, match="200"):
        rag_tool.search_uploaded_docs(
            "evidence",
            owner_id="alice",
            doc_id="d" * 201,
        )


def test_malformed_or_cross_owner_chunks_are_not_cited(monkeypatch):
    chunks = [
        Chunk(
            id="good",
            text="good evidence",
            metadata={
                "owner_id": "alice",
                "doc_id": "doc-1",
                "filename": "paper.pdf",
                "parent_text": "parent evidence",
            },
            distance=0.1,
            score=0.9,
        ),
        Chunk(
            id="wrong-owner",
            text="private evidence",
            metadata={"owner_id": "bob", "doc_id": "doc-1"},
            distance=0.2,
            score=0.8,
        ),
        Chunk(
            id="wrong-document",
            text="other document",
            metadata={"owner_id": "alice", "doc_id": "doc-2"},
            distance=0.3,
            score=0.7,
        ),
        Chunk(
            id="missing-document",
            text="unattributed evidence",
            metadata={"owner_id": "alice"},
            distance=0.4,
            score=0.6,
        ),
    ]
    rag = SimpleNamespace(
        query=lambda *_args, **_kwargs: chunks,
        generate_hyde_query=lambda query, *_args, **_kwargs: query,
    )
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    citations = rag_tool.search_uploaded_docs(
        "evidence",
        owner_id="alice",
        doc_id="doc-1",
    )

    assert len(citations) == 1
    assert citations[0].label == "[1]"
    assert citations[0].doc_id == "doc-1"
    assert citations[0].source_id == "good"
    assert citations[0].quote == "good evidence"


def test_owner_id_is_validated_before_vector_store(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "get_rag_layer",
        lambda: (_ for _ in ()).throw(AssertionError("RAG should not initialize")),
    )

    with pytest.raises(ValueError, match="Owner identifiers"):
        rag_tool.search_uploaded_docs("evidence", owner_id="../alice")
