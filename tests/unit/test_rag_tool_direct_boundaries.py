from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.rag_tool as rag_tool


def test_direct_arguments_are_validated_before_vector_initialization(monkeypatch):
    initializer = MagicMock(side_effect=AssertionError("vector layer should not initialize"))
    monkeypatch.setattr(rag_tool, "get_rag_layer", initializer)

    cases = [
        {"query": object()},
        {"query": "q" * 10_001},
        {"query": "bad\x00query"},
        {"query": "bad\nquery"},
        {"query": "bad\rquery"},
        {"query": "bad\tquery"},
        {"query": "bad\x7fquery"},
        {"query": "query", "owner_id": object()},
        {"query": "query", "doc_id": "d" * 201},
        {"query": "query", "use_hyde": "yes"},
        {"query": "query", "use_multi_query": "yes"},
        {"query": "query", "expansion_model": "m" * 201},
        {"query": "query", "n_results": "bad"},
        {"query": "query", "n_results": True},
        {"query": "query", "n_results": 1.5},
        {"query": "query", "n_results": Decimal("1.5")},
        {"query": "query", "n_results": Fraction(3, 2)},
        {"query": "query", "n_results": 51},
    ]
    for arguments in cases:
        with pytest.raises(ValueError):
            rag_tool.search_uploaded_docs(**arguments)

    initializer.assert_not_called()


def test_exact_index_protocol_result_count_is_accepted(monkeypatch):
    class ExactInteger:
        def __index__(self):
            return 3

    rag = MagicMock()
    rag.query.return_value = []
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    assert rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        n_results=ExactInteger(),
    ) == []
    assert rag.query.call_args.kwargs["n_results"] == 3


def test_empty_query_returns_without_vector_initialization(monkeypatch):
    initializer = MagicMock(side_effect=AssertionError("vector layer should not initialize"))
    monkeypatch.setattr(rag_tool, "get_rag_layer", initializer)

    assert rag_tool.search_uploaded_docs("   ") == []
    initializer.assert_not_called()


def test_adapter_filters_owner_document_and_malformed_chunks(monkeypatch):
    chunks = [
        SimpleNamespace(
            id="good",
            text="supporting quote",
            score=0.75,
            metadata={
                "owner_id": "alice",
                "doc_id": "doc-1",
                "filename": "alice@example.com-paper.pdf",
                "parent_text": "parent evidence",
                "page_number": 2,
                "section_title": "Methods",
            },
        ),
        SimpleNamespace(
            id="wrong-owner",
            text="secret",
            score=1.0,
            metadata={"owner_id": "bob", "doc_id": "doc-1"},
        ),
        SimpleNamespace(
            id="padded-owner",
            text="secret",
            score=1.0,
            metadata={"owner_id": " alice ", "doc_id": "doc-1"},
        ),
        SimpleNamespace(
            id="wrong-doc",
            text="other",
            score=1.0,
            metadata={"owner_id": "alice", "doc_id": "doc-2"},
        ),
        SimpleNamespace(
            id="oversized-page",
            text="evidence without valid page provenance",
            score=0.5,
            metadata={
                "owner_id": "alice",
                "doc_id": "doc-1",
                "page_number": 1_000_001,
            },
        ),
        SimpleNamespace(
            id=object(),
            text="bad id",
            score=float("nan"),
            metadata={"owner_id": "alice", "doc_id": "doc-1"},
        ),
        object(),
    ]
    rag = MagicMock()
    rag.query.return_value = chunks
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    citations = rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        doc_id="doc-1",
        n_results=6,
    )

    assert len(citations) == 2
    citation = citations[0]
    assert citation.doc_id == "doc-1"
    assert citation.chunk_id == "good"
    assert citation.quote == "supporting quote"
    assert citation.page_number == 2
    assert citation.metadata["relevance"] == 0.75
    assert "alice@example.com" not in citation.title
    assert citations[1].chunk_id == "oversized-page"
    assert citations[1].page_number is None


def test_hyde_empty_result_returns_without_query(monkeypatch):
    rag = MagicMock()
    rag.generate_hyde_query.return_value = ""
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    assert rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        use_hyde=True,
    ) == []
    rag.query.assert_not_called()


def test_hyde_invalid_text_fails_before_query(monkeypatch):
    for generated in (object(), "bad\x00text", "bad\ntext", "bad\x7ftext", "x" * 20_001):
        rag = MagicMock()
        rag.generate_hyde_query.return_value = generated
        monkeypatch.setattr(rag_tool, "get_rag_layer", lambda rag=rag: rag)

        with pytest.raises(RuntimeError, match="invalid text"):
            rag_tool.search_uploaded_docs(
                "question",
                owner_id="alice",
                use_hyde=True,
            )
        rag.query.assert_not_called()


def test_infinite_chunk_stream_is_bounded_to_requested_results(monkeypatch):
    def chunks():
        index = 0
        while True:
            yield SimpleNamespace(
                id=f"chunk-{index}",
                text="evidence",
                score=1.0,
                metadata={"owner_id": "alice", "doc_id": "doc-1"},
            )
            index += 1

    rag = MagicMock()
    rag.query.return_value = chunks()
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    citations = rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        n_results=3,
    )

    assert len(citations) == 3
    assert [citation.chunk_id for citation in citations] == [
        "chunk-0",
        "chunk-1",
        "chunk-2",
    ]
