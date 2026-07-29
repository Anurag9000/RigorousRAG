from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.rag_tool as rag_tool


def test_noniterable_chunk_backend_is_explicitly_unavailable(monkeypatch):
    rag = MagicMock()
    rag.query.return_value = object()
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    with pytest.raises(RuntimeError, match="invalid chunk collection"):
        rag_tool.search_uploaded_docs("question", owner_id="alice")


def test_hostile_chunk_iterator_is_explicitly_unavailable(monkeypatch):
    class BrokenChunks:
        def __iter__(self):
            raise RuntimeError("private iterator detail")

    rag = MagicMock()
    rag.query.return_value = BrokenChunks()
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    with pytest.raises(RuntimeError, match="invalid chunk collection"):
        rag_tool.search_uploaded_docs("question", owner_id="alice")


def test_non_string_hyde_output_fails_before_vector_query(monkeypatch):
    rag = MagicMock()
    rag.generate_hyde_query.return_value = object()
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    with pytest.raises(RuntimeError, match="invalid text"):
        rag_tool.search_uploaded_docs(
            "question",
            owner_id="alice",
            use_hyde=True,
        )

    rag.query.assert_not_called()


def test_boolean_relevance_is_normalized_to_zero(monkeypatch):
    rag = MagicMock()
    rag.query.return_value = [
        SimpleNamespace(
            id="chunk-1",
            text="evidence",
            score=True,
            metadata={"owner_id": "alice", "doc_id": "doc-1"},
        )
    ]
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    citations = rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
    )

    assert citations[0].metadata["relevance"] == 0.0


def test_hostile_metadata_mapping_is_skipped_without_leak(monkeypatch):
    class BrokenMetadata(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("private metadata detail")

    rag = MagicMock()
    rag.query.return_value = [
        SimpleNamespace(
            id="chunk-1",
            text="evidence",
            score=1.0,
            metadata=BrokenMetadata(),
        )
    ]
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)

    assert rag_tool.search_uploaded_docs("question", owner_id="alice") == []
