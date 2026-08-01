from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tools.rag_tool as rag_tool
from tools.corpus_hybrid_retrieval import CorpusEvidence


def evidence(
    *,
    evidence_id="sparse:doc-1:body",
    score=0.9,
    metadata=None,
):
    return CorpusEvidence(
        evidence_id=evidence_id,
        doc_id="doc-1",
        text="grounded corpus evidence",
        score=score,
        dense_score=0.4,
        sparse_score=0.9,
        generation_sequence=7,
        profile_fingerprint="a" * 64,
        source_kind="sparse_field",
        page_number=2,
        section="Results",
        metadata=metadata or {"filename": "paper.pdf", "field_type": "body"},
    )


def runtime(monkeypatch, *, expanded=None):
    rag = MagicMock()
    rag.generate_expanded_queries.return_value = expanded or ["question"]
    monkeypatch.setattr(rag_tool, "get_rag_layer", lambda: rag)
    sparse = object()
    generations = object()
    monkeypatch.setattr(rag_tool, "get_sparse_index", lambda: sparse)
    monkeypatch.setattr(rag_tool, "get_generation_store", lambda: generations)
    return rag, sparse, generations


def test_corpus_sparse_routes_to_independent_store_and_returns_provenance(monkeypatch):
    rag, sparse, generations = runtime(monkeypatch)
    calls = []

    def retrieve(query, **kwargs):
        calls.append((query, kwargs))
        return (evidence(),)

    monkeypatch.setattr(rag_tool, "retrieve_corpus_evidence", retrieve)
    citations = rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        retrieval_mode="corpus-sparse",
        n_results=3,
    )

    assert len(citations) == 1
    citation = citations[0]
    assert citation.title == "paper.pdf"
    assert citation.doc_id == "doc-1"
    assert citation.page_number == 2
    assert citation.metadata["retrieval_mode"] == "corpus-sparse"
    assert citation.metadata["generation_sequence"] == 7
    assert citation.metadata["embedding_profile_fingerprint"] == "a" * 64
    assert citation.metadata["evidence_kind"] == "sparse_field"
    assert calls[0][1]["rag"] is rag
    assert calls[0][1]["sparse"] is sparse
    assert calls[0][1]["generations"] is generations
    assert calls[0][1]["mode"] == "sparse"


def test_corpus_metadata_cannot_override_protected_fields(monkeypatch):
    runtime(monkeypatch)
    malicious = {
        "filename": "paper.pdf",
        "field_type": "body",
        "retrieval_mode": "dense",
        "generation_sequence": 999,
        "embedding_profile_fingerprint": "f" * 64,
        "evidence_kind": "dense_chunk",
        "dense_score": 1.0,
        "fused_score": 0.0,
    }
    monkeypatch.setattr(
        rag_tool,
        "retrieve_corpus_evidence",
        lambda *_args, **_kwargs: (evidence(metadata=malicious),),
    )

    citation = rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        retrieval_mode="corpus-hybrid",
    )[0]

    assert citation.metadata["retrieval_mode"] == "corpus-hybrid"
    assert citation.metadata["generation_sequence"] == 7
    assert citation.metadata["embedding_profile_fingerprint"] == "a" * 64
    assert citation.metadata["evidence_kind"] == "sparse_field"
    assert citation.metadata["dense_score"] == 0.4
    assert citation.metadata["fused_score"] == 0.9
    assert citation.metadata["field_type"] == "body"


def test_partial_multi_query_failure_preserves_successful_evidence(monkeypatch):
    runtime(monkeypatch, expanded=["question", "variant one", "variant two"])
    calls = []

    def retrieve(query, **_kwargs):
        calls.append(query)
        if query == "variant one":
            raise RuntimeError("provider detail")
        return (evidence(evidence_id=f"sparse:doc-1:{len(calls)}", score=0.5 + len(calls) / 10),)

    monkeypatch.setattr(rag_tool, "retrieve_corpus_evidence", retrieve)
    citations = rag_tool.search_uploaded_docs(
        "question",
        owner_id="alice",
        retrieval_mode="corpus-hybrid",
        use_multi_query=True,
        n_results=5,
    )

    assert calls == ["question", "variant one", "variant two"]
    assert len(citations) == 2


def test_all_corpus_queries_failing_is_explicitly_unavailable(monkeypatch):
    runtime(monkeypatch, expanded=["question", "variant"])
    monkeypatch.setattr(
        rag_tool,
        "retrieve_corpus_evidence",
        MagicMock(side_effect=RuntimeError("private backend detail")),
    )

    with pytest.raises(RuntimeError, match="Corpus retrieval is unavailable"):
        rag_tool.search_uploaded_docs(
            "question",
            owner_id="alice",
            retrieval_mode="corpus-hybrid",
            use_multi_query=True,
        )


def test_corpus_validation_precedes_sparse_and_generation_initialization(monkeypatch):
    monkeypatch.setattr(
        rag_tool,
        "get_rag_layer",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr(
        rag_tool,
        "get_sparse_index",
        lambda: (_ for _ in ()).throw(AssertionError("sparse initialized")),
    )
    monkeypatch.setattr(
        rag_tool,
        "get_generation_store",
        lambda: (_ for _ in ()).throw(AssertionError("generation initialized")),
    )

    with pytest.raises(ValueError, match="retrieval_mode"):
        rag_tool.search_uploaded_docs(
            "question",
            owner_id="alice",
            retrieval_mode="corpus-invalid",
        )
    with pytest.raises(ValueError, match="candidate_pool"):
        rag_tool.search_uploaded_docs(
            "question",
            owner_id="alice",
            retrieval_mode="corpus-hybrid",
            candidate_pool=True,
        )
