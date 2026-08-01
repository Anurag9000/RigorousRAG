from types import SimpleNamespace

import pytest

import tools.adaptive_rag_tool as tool
from tools.adaptive_retrieval import EvidenceSignals, RetrievalAttempt
from tools.adaptive_retrieval_runner import (
    AdaptiveAttemptTrace,
    AdaptiveRetrievalResult,
)
from tools.models import Citation


def citation():
    return Citation(
        label="[1]",
        title="Paper",
        url="local://doc-1",
        source_type="uploaded_document",
        snippet="evidence",
        quote="evidence",
        source_id="chunk-1",
        doc_id="doc-1",
        chunk_id="chunk-1",
        page_number=2,
        metadata={"fused_score": 0.9, "generation_sequence": 1},
    )


def signals(decision="sufficient"):
    return EvidenceSignals(
        evidence_count=1,
        unique_documents=1,
        top_score=0.9,
        mean_score=0.9,
        strong_evidence_count=1,
        provenance_fraction=1.0,
        generation_fraction=1.0,
        source_kind_count=1,
        sufficiency=0.75,
        decision=decision,
    )


def result():
    attempt = RetrievalAttempt(
        mode="corpus-hybrid",
        top_k=5,
        candidate_pool=20,
        use_multi_query=True,
        reranker="heuristic",
        reason="test",
    )
    trace = AdaptiveAttemptTrace(
        attempt=attempt,
        returned_evidence=1,
        accumulated_evidence=1,
        signals=signals(),
        error_type=None,
    )
    return AdaptiveRetrievalResult(
        evidence=(citation(),),
        traces=(trace,),
        final_signals=signals(),
        exhausted=False,
        abstain=False,
        estimated_cost=attempt.estimated_cost,
    )


def test_tool_definition_is_explicit_and_bounded():
    function = tool.ADAPTIVE_RAG_SEARCH_TOOL_DEF["function"]
    assert function["name"] == "search_uploaded_docs_adaptive"
    parameters = function["parameters"]
    assert parameters["additionalProperties"] is False
    assert parameters["properties"]["max_attempts"]["maximum"] == 6
    assert parameters["properties"]["max_estimated_cost"]["maximum"] == 5000


def test_wrapper_forwards_to_adaptive_runner(monkeypatch):
    calls = []
    expected = result()
    monkeypatch.setattr(
        tool,
        "run_adaptive_retrieval",
        lambda query, **kwargs: calls.append((query, kwargs)) or expected,
    )
    returned = tool.search_uploaded_docs_adaptive(
        "question",
        owner_id="alice",
        doc_id="doc-1",
        top_k=7,
        max_attempts=3,
        max_estimated_cost=200,
        expansion_model="model",
        diversity_lambda=0.7,
    )
    assert returned is expected
    assert calls[0][0] == "question"
    assert calls[0][1]["search"] is tool.search_uploaded_docs
    assert calls[0][1]["owner_id"] == "alice"
    assert calls[0][1]["doc_id"] == "doc-1"
    assert calls[0][1]["top_k"] == 7
    assert calls[0][1]["max_attempts"] == 3


def test_payload_contains_citations_traces_and_abstention_state():
    payload = tool.adaptive_result_payload(result())
    assert payload["citations"][0]["doc_id"] == "doc-1"
    assert payload["citations"][0]["page_number"] == 2
    assert payload["traces"][0]["attempt"]["mode"] == "corpus-hybrid"
    assert payload["traces"][0]["error_type"] is None
    assert payload["final_signals"]["decision"] == "sufficient"
    assert payload["abstain"] is False
    assert payload["estimated_cost"] > 0
    assert "/private" not in repr(payload)


def test_non_serializable_evidence_is_skipped_without_stringification():
    base = result()

    class Hostile:
        def __str__(self):
            raise AssertionError("must not stringify")

    value = AdaptiveRetrievalResult(
        evidence=(Hostile(),),
        traces=base.traces,
        final_signals=base.final_signals,
        exhausted=True,
        abstain=True,
        estimated_cost=base.estimated_cost,
    )
    assert tool.adaptive_result_payload(value)["citations"] == []


def test_invalid_result_is_rejected():
    with pytest.raises(ValueError, match="AdaptiveRetrievalResult"):
        tool.adaptive_result_payload(SimpleNamespace())
