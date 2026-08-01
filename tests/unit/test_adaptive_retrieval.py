from types import SimpleNamespace

import pytest

from tools.adaptive_retrieval import (
    EvidenceSignals,
    analyze_query,
    build_corrective_plan,
    evaluate_evidence,
    initial_attempt,
)


def test_exact_identifier_routes_to_corpus_sparse():
    analysis = analyze_query("Find DOI 10.1000/xyz123 exactly")
    attempt = initial_attempt("Find DOI 10.1000/xyz123 exactly", top_k=3)
    assert analysis.intent == "exact_lookup"
    assert analysis.exact_identifier is True
    assert attempt.mode == "corpus-sparse"
    assert attempt.use_multi_query is False
    assert attempt.reranker == "none"


def test_comparison_routes_to_multi_query_hybrid():
    analysis = analyze_query(
        "Compare the effect estimates and methods in treatment A versus B"
    )
    attempt = initial_attempt(
        "Compare the effect estimates and methods in treatment A versus B"
    )
    assert analysis.intent == "comparison"
    assert analysis.comparative and analysis.quantitative and analysis.methodological
    assert attempt.mode == "corpus-hybrid"
    assert attempt.use_multi_query is True
    assert attempt.reranker == "heuristic"


def test_evidence_signals_include_provenance_and_generation():
    rows = [
        SimpleNamespace(
            doc_id="doc-1",
            score=0.9,
            page_number=2,
            source_id="chunk-1",
            generation_sequence=3,
            source_kind="dense_chunk",
            metadata={},
        ),
        SimpleNamespace(
            doc_id="doc-2",
            score=0.8,
            page_number=4,
            source_id="field-1",
            generation_sequence=2,
            source_kind="sparse_field",
            metadata={},
        ),
        SimpleNamespace(
            doc_id="doc-3",
            score=0.7,
            page_number=5,
            source_id="field-2",
            generation_sequence=1,
            source_kind="sparse_field",
            metadata={},
        ),
    ]
    signals = evaluate_evidence(rows)
    assert signals.unique_documents == 3
    assert signals.strong_evidence_count == 3
    assert signals.provenance_fraction == 1.0
    assert signals.generation_fraction == 1.0
    assert signals.source_kind_count == 2
    assert signals.decision == "sufficient"


def test_empty_and_malformed_scores_fail_closed():
    assert evaluate_evidence([]).decision == "empty"
    signals = evaluate_evidence(
        [
            {"doc_id": "doc-1", "score": True, "metadata": {}},
            {"doc_id": "doc-2", "score": float("nan"), "metadata": {}},
        ]
    )
    assert signals.top_score == 0.0
    assert signals.decision == "insufficient"


def test_sufficient_evidence_stops_after_initial_attempt():
    signals = EvidenceSignals(
        evidence_count=3,
        unique_documents=2,
        top_score=0.9,
        mean_score=0.8,
        strong_evidence_count=3,
        provenance_fraction=1.0,
        generation_fraction=1.0,
        source_kind_count=2,
        sufficiency=0.8,
        decision="sufficient",
    )
    plan = build_corrective_plan(
        "Explain the mechanism with evidence",
        signals=signals,
        max_attempts=4,
    )
    assert len(plan.attempts) == 1
    assert plan.estimated_cost == plan.attempts[0].estimated_cost


def test_weak_evidence_produces_bounded_deduplicated_plan():
    signals = EvidenceSignals(
        evidence_count=1,
        unique_documents=1,
        top_score=0.3,
        mean_score=0.3,
        strong_evidence_count=0,
        provenance_fraction=0.0,
        generation_fraction=0.0,
        source_kind_count=1,
        sufficiency=0.2,
        decision="insufficient",
    )
    plan = build_corrective_plan(
        "Why does this mechanism change over time?",
        signals=signals,
        max_attempts=3,
        max_estimated_cost=300,
    )
    assert 1 < len(plan.attempts) <= 3
    assert plan.estimated_cost <= 300
    assert plan.abstain_after_exhaustion is True
    keys = {
        (
            attempt.mode,
            attempt.candidate_pool,
            attempt.use_multi_query,
            attempt.use_hyde,
            attempt.reranker,
        )
        for attempt in plan.attempts
    }
    assert len(keys) == len(plan.attempts)
    assert any(attempt.use_hyde for attempt in plan.attempts)


def test_iterator_and_input_boundaries():
    class Hostile:
        def __iter__(self):
            yield {"score": 0.5}
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="safely iterable"):
        evaluate_evidence(Hostile())
    with pytest.raises(ValueError, match="query"):
        analyze_query("bad\x01query")
    with pytest.raises(ValueError, match="max_attempts"):
        build_corrective_plan("question", max_attempts=True)
    with pytest.raises(ValueError, match="signals"):
        build_corrective_plan("question", signals=object())
