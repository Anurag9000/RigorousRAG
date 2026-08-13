from __future__ import annotations

import math

from evaluation.conformal import (
    conformal_retrieval_set,
    empirical_set_coverage,
    fit_retrieval_calibration,
)
from evaluation.evidence_influence import exact_shapley_influence, leave_one_out_influence
from evaluation.uncertainty import combine_rag_uncertainty, decompose_binary_uncertainty
from tools.claim_retrieval import retrieve_claim_evidence
from tools.hybrid_retrieval import RetrievalCandidate
from tools.retrieval_controls import RerankerStage, StructuredFilter, filter_candidates, run_reranker_cascade
from tools.review_routing import ReviewQueue, route_for_review
from tools.semantic_cache import CachePartition, PartitionedSemanticCache, SemanticCacheEntry
from tools.temporal_beliefs import BeliefEvent, revise_belief


def _candidate(identifier: str, score: float, **metadata: object) -> RetrievalCandidate:
    return RetrievalCandidate(identifier, f"text for {identifier}", f"source-{identifier}", score, metadata)


def test_structured_filter_and_budgeted_reranker_cascade() -> None:
    candidates = (
        _candidate("a", 0.6, mime_type="application/pdf", section="methods", page_number=3, provenance="primary", modified_at="2026-08-01T00:00:00Z"),
        _candidate("b", 0.9, mime_type="text/html", section="methods", page_number=3, provenance="primary", modified_at="2026-08-01T00:00:00Z"),
        _candidate("c", 0.4, mime_type="application/pdf", section="results", page_number=8, provenance="primary", modified_at="2026-08-01T00:00:00Z"),
    )
    spec = StructuredFilter(mime_types=("application/pdf",), sections=("methods",), min_page=2, max_page=5)
    assert [item.candidate_id for item in filter_candidates(candidates, spec)] == ["a"]

    cheap = RerankerStage("cheap", lambda _q, rows: {row.candidate_id: 1.0 if row.candidate_id == "c" else 0.2 for row in rows}, cost_units=1.0)
    expensive = RerankerStage("expensive", lambda _q, rows: {row.candidate_id: 1.0 for row in rows}, cost_units=5.0)
    result = run_reranker_cascade("query", candidates, (cheap, expensive), top_k=2, max_cost_units=1.5)
    assert result.stages_run == ("cheap",)
    assert result.stages_skipped == ("expensive",)
    assert result.candidates[0].candidate_id == "c"
    assert result.cost_units == 1.0


def test_conformal_retrieval_set_and_empirical_coverage() -> None:
    calibration = fit_retrieval_calibration((0.8, 0.85, 0.9, 0.95), alpha=0.25)
    selected = conformal_retrieval_set({"a": 0.9, "b": 0.7, "c": 0.99}, calibration)
    assert selected == ("c", "a")
    assert empirical_set_coverage((selected, ("x",)), (("a",), ("x",))) == 1.0


def test_uncertainty_decomposition_and_abstention_signal() -> None:
    breakdown = decompose_binary_uncertainty((0.1, 0.9))
    assert math.isclose(breakdown.predictive_mean, 0.5)
    assert math.isclose(breakdown.total, 0.25)
    assert math.isclose(breakdown.aleatoric, 0.09)
    assert math.isclose(breakdown.epistemic, 0.16)
    signal = combine_rag_uncertainty(
        retrieval_confidence=0.2,
        generation_confidence=0.3,
        evidence_conflict=0.8,
        proof_completeness=0.4,
        abstain_threshold=0.5,
    )
    assert signal.should_abstain is True
    assert signal.aggregate_uncertainty > 0.5


def test_temporal_retraction_and_source_caps() -> None:
    events = (
        BeliefEvent("claim", "old", "source-a", "support", 0.95, 100.0),
        BeliefEvent("claim", "dup", "source-a", "support", 0.80, 101.0),
        BeliefEvent("claim", "retract-old", "source-a", "retract", 1.0, 102.0, retracts_evidence_id="old"),
        BeliefEvent("claim", "counter", "source-b", "contradict", 0.90, 103.0),
    )
    state = revise_belief("claim", events, as_of=104.0)
    assert "old" not in state.active_evidence_ids
    assert state.retracted_evidence_ids == ("old",)
    assert state.status == "contradicted"
    assert state.independent_sources == ("source-a", "source-b")


def test_semantic_cache_enforces_partition_freshness_and_source_versions() -> None:
    partition = CachePartition("alice", "acl-v1", "model-v1", "policy-v1", "generation-7")
    other_partition = CachePartition("bob", "acl-v1", "model-v1", "policy-v1", "generation-7")
    cache = PartitionedSemanticCache(max_entries=4)
    cache.put(
        SemanticCacheEntry(
            "entry-1",
            partition,
            "what is x",
            "answer",
            (1.0, 0.0),
            {"doc-1": "v3"},
            100.0,
        )
    )
    hit = cache.lookup((0.99, 0.01), partition=partition, current_source_versions={"doc-1": "v3"}, now=110.0, max_age_seconds=20.0)
    assert hit is not None and hit.entry.answer == "answer"
    assert cache.lookup((0.99, 0.01), partition=other_partition, current_source_versions={"doc-1": "v3"}, now=110.0) is None
    assert cache.lookup((0.99, 0.01), partition=partition, current_source_versions={"doc-1": "v4"}, now=110.0) is None
    assert cache.lookup((0.99, 0.01), partition=partition, current_source_versions={"doc-1": "v3"}, now=200.0, max_age_seconds=20.0) is None


def test_counterfactual_and_shapley_evidence_influence() -> None:
    weights = {"a": 0.6, "b": 0.3}

    def scorer(ids: tuple[str, ...] | list[str]) -> float:
        return sum(weights[item] for item in ids)

    loo = leave_one_out_influence(("a", "b"), scorer)
    assert loo[0].evidence_id == "a"
    shapley = exact_shapley_influence(("a", "b"), scorer)
    values = {row.evidence_id: row.value for row in shapley}
    assert math.isclose(values["a"], 0.6)
    assert math.isclose(values["b"], 0.3)


def test_claim_conditioned_support_and_counterfactual_retrieval() -> None:
    calls: list[str] = []

    def retrieve(query: str, limit: int) -> tuple[RetrievalCandidate, ...]:
        calls.append(query)
        marker = "counter" if query.startswith("Find credible evidence") else "support"
        return tuple(_candidate(f"{marker}-{index}", 1.0 - index / 10) for index in range(limit))

    bundles = retrieve_claim_evidence(("The treatment improves survival.",), retrieve, per_claim=2)
    assert len(calls) == 2
    assert len(bundles[0].support) == 2
    assert len(bundles[0].counter) == 2
    assert bundles[0].counter_query is not None


def test_review_routing_and_priority_queue() -> None:
    decision = route_for_review(
        aggregate_uncertainty=0.45,
        evidence_conflict=0.5,
        proof_completeness=0.7,
        independent_sources=1,
    )
    assert decision.route == "human_review"
    assert "high_uncertainty" in decision.reasons
    assert "evidence_conflict" in decision.reasons
    queue = ReviewQueue(max_items=2)
    queue.push("request-1", decision, metadata={"claim": "x"})
    item = queue.pop()
    assert item is not None
    request_id, popped_decision, metadata = item
    assert request_id == "request-1"
    assert popped_decision == decision
    assert metadata["claim"] == "x"
    blocked = route_for_review(aggregate_uncertainty=0.1, security_violation=True)
    assert blocked.route == "block"
