from types import SimpleNamespace

import pytest

from tools.hybrid_retrieval import (
    RetrievalCandidate,
    bm25_scores,
    mmr_select,
    rank_candidates,
    reciprocal_rank_fusion,
    weighted_fusion,
)
from tools.reranking import HeuristicReranker


def candidate(identifier, text, dense=0.0, source=None):
    return RetrievalCandidate(identifier, text, source or identifier, dense)


def test_bm25_counts_document_frequency_not_term_frequency():
    values = [candidate("a", "alpha alpha alpha"), candidate("b", "alpha beta"), candidate("c", "beta")]
    scores = bm25_scores("alpha", values)
    assert scores["a"] > scores["b"] > scores["c"]
    assert scores["a"] == 1.0


def test_fusion_is_bounded_deterministic_and_deduplicated():
    rrf = reciprocal_rank_fusion({"dense": ["a", "a", "b"], "lexical": ["b", "a"]})
    assert set(rrf) == {"a", "b"}
    assert all(0.0 <= value <= 1.0 for value in rrf.values())
    weighted = weighted_fusion({"dense": {"a": 1.0, "b": 0.0}, "lexical": {"a": 0.0, "b": 1.0}}, weights={"dense": 3.0, "lexical": 1.0})
    assert weighted["a"] == pytest.approx(0.75)
    assert weighted["b"] == pytest.approx(0.25)


def test_mmr_reduces_near_duplicate_results():
    values = [
        (candidate("a", "alpha beta gamma", 1.0, "s1"), 1.0),
        (candidate("b", "alpha beta gamma", 0.99, "s2"), 0.99),
        (candidate("c", "alpha delta epsilon", 0.8, "s3"), 0.8),
    ]
    selected = mmr_select(values, top_k=2, diversity_lambda=0.5, max_per_source=2)
    assert [item.candidate_id for item, _ in selected] == ["a", "c"]


def test_rank_modes_and_reranker_components():
    values = [candidate("dense", "unrelated", 1.0), candidate("lexical", "target phrase", 0.1)]
    assert rank_candidates("target", values, mode="dense", top_k=1)[0].candidate.candidate_id == "dense"
    lexical = rank_candidates("target", values, mode="lexical", top_k=1)
    assert lexical[0].candidate.candidate_id == "lexical"
    hybrid = rank_candidates("target phrase", values, mode="hybrid", top_k=2, reranker=HeuristicReranker().score)
    assert hybrid[0].components["reranker"] >= 0.0
    assert all(0.0 <= row.score <= 1.0 for row in hybrid)


def test_bad_inputs_are_rejected_or_contained():
    with pytest.raises(ValueError):
        RetrievalCandidate("", "text", "source")
    with pytest.raises(ValueError):
        bm25_scores("query", "not candidates")
    with pytest.raises(ValueError):
        rank_candidates("query", [], mode="unknown")
    values = [candidate("a", "alpha", float("nan"))]
    assert values[0].dense_score == 0.0
