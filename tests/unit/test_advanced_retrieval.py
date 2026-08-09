from __future__ import annotations

from tools.advanced_retrieval import AdvancedRetrievalConfig, rank_advanced_candidates
from tools.hybrid_retrieval import RetrievalCandidate
from tools.retrieval_architectures import ScoreCalibration


def candidates():
    return [
        RetrievalCandidate(
            "dense",
            "general unrelated passage",
            "source-a",
            dense_score=0.95,
        ),
        RetrievalCandidate(
            "evidence",
            "retrieval augmented generation evidence graph",
            "source-b",
            dense_score=0.55,
        ),
        RetrievalCandidate(
            "mixed",
            "retrieval system overview",
            "source-c",
            dense_score=0.65,
        ),
    ]


class SparseScorer:
    def query_weights(self, query):
        assert query
        return {"retrieval": 2.0, "evidence": 1.0}

    def document_weights(self, text):
        if "evidence graph" in text:
            return {"retrieval": 2.0, "evidence": 1.0}
        if "retrieval" in text:
            return {"retrieval": 1.0}
        return {"unrelated": 1.0}


class LateScorer:
    def query_vectors(self, query):
        assert query
        return ((1.0, 0.0), (0.0, 1.0))

    def document_vectors(self, text):
        if "evidence graph" in text:
            return ((1.0, 0.0), (0.0, 1.0))
        if "retrieval" in text:
            return ((1.0, 0.0), (1.0, 0.0))
        return ((-1.0, 0.0), (0.0, -1.0))


def test_advanced_pipeline_combines_dense_bm25_sparse_late_and_rerank():
    result = rank_advanced_candidates(
        "retrieval evidence",
        candidates(),
        top_k=3,
        sparse_scorer=SparseScorer(),
        late_interaction_scorer=LateScorer(),
        reranker=lambda query, values: {
            item.candidate_id: (1.0 if item.candidate_id == "evidence" else 0.0)
            for item in values
        },
    )

    assert result[0].candidate.candidate_id == "evidence"
    assert result[0].components["splade"] > 0.0
    assert result[0].components["late_interaction"] == 1.0
    assert result[0].components["reranker"] == 1.0
    assert [item.rank for item in result] == [1, 2, 3]


def test_component_calibration_changes_fusion_without_changing_raw_component_scores():
    baseline = rank_advanced_candidates(
        "retrieval evidence",
        candidates(),
        config=AdvancedRetrievalConfig(
            dense_weight=1.0,
            lexical_weight=1.0,
            splade_weight=0.0,
            late_interaction_weight=0.0,
            reranker_weight=0.0,
        ),
    )
    calibrated = rank_advanced_candidates(
        "retrieval evidence",
        candidates(),
        config=AdvancedRetrievalConfig(
            dense_weight=1.0,
            lexical_weight=1.0,
            splade_weight=0.0,
            late_interaction_weight=0.0,
            reranker_weight=0.0,
            calibrations={"dense": ScoreCalibration(temperature=3.0)},
        ),
    )

    baseline_scores = {
        item.candidate.candidate_id: item.score for item in baseline
    }
    calibrated_scores = {
        item.candidate.candidate_id: item.score for item in calibrated
    }
    assert baseline_scores != calibrated_scores
    assert {
        item.candidate.candidate_id: item.components["dense"] for item in baseline
    } == {
        item.candidate.candidate_id: item.components["dense"] for item in calibrated
    }


def test_optional_adapter_failure_falls_back_to_safe_base_components():
    class BrokenSparse:
        def query_weights(self, query):
            raise RuntimeError("private model failure")

        def document_weights(self, text):
            raise AssertionError("must not be reached")

    class BrokenLate:
        def query_vectors(self, query):
            raise RuntimeError("private model failure")

        def document_vectors(self, text):
            raise AssertionError("must not be reached")

    result = rank_advanced_candidates(
        "retrieval evidence",
        candidates(),
        sparse_scorer=BrokenSparse(),
        late_interaction_scorer=BrokenLate(),
        reranker=lambda *_args: (_ for _ in ()).throw(RuntimeError("private")),
    )

    assert len(result) == 3
    assert all(item.components["splade"] == 0.0 for item in result)
    assert all(item.components["late_interaction"] == 0.0 for item in result)
    assert all(item.components["reranker"] == 0.0 for item in result)


def test_source_diversity_cap_is_enforced_after_advanced_fusion():
    values = [
        RetrievalCandidate(
            f"same-{index}",
            "retrieval evidence",
            "same-source",
            dense_score=1.0 - index * 0.01,
        )
        for index in range(4)
    ]
    values.append(
        RetrievalCandidate("other", "retrieval", "other-source", dense_score=0.5)
    )
    result = rank_advanced_candidates(
        "retrieval",
        values,
        top_k=3,
        config=AdvancedRetrievalConfig(max_per_source=1),
    )
    assert len(result) == 2
    assert len({item.candidate.source_id for item in result}) == 2
