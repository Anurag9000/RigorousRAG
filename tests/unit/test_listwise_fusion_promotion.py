from __future__ import annotations

import hashlib

import pytest

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.listwise_fusion_promotion import ListwiseFusionPromotionPolicy, qualify_listwise_fusion_weights
from tools.corpus_fusion import RetrievalCandidate
from tools.cross_profile_fusion import CalibrationContract, ProfileRankedList, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator
from tools.promoted_listwise_cross_profile_policy import run_promoted_listwise_cross_profile_fusion
from training.cross_profile_listwise_fusion import FusionRankingCandidate, FusionRankingQuery, ListwiseFusionTrainingConfig, ListwiseFusionTrainingSpec, fit_listwise_fusion_weights


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(name: str):
    return RetrieverScoreProfile(name, name, sha(f"score:{name}"), sha(f"model:{name}"))


def contract():
    return CalibrationContract(sha("dataset"), sha("split"), sha("relevance"), sha("universe"), "scientific")


def calibrator(p, c):
    return fit_isotonic_calibrator(
        profile=p,
        contract=c,
        examples=(
            ScoreCalibrationExample(0.0, False),
            ScoreCalibrationExample(0.1, False),
            ScoreCalibrationExample(0.9, True),
            ScoreCalibrationExample(1.0, True),
        ),
    )


def qualification(artifact):
    return qualify_calibrator(
        artifact,
        (
            ScoreCalibrationExample(0.05, False),
            ScoreCalibrationExample(0.15, False),
            ScoreCalibrationExample(0.85, True),
            ScoreCalibrationExample(0.95, True),
        ),
        policy=CalibrationQualificationPolicy(
            min_examples=4,
            min_positive_examples=2,
            min_negative_examples=2,
            max_brier=0.25,
            max_ece=0.25,
            ece_bin_count=2,
        ),
    )


def ranking_query(tag: str):
    return FusionRankingQuery(
        sha(f"query:{tag}"),
        (
            FusionRankingCandidate("good", {"dense": 0.95, "sparse": 0.55}, 2.0),
            FusionRankingCandidate("mid", {"dense": 0.55, "sparse": 0.60}, 1.0),
            FusionRankingCandidate("bad", {"dense": 0.05, "sparse": 0.45}, 0.0),
        ),
    )


def train(dense_cal, sparse_cal):
    spec = ListwiseFusionTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=dense_cal.calibration_contract_sha256,
        calibration_artifact_sha256s=(("dense", dense_cal.artifact_sha256), ("sparse", sparse_cal.artifact_sha256)),
        train_split_sha256=sha("listwise-train"),
        validation_split_sha256=sha("listwise-validation"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=ListwiseFusionTrainingConfig(epochs=30, batch_size=1, learning_rate=0.2, patience=6, seed=9),
    )
    return fit_listwise_fusion_weights(
        spec,
        (ranking_query("train-a"), ranking_query("train-b")),
        (ranking_query("validation"),),
    )


def ranked(p, score):
    return ProfileRankedList(
        f"{p.profile_id}-list",
        p,
        (
            RetrievalCandidate(
                candidate_id=f"{p.profile_id}-candidate",
                corpus_id="papers",
                retriever_id=p.profile_id,
                document_id="doc",
                chunk_id="chunk",
                rank=1,
                raw_score=score,
            ),
        ),
    )


def promotion_policy():
    return ListwiseFusionPromotionPolicy(
        min_queries=2,
        ndcg_k=3,
        min_mean_ndcg=0.0,
        min_mean_ndcg_improvement=0.0,
        max_query_regression_fraction=1.0,
        max_single_profile_weight=1.0,
    )


def test_listwise_promotion_emits_rank_aware_evidence() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibrator(dense, c), calibrator(sparse, c)
    artifact = train(dense_cal, sparse_cal)
    receipt = qualify_listwise_fusion_weights(
        artifact,
        (ranking_query("promotion-a"), ranking_query("promotion-b")),
        evaluation_split_sha256=sha("promotion-split"),
        policy=promotion_policy(),
    )
    assert receipt.eligible
    assert receipt.learned_mean_ndcg >= receipt.uniform_mean_ndcg
    assert 0.0 <= receipt.regression_fraction <= 1.0


def test_listwise_promotion_blocks_insufficient_query_evidence() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    artifact = train(calibrator(dense, c), calibrator(sparse, c))
    receipt = qualify_listwise_fusion_weights(
        artifact,
        (ranking_query("only"),),
        evaluation_split_sha256=sha("promotion-split"),
        policy=ListwiseFusionPromotionPolicy(min_queries=2, max_query_regression_fraction=1.0, max_single_profile_weight=1.0),
    )
    assert not receipt.eligible
    assert "insufficient_evaluation_queries" in receipt.reason_codes


def test_promoted_listwise_runtime_requires_exact_calibration_lineage() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibrator(dense, c), calibrator(sparse, c)
    artifact = train(dense_cal, sparse_cal)
    promotion = qualify_listwise_fusion_weights(
        artifact,
        (ranking_query("promotion-a"), ranking_query("promotion-b")),
        evaluation_split_sha256=sha("promotion-split"),
        policy=promotion_policy(),
    )
    execution = run_promoted_listwise_cross_profile_fusion(
        (ranked(dense, 0.9), ranked(sparse, 0.9)),
        learned_weights=artifact,
        promotion=promotion,
        calibrators={"dense": dense_cal, "sparse": sparse_cal},
        qualifications={"dense": qualification(dense_cal), "sparse": qualification(sparse_cal)},
    )
    assert execution.receipt.learned_artifact_sha256 == artifact.artifact_sha256
    assert execution.receipt.promotion_receipt_sha256 == promotion.receipt_sha256


def test_promoted_listwise_runtime_rejects_replacement_calibrator() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibrator(dense, c), calibrator(sparse, c)
    artifact = train(dense_cal, sparse_cal)
    promotion = qualify_listwise_fusion_weights(
        artifact,
        (ranking_query("promotion-a"), ranking_query("promotion-b")),
        evaluation_split_sha256=sha("promotion-split"),
        policy=promotion_policy(),
    )
    replacement = fit_isotonic_calibrator(
        profile=dense,
        contract=c,
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(0.5, True), ScoreCalibrationExample(1.0, True)),
    )
    with pytest.raises(ValueError, match="training lineage"):
        run_promoted_listwise_cross_profile_fusion(
            (ranked(dense, 0.9), ranked(sparse, 0.9)),
            learned_weights=artifact,
            promotion=promotion,
            calibrators={"dense": replacement, "sparse": sparse_cal},
            qualifications={"dense": qualification(replacement), "sparse": qualification(sparse_cal)},
        )
