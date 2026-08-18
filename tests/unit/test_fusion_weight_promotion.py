from __future__ import annotations

import hashlib

import pytest

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.fusion_weight_promotion import FusionWeightPromotionPolicy, qualify_learned_fusion_weights
from tools.corpus_fusion import RetrievalCandidate
from tools.cross_profile_fusion import CalibrationContract, ProfileRankedList, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator
from tools.promoted_learned_cross_profile_policy import run_promoted_learned_cross_profile_fusion
from training.cross_profile_fusion_fitting import FusionWeightExample, FusionWeightTrainingConfig, FusionWeightTrainingSpec, fit_fusion_weights


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(name: str) -> RetrieverScoreProfile:
    return RetrieverScoreProfile(name, name, sha(f"score:{name}"), sha(f"model:{name}"))


def contract() -> CalibrationContract:
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


def qreceipt(artifact):
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


def train(dense_cal, sparse_cal):
    spec = FusionWeightTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=dense_cal.calibration_contract_sha256,
        calibration_artifact_sha256s=(("dense", dense_cal.artifact_sha256), ("sparse", sparse_cal.artifact_sha256)),
        train_split_sha256=sha("fusion-train"),
        validation_split_sha256=sha("fusion-validation"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=FusionWeightTrainingConfig(epochs=30, batch_size=2, learning_rate=0.2, patience=6, seed=5),
    )
    examples = (
        FusionWeightExample({"dense": 0.95, "sparse": 0.55}, True),
        FusionWeightExample({"dense": 0.90, "sparse": 0.45}, True),
        FusionWeightExample({"dense": 0.10, "sparse": 0.55}, False),
        FusionWeightExample({"dense": 0.05, "sparse": 0.45}, False),
    )
    validation = (
        FusionWeightExample({"dense": 0.92, "sparse": 0.52}, True),
        FusionWeightExample({"dense": 0.08, "sparse": 0.48}, False),
    )
    return fit_fusion_weights(spec, examples, validation)


def promotion_examples():
    return (
        FusionWeightExample({"dense": 0.96, "sparse": 0.55}, True),
        FusionWeightExample({"dense": 0.91, "sparse": 0.45}, True),
        FusionWeightExample({"dense": 0.09, "sparse": 0.55}, False),
        FusionWeightExample({"dense": 0.04, "sparse": 0.45}, False),
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
    return FusionWeightPromotionPolicy(
        min_examples=4,
        min_positive_examples=2,
        min_negative_examples=2,
        max_log_loss=1.0,
        max_brier=0.5,
        min_log_loss_improvement=0.0,
        min_brier_improvement=0.0,
        max_single_profile_weight=1.0,
    )


def test_learned_weight_promotion_compares_against_uniform_baseline() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibrator(dense, c), calibrator(sparse, c)
    artifact = train(dense_cal, sparse_cal)
    receipt = qualify_learned_fusion_weights(
        artifact,
        promotion_examples(),
        evaluation_split_sha256=sha("promotion-split"),
        policy=promotion_policy(),
    )
    assert receipt.eligible
    assert receipt.learned_log_loss <= receipt.uniform_log_loss
    assert receipt.learned_brier <= receipt.uniform_brier


def test_promotion_blocks_one_class_evidence() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    artifact = train(calibrator(dense, c), calibrator(sparse, c))
    receipt = qualify_learned_fusion_weights(
        artifact,
        (
            FusionWeightExample({"dense": 0.95, "sparse": 0.55}, True),
            FusionWeightExample({"dense": 0.90, "sparse": 0.45}, True),
        ),
        evaluation_split_sha256=sha("promotion-split"),
        policy=FusionWeightPromotionPolicy(
            min_examples=2,
            min_positive_examples=1,
            min_negative_examples=1,
            max_log_loss=1.0,
            max_brier=1.0,
            max_single_profile_weight=1.0,
        ),
    )
    assert not receipt.eligible
    assert "insufficient_negative_examples" in receipt.reason_codes


def test_authoritative_execution_requires_eligible_promotion_receipt() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibrator(dense, c), calibrator(sparse, c)
    artifact = train(dense_cal, sparse_cal)
    promotion = qualify_learned_fusion_weights(
        artifact,
        promotion_examples(),
        evaluation_split_sha256=sha("promotion-split"),
        policy=promotion_policy(),
    )
    execution = run_promoted_learned_cross_profile_fusion(
        (ranked(dense, 0.9), ranked(sparse, 0.9)),
        learned_weights=artifact,
        promotion=promotion,
        calibrators={"dense": dense_cal, "sparse": sparse_cal},
        qualifications={"dense": qreceipt(dense_cal), "sparse": qreceipt(sparse_cal)},
    )
    assert execution.receipt.promotion_receipt_sha256 == promotion.receipt_sha256
    assert execution.receipt.learned_artifact_sha256 == artifact.artifact_sha256


def test_authoritative_execution_rejects_promotion_for_different_artifact() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibrator(dense, c), calibrator(sparse, c)
    artifact = train(dense_cal, sparse_cal)
    promotion = qualify_learned_fusion_weights(
        artifact,
        promotion_examples(),
        evaluation_split_sha256=sha("promotion-split"),
        policy=promotion_policy(),
    )
    other_dense = fit_isotonic_calibrator(
        profile=dense,
        contract=c,
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(1.0, True), ScoreCalibrationExample(2.0, True)),
    )
    other_artifact = train(other_dense, sparse_cal)
    with pytest.raises(ValueError, match="does not cover"):
        run_promoted_learned_cross_profile_fusion(
            (ranked(dense, 0.9), ranked(sparse, 0.9)),
            learned_weights=other_artifact,
            promotion=promotion,
            calibrators={"dense": other_dense, "sparse": sparse_cal},
            qualifications={"dense": qreceipt(other_dense), "sparse": qreceipt(sparse_cal)},
        )
