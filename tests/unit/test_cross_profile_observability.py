from __future__ import annotations

import hashlib

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.cross_profile_observability import (
    observations_from_calibration_qualification,
    observations_from_listwise_fusion_promotion,
    observations_from_pointwise_fusion_promotion,
)
from evaluation.fusion_weight_promotion import FusionWeightPromotionPolicy, qualify_learned_fusion_weights
from evaluation.listwise_fusion_promotion import ListwiseFusionPromotionPolicy, qualify_listwise_fusion_weights
from evaluation.quality_observability import QualityProvenance, QualitySLO, QualitySnapshot, QualityWindow, build_quality_dashboard
from tools.cross_profile_fusion import CalibrationContract, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator
from training.cross_profile_fusion_fitting import FusionWeightExample, FusionWeightTrainingConfig, FusionWeightTrainingSpec, fit_fusion_weights
from training.cross_profile_listwise_fusion import FusionRankingCandidate, FusionRankingQuery, ListwiseFusionTrainingConfig, ListwiseFusionTrainingSpec, fit_listwise_fusion_weights


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(name: str) -> RetrieverScoreProfile:
    return RetrieverScoreProfile(name, name, sha(f"score:{name}"), sha(f"model:{name}"))


def contract() -> CalibrationContract:
    return CalibrationContract(sha("dataset"), sha("split"), sha("relevance"), sha("universe"), "science")


def calibration_artifact(p: RetrieverScoreProfile, c: CalibrationContract):
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


def calibration_receipt(artifact):
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


def pointwise_artifact(dense_cal, sparse_cal):
    spec = FusionWeightTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=dense_cal.calibration_contract_sha256,
        calibration_artifact_sha256s=(("dense", dense_cal.artifact_sha256), ("sparse", sparse_cal.artifact_sha256)),
        train_split_sha256=sha("point-train"),
        validation_split_sha256=sha("point-valid"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=FusionWeightTrainingConfig(epochs=10, batch_size=2, learning_rate=0.2, patience=3),
    )
    train = (
        FusionWeightExample({"dense": 0.95, "sparse": 0.55}, True),
        FusionWeightExample({"dense": 0.90, "sparse": 0.45}, True),
        FusionWeightExample({"dense": 0.10, "sparse": 0.55}, False),
        FusionWeightExample({"dense": 0.05, "sparse": 0.45}, False),
    )
    valid = (
        FusionWeightExample({"dense": 0.92, "sparse": 0.52}, True),
        FusionWeightExample({"dense": 0.08, "sparse": 0.48}, False),
    )
    return fit_fusion_weights(spec, train, valid), train


def ranking_query(tag: str) -> FusionRankingQuery:
    return FusionRankingQuery(
        sha(f"query:{tag}"),
        (
            FusionRankingCandidate("good", {"dense": 0.95, "sparse": 0.55}, 2.0),
            FusionRankingCandidate("mid", {"dense": 0.55, "sparse": 0.60}, 1.0),
            FusionRankingCandidate("bad", {"dense": 0.05, "sparse": 0.45}, 0.0),
        ),
    )


def listwise_artifact(dense_cal, sparse_cal):
    spec = ListwiseFusionTrainingSpec(
        profile_ids=("dense", "sparse"),
        calibration_contract_sha256=dense_cal.calibration_contract_sha256,
        calibration_artifact_sha256s=(("dense", dense_cal.artifact_sha256), ("sparse", sparse_cal.artifact_sha256)),
        train_split_sha256=sha("list-train"),
        validation_split_sha256=sha("list-valid"),
        source_revision="0123456789abcdef0123456789abcdef01234567",
        config=ListwiseFusionTrainingConfig(epochs=10, batch_size=1, learning_rate=0.2, patience=3),
    )
    return fit_listwise_fusion_weights(spec, (ranking_query("train-a"), ranking_query("train-b")), (ranking_query("valid"),))


def test_calibration_observations_use_safe_digest_dimensions() -> None:
    p = profile("dense")
    receipt = calibration_receipt(calibration_artifact(p, contract()))
    observations = observations_from_calibration_qualification(receipt)
    names = {item.name for item in observations}
    assert "cross_profile.calibration.brier" in names
    assert "cross_profile.calibration.ece" in names
    for item in observations:
        tags = dict(item.tags)
        assert tags["profile_id"] == "dense"
        assert tags["artifact_digest"] == receipt.artifact_sha256
        assert all("query" not in key and "text" not in key for key in tags)


def test_pointwise_and_listwise_promotion_metrics_flow_into_slo_dashboard() -> None:
    c = contract()
    dense, sparse = profile("dense"), profile("sparse")
    dense_cal, sparse_cal = calibration_artifact(dense, c), calibration_artifact(sparse, c)
    point_artifact, point_examples = pointwise_artifact(dense_cal, sparse_cal)
    point_receipt = qualify_learned_fusion_weights(
        point_artifact,
        point_examples,
        evaluation_split_sha256=sha("point-promotion"),
        policy=FusionWeightPromotionPolicy(
            min_examples=4,
            min_positive_examples=2,
            min_negative_examples=2,
            max_log_loss=1.0,
            max_brier=1.0,
            max_single_profile_weight=1.0,
        ),
    )
    list_artifact = listwise_artifact(dense_cal, sparse_cal)
    list_receipt = qualify_listwise_fusion_weights(
        list_artifact,
        (ranking_query("promotion-a"), ranking_query("promotion-b")),
        evaluation_split_sha256=sha("list-promotion"),
        policy=ListwiseFusionPromotionPolicy(min_queries=2, max_query_regression_fraction=1.0, max_single_profile_weight=1.0),
    )
    metrics = observations_from_pointwise_fusion_promotion(point_receipt) + observations_from_listwise_fusion_promotion(list_receipt)
    snapshot = QualitySnapshot(
        QualityWindow(1.0, 2.0, 3.0),
        QualityProvenance(
            run_id="fusion-run",
            system_id="rigorousrag",
            domain_id="science",
            dataset_manifest_digest=sha("dataset-manifest"),
            split_digest=sha("quality-split"),
            evaluation_contract_digest=sha("quality-contract"),
            code_revision="0123456789abcdef0123456789abcdef01234567",
            retrieval_stack_digest=sha("fusion-stack"),
        ),
        metrics,
    )
    dashboard = build_quality_dashboard(
        snapshot,
        (
            QualitySLO("pointwise eligible", "cross_profile.fusion.eligible", ">=", 1.0, tag_match=(("metric_family", "cross_profile_pointwise_fusion"),)),
            QualitySLO("listwise regressions", "cross_profile.listwise.regression_fraction", "<=", 1.0, tag_match=(("metric_family", "cross_profile_listwise_fusion"),)),
        ),
    )
    assert dashboard.healthy
    assert not dashboard.failed_slos
