"""Privacy-safe observability adapters for heterogeneous retrieval fusion evidence."""

from __future__ import annotations

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt
from evaluation.fusion_weight_promotion import FusionWeightPromotionReceipt
from evaluation.listwise_fusion_promotion import ListwiseFusionPromotionReceipt
from evaluation.quality_observability import MetricObservation


def observations_from_calibration_qualification(
    receipt: CalibrationQualificationReceipt,
) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, CalibrationQualificationReceipt):
        raise ValueError("receipt must be CalibrationQualificationReceipt")
    tags = (
        ("profile_id", receipt.profile_id),
        ("artifact_digest", receipt.artifact_sha256),
        ("metric_family", "cross_profile_calibration"),
    )
    return (
        MetricObservation("cross_profile.calibration.brier", receipt.brier, "lower", "ratio", receipt.example_count, "evaluation.cross_profile_calibration", tags),
        MetricObservation("cross_profile.calibration.ece", receipt.ece, "lower", "ratio", receipt.example_count, "evaluation.cross_profile_calibration", tags),
        MetricObservation("cross_profile.calibration.eligible", float(receipt.eligible), "higher", "ratio", receipt.example_count, "evaluation.cross_profile_calibration", tags),
        MetricObservation("cross_profile.calibration.positive_count", float(receipt.positive_count), "neutral", "count", receipt.example_count, "evaluation.cross_profile_calibration", tags),
        MetricObservation("cross_profile.calibration.negative_count", float(receipt.negative_count), "neutral", "count", receipt.example_count, "evaluation.cross_profile_calibration", tags),
    )


def observations_from_pointwise_fusion_promotion(
    receipt: FusionWeightPromotionReceipt,
) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, FusionWeightPromotionReceipt):
        raise ValueError("receipt must be FusionWeightPromotionReceipt")
    tags = (
        ("artifact_digest", receipt.learned_artifact_sha256),
        ("metric_family", "cross_profile_pointwise_fusion"),
    )
    return (
        MetricObservation("cross_profile.fusion.log_loss", receipt.learned_log_loss, "lower", "loss", receipt.example_count, "evaluation.fusion_weight_promotion", tags),
        MetricObservation("cross_profile.fusion.log_loss_improvement", receipt.uniform_log_loss - receipt.learned_log_loss, "higher", "loss", receipt.example_count, "evaluation.fusion_weight_promotion", tags),
        MetricObservation("cross_profile.fusion.brier", receipt.learned_brier, "lower", "ratio", receipt.example_count, "evaluation.fusion_weight_promotion", tags),
        MetricObservation("cross_profile.fusion.brier_improvement", receipt.uniform_brier - receipt.learned_brier, "higher", "ratio", receipt.example_count, "evaluation.fusion_weight_promotion", tags),
        MetricObservation("cross_profile.fusion.eligible", float(receipt.eligible), "higher", "ratio", receipt.example_count, "evaluation.fusion_weight_promotion", tags),
    )


def observations_from_listwise_fusion_promotion(
    receipt: ListwiseFusionPromotionReceipt,
) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, ListwiseFusionPromotionReceipt):
        raise ValueError("receipt must be ListwiseFusionPromotionReceipt")
    tags = (
        ("artifact_digest", receipt.learned_artifact_sha256),
        ("metric_family", "cross_profile_listwise_fusion"),
    )
    return (
        MetricObservation("cross_profile.listwise.mean_ndcg", receipt.learned_mean_ndcg, "higher", "ratio", receipt.query_count, "evaluation.listwise_fusion_promotion", tags),
        MetricObservation("cross_profile.listwise.ndcg_improvement", receipt.learned_mean_ndcg - receipt.uniform_mean_ndcg, "higher", "ratio", receipt.query_count, "evaluation.listwise_fusion_promotion", tags),
        MetricObservation("cross_profile.listwise.regression_fraction", receipt.regression_fraction, "lower", "ratio", receipt.query_count, "evaluation.listwise_fusion_promotion", tags),
        MetricObservation("cross_profile.listwise.eligible", float(receipt.eligible), "higher", "ratio", receipt.query_count, "evaluation.listwise_fusion_promotion", tags),
    )


__all__ = [
    "observations_from_calibration_qualification",
    "observations_from_listwise_fusion_promotion",
    "observations_from_pointwise_fusion_promotion",
]
