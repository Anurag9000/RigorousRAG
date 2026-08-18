"""Promotion-required runtime binding for listwise-learned cross-profile fusion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt, run_qualified_calibrated_fusion
from evaluation.listwise_fusion_promotion import ListwiseFusionPromotionReceipt
from tools.cross_profile_fusion import CrossProfileFusionMode, CrossProfileFusionPolicy, IsotonicCalibrationArtifact, ProfileRankedList
from tools.cross_profile_fusion_governance import GovernedCrossProfileFusionRun
from training.cross_profile_listwise_fusion import LearnedListwiseFusionArtifact


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


@dataclass(frozen=True)
class PromotedListwiseFusionReceipt:
    learned_artifact_sha256: str
    promotion_receipt_sha256: str
    governed_fusion_receipt_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("learned_artifact_sha256", "promotion_receipt_sha256", "governed_fusion_receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        payload = {
            "schema": "rigorousrag-promoted-listwise-fusion-execution/v1",
            "learned_artifact_sha256": self.learned_artifact_sha256,
            "promotion_receipt_sha256": self.promotion_receipt_sha256,
            "governed_fusion_receipt_sha256": self.governed_fusion_receipt_sha256,
        }
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if _digest(payload) != provided:
            raise ValueError("receipt_sha256 does not match promoted listwise execution")
        object.__setattr__(self, "receipt_sha256", provided)


@dataclass(frozen=True)
class PromotedListwiseFusionExecution:
    run: GovernedCrossProfileFusionRun
    receipt: PromotedListwiseFusionReceipt


def run_promoted_listwise_cross_profile_fusion(
    ranked_lists: Sequence[ProfileRankedList],
    *,
    learned_weights: LearnedListwiseFusionArtifact,
    promotion: ListwiseFusionPromotionReceipt,
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    qualifications: Mapping[str, CalibrationQualificationReceipt],
    max_fused_candidates: int = 1000,
    max_per_document: int = 3,
    max_per_source: int | None = None,
    rrf_k: int = 60,
) -> PromotedListwiseFusionExecution:
    if not isinstance(learned_weights, LearnedListwiseFusionArtifact):
        raise ValueError("learned_weights must be LearnedListwiseFusionArtifact")
    if not isinstance(promotion, ListwiseFusionPromotionReceipt):
        raise ValueError("promotion must be ListwiseFusionPromotionReceipt")
    if promotion.learned_artifact_sha256 != learned_weights.artifact_sha256:
        raise ValueError("promotion receipt does not cover learned listwise artifact")
    if not promotion.eligible:
        raise ValueError("listwise fusion artifact has not passed promotion gates")
    profiles = {profile for profile, _ in learned_weights.profile_weights}
    if {item.profile.profile_id for item in ranked_lists} != profiles:
        raise ValueError("runtime profile set differs from learned listwise artifact")
    if set(calibrators) != profiles or set(qualifications) != profiles:
        raise ValueError("calibrators and qualifications must exactly cover learned profiles")
    expected_artifacts = dict(learned_weights.calibration_artifact_sha256s)
    for profile_id in profiles:
        calibrator = calibrators[profile_id]
        qualification = qualifications[profile_id]
        if calibrator.artifact_sha256 != expected_artifacts[profile_id]:
            raise ValueError("runtime calibrator differs from listwise training lineage")
        if calibrator.calibration_contract_sha256 != learned_weights.calibration_contract_sha256:
            raise ValueError("runtime calibration contract differs from listwise training lineage")
        if qualification.artifact_sha256 != calibrator.artifact_sha256 or not qualification.eligible:
            raise ValueError("qualification does not authorize runtime calibrator")
    policy = CrossProfileFusionPolicy(
        mode=CrossProfileFusionMode.CALIBRATED_LOGIT,
        profile_weights=dict(learned_weights.profile_weights),
        max_fused_candidates=max_fused_candidates,
        max_per_document=max_per_document,
        max_per_source=max_per_source,
        rrf_k=rrf_k,
    )
    run = run_qualified_calibrated_fusion(
        tuple(ranked_lists),
        calibrators=calibrators,
        qualifications=qualifications,
        policy=policy,
    )
    payload = {
        "schema": "rigorousrag-promoted-listwise-fusion-execution/v1",
        "learned_artifact_sha256": learned_weights.artifact_sha256,
        "promotion_receipt_sha256": promotion.receipt_sha256,
        "governed_fusion_receipt_sha256": run.receipt.receipt_sha256,
    }
    receipt = PromotedListwiseFusionReceipt(
        learned_artifact_sha256=payload["learned_artifact_sha256"],
        promotion_receipt_sha256=payload["promotion_receipt_sha256"],
        governed_fusion_receipt_sha256=payload["governed_fusion_receipt_sha256"],
        receipt_sha256=_digest(payload),
    )
    return PromotedListwiseFusionExecution(run=run, receipt=receipt)


__all__ = ["PromotedListwiseFusionExecution", "PromotedListwiseFusionReceipt", "run_promoted_listwise_cross_profile_fusion"]
