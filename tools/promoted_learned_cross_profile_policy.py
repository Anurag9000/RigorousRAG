"""Promotion-required entrypoint for learned heterogeneous-retriever fusion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt
from evaluation.fusion_weight_promotion import FusionWeightPromotionReceipt
from tools.cross_profile_fusion import IsotonicCalibrationArtifact, ProfileRankedList
from tools.learned_cross_profile_policy import LearnedFusionExecution, run_learned_cross_profile_fusion
from training.cross_profile_fusion_fitting import LearnedFusionWeightArtifact


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
class PromotedLearnedFusionReceipt:
    learned_artifact_sha256: str
    promotion_receipt_sha256: str
    execution_receipt_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("learned_artifact_sha256", "promotion_receipt_sha256", "execution_receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        payload = {
            "schema": "rigorousrag-promoted-learned-fusion-execution/v1",
            "learned_artifact_sha256": self.learned_artifact_sha256,
            "promotion_receipt_sha256": self.promotion_receipt_sha256,
            "execution_receipt_sha256": self.execution_receipt_sha256,
        }
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if _digest(payload) != provided:
            raise ValueError("receipt_sha256 does not match promoted execution content")
        object.__setattr__(self, "receipt_sha256", provided)


@dataclass(frozen=True)
class PromotedLearnedFusionExecution:
    execution: LearnedFusionExecution
    receipt: PromotedLearnedFusionReceipt


def run_promoted_learned_cross_profile_fusion(
    ranked_lists: Sequence[ProfileRankedList],
    *,
    learned_weights: LearnedFusionWeightArtifact,
    promotion: FusionWeightPromotionReceipt,
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    qualifications: Mapping[str, CalibrationQualificationReceipt],
    max_fused_candidates: int = 1000,
    max_per_document: int = 3,
    max_per_source: int | None = None,
    rrf_k: int = 60,
) -> PromotedLearnedFusionExecution:
    if not isinstance(learned_weights, LearnedFusionWeightArtifact):
        raise ValueError("learned_weights must be LearnedFusionWeightArtifact")
    if not isinstance(promotion, FusionWeightPromotionReceipt):
        raise ValueError("promotion must be FusionWeightPromotionReceipt")
    if promotion.learned_artifact_sha256 != learned_weights.artifact_sha256:
        raise ValueError("promotion receipt does not cover the supplied learned artifact")
    if not promotion.eligible:
        raise ValueError("learned fusion artifact has not passed promotion gates")
    execution = run_learned_cross_profile_fusion(
        ranked_lists,
        learned_weights=learned_weights,
        calibrators=calibrators,
        qualifications=qualifications,
        max_fused_candidates=max_fused_candidates,
        max_per_document=max_per_document,
        max_per_source=max_per_source,
        rrf_k=rrf_k,
    )
    payload = {
        "schema": "rigorousrag-promoted-learned-fusion-execution/v1",
        "learned_artifact_sha256": learned_weights.artifact_sha256,
        "promotion_receipt_sha256": promotion.receipt_sha256,
        "execution_receipt_sha256": execution.receipt.receipt_sha256,
    }
    receipt = PromotedLearnedFusionReceipt(
        learned_artifact_sha256=payload["learned_artifact_sha256"],
        promotion_receipt_sha256=payload["promotion_receipt_sha256"],
        execution_receipt_sha256=payload["execution_receipt_sha256"],
        receipt_sha256=_digest(payload),
    )
    return PromotedLearnedFusionExecution(execution=execution, receipt=receipt)


__all__ = ["PromotedLearnedFusionExecution", "PromotedLearnedFusionReceipt", "run_promoted_learned_cross_profile_fusion"]
