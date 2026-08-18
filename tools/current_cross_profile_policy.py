"""Freshness-required serving entrypoints for promoted cross-profile fusion policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt
from evaluation.cross_profile_calibration_drift import CalibrationDriftDecision
from evaluation.fusion_weight_promotion import FusionWeightPromotionReceipt
from evaluation.listwise_fusion_promotion import ListwiseFusionPromotionReceipt
from tools.cross_profile_fusion import IsotonicCalibrationArtifact, ProfileRankedList
from tools.promoted_learned_cross_profile_policy import PromotedLearnedFusionExecution, run_promoted_learned_cross_profile_fusion
from tools.promoted_listwise_cross_profile_policy import PromotedListwiseFusionExecution, run_promoted_listwise_cross_profile_fusion
from training.cross_profile_fusion_fitting import LearnedFusionWeightArtifact
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


def validate_current_calibrators(
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    drift_decisions: Mapping[str, CalibrationDriftDecision],
) -> tuple[tuple[str, str], ...]:
    if set(calibrators) != set(drift_decisions):
        raise ValueError("drift decisions must exactly cover runtime calibrators")
    rows = []
    for profile_id in sorted(calibrators):
        calibrator = calibrators[profile_id]
        decision = drift_decisions[profile_id]
        if decision.profile_id != profile_id:
            raise ValueError("drift decision mapping key does not match profile id")
        if decision.profile_sha256 != calibrator.profile.profile_sha256 or decision.artifact_sha256 != calibrator.artifact_sha256:
            raise ValueError("drift decision does not cover runtime calibrator lineage")
        if decision.action != "calibrated_ok":
            raise ValueError(f"profile {profile_id!r} calibration is not current; use RRF-only fallback")
        rows.append((profile_id, decision.decision_sha256))
    return tuple(rows)


@dataclass(frozen=True)
class CurrentFusionServingReceipt:
    underlying_execution_sha256: str
    drift_decision_sha256s: tuple[tuple[str, str], ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "underlying_execution_sha256", _sha(self.underlying_execution_sha256, "underlying_execution_sha256"))
        rows = tuple(sorted((profile, _sha(digest, "drift decision sha256")) for profile, digest in self.drift_decision_sha256s))
        if not rows or len({profile for profile, _ in rows}) != len(rows):
            raise ValueError("drift decision identities must be unique and non-empty")
        object.__setattr__(self, "drift_decision_sha256s", rows)
        payload = {
            "schema": "rigorousrag-current-cross-profile-serving/v1",
            "underlying_execution_sha256": self.underlying_execution_sha256,
            "drift_decision_sha256s": rows,
        }
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if _digest(payload) != provided:
            raise ValueError("receipt_sha256 does not match current serving receipt")
        object.__setattr__(self, "receipt_sha256", provided)


@dataclass(frozen=True)
class CurrentPointwiseFusionExecution:
    execution: PromotedLearnedFusionExecution
    receipt: CurrentFusionServingReceipt


@dataclass(frozen=True)
class CurrentListwiseFusionExecution:
    execution: PromotedListwiseFusionExecution
    receipt: CurrentFusionServingReceipt


def _receipt(underlying: str, decisions: tuple[tuple[str, str], ...]) -> CurrentFusionServingReceipt:
    payload = {
        "schema": "rigorousrag-current-cross-profile-serving/v1",
        "underlying_execution_sha256": underlying,
        "drift_decision_sha256s": decisions,
    }
    return CurrentFusionServingReceipt(underlying, decisions, _digest(payload))


def run_current_pointwise_fusion(
    ranked_lists: Sequence[ProfileRankedList],
    *,
    learned_weights: LearnedFusionWeightArtifact,
    promotion: FusionWeightPromotionReceipt,
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    qualifications: Mapping[str, CalibrationQualificationReceipt],
    drift_decisions: Mapping[str, CalibrationDriftDecision],
    max_fused_candidates: int = 1000,
    max_per_document: int = 3,
    max_per_source: int | None = None,
    rrf_k: int = 60,
) -> CurrentPointwiseFusionExecution:
    decisions = validate_current_calibrators(calibrators, drift_decisions)
    execution = run_promoted_learned_cross_profile_fusion(
        ranked_lists,
        learned_weights=learned_weights,
        promotion=promotion,
        calibrators=calibrators,
        qualifications=qualifications,
        max_fused_candidates=max_fused_candidates,
        max_per_document=max_per_document,
        max_per_source=max_per_source,
        rrf_k=rrf_k,
    )
    return CurrentPointwiseFusionExecution(execution, _receipt(execution.receipt.receipt_sha256, decisions))


def run_current_listwise_fusion(
    ranked_lists: Sequence[ProfileRankedList],
    *,
    learned_weights: LearnedListwiseFusionArtifact,
    promotion: ListwiseFusionPromotionReceipt,
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    qualifications: Mapping[str, CalibrationQualificationReceipt],
    drift_decisions: Mapping[str, CalibrationDriftDecision],
    max_fused_candidates: int = 1000,
    max_per_document: int = 3,
    max_per_source: int | None = None,
    rrf_k: int = 60,
) -> CurrentListwiseFusionExecution:
    decisions = validate_current_calibrators(calibrators, drift_decisions)
    execution = run_promoted_listwise_cross_profile_fusion(
        ranked_lists,
        learned_weights=learned_weights,
        promotion=promotion,
        calibrators=calibrators,
        qualifications=qualifications,
        max_fused_candidates=max_fused_candidates,
        max_per_document=max_per_document,
        max_per_source=max_per_source,
        rrf_k=rrf_k,
    )
    return CurrentListwiseFusionExecution(execution, _receipt(execution.receipt.receipt_sha256, decisions))


__all__ = [
    "CurrentFusionServingReceipt",
    "CurrentListwiseFusionExecution",
    "CurrentPointwiseFusionExecution",
    "run_current_listwise_fusion",
    "run_current_pointwise_fusion",
    "validate_current_calibrators",
]
