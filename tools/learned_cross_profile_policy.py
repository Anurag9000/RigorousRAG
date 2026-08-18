"""Bind learned cross-profile weights to qualified calibrated fusion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt, run_qualified_calibrated_fusion
from tools.cross_profile_fusion import CrossProfileFusionMode, CrossProfileFusionPolicy, IsotonicCalibrationArtifact, ProfileRankedList
from tools.cross_profile_fusion_governance import GovernedCrossProfileFusionRun
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
class LearnedFusionExecutionReceipt:
    learned_artifact_sha256: str
    governed_receipt_sha256: str
    calibration_contract_sha256: str
    profile_artifact_sha256s: tuple[tuple[str, str], ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "learned_artifact_sha256", _sha(self.learned_artifact_sha256, "learned_artifact_sha256"))
        object.__setattr__(self, "governed_receipt_sha256", _sha(self.governed_receipt_sha256, "governed_receipt_sha256"))
        object.__setattr__(self, "calibration_contract_sha256", _sha(self.calibration_contract_sha256, "calibration_contract_sha256"))
        artifacts = tuple(sorted((profile, _sha(digest, "profile artifact sha256")) for profile, digest in self.profile_artifact_sha256s))
        object.__setattr__(self, "profile_artifact_sha256s", artifacts)
        payload = {
            "schema": "rigorousrag-learned-cross-profile-fusion-execution/v1",
            "learned_artifact_sha256": self.learned_artifact_sha256,
            "governed_receipt_sha256": self.governed_receipt_sha256,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "profile_artifact_sha256s": self.profile_artifact_sha256s,
        }
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if _digest(payload) != provided:
            raise ValueError("receipt_sha256 does not match execution receipt content")
        object.__setattr__(self, "receipt_sha256", provided)


@dataclass(frozen=True)
class LearnedFusionExecution:
    run: GovernedCrossProfileFusionRun
    receipt: LearnedFusionExecutionReceipt


def run_learned_cross_profile_fusion(
    ranked_lists: Sequence[ProfileRankedList],
    *,
    learned_weights: LearnedFusionWeightArtifact,
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    qualifications: Mapping[str, CalibrationQualificationReceipt],
    max_fused_candidates: int = 1000,
    max_per_document: int = 3,
    max_per_source: int | None = None,
    rrf_k: int = 60,
) -> LearnedFusionExecution:
    lists = tuple(ranked_lists)
    if not lists:
        raise ValueError("ranked_lists must be non-empty")
    learned_profiles = {profile for profile, _ in learned_weights.profile_weights}
    if {item.profile.profile_id for item in lists} != learned_profiles:
        raise ValueError("ranked profile set differs from learned weight artifact")
    if set(calibrators) != learned_profiles or set(qualifications) != learned_profiles:
        raise ValueError("calibrators and qualifications must exactly cover learned profiles")
    expected_artifacts = dict(learned_weights.calibration_artifact_sha256s)
    for profile_id in learned_profiles:
        calibrator = calibrators[profile_id]
        qualification = qualifications[profile_id]
        if calibrator.artifact_sha256 != expected_artifacts[profile_id]:
            raise ValueError("calibrator differs from learned training lineage")
        if calibrator.calibration_contract_sha256 != learned_weights.calibration_contract_sha256:
            raise ValueError("calibration contract differs from learned training lineage")
        if qualification.artifact_sha256 != calibrator.artifact_sha256 or not qualification.eligible:
            raise ValueError("qualification does not authorize the supplied calibrator")

    policy = CrossProfileFusionPolicy(
        mode=CrossProfileFusionMode.CALIBRATED_LOGIT,
        profile_weights=dict(learned_weights.profile_weights),
        max_fused_candidates=max_fused_candidates,
        max_per_document=max_per_document,
        max_per_source=max_per_source,
        rrf_k=rrf_k,
    )
    run = run_qualified_calibrated_fusion(lists, calibrators=calibrators, qualifications=qualifications, policy=policy)
    payload = {
        "schema": "rigorousrag-learned-cross-profile-fusion-execution/v1",
        "learned_artifact_sha256": learned_weights.artifact_sha256,
        "governed_receipt_sha256": run.receipt.receipt_sha256,
        "calibration_contract_sha256": learned_weights.calibration_contract_sha256,
        "profile_artifact_sha256s": learned_weights.calibration_artifact_sha256s,
    }
    receipt = LearnedFusionExecutionReceipt(
        learned_artifact_sha256=payload["learned_artifact_sha256"],
        governed_receipt_sha256=payload["governed_receipt_sha256"],
        calibration_contract_sha256=payload["calibration_contract_sha256"],
        profile_artifact_sha256s=payload["profile_artifact_sha256s"],
        receipt_sha256=_digest(payload),
    )
    return LearnedFusionExecution(run=run, receipt=receipt)


__all__ = ["LearnedFusionExecution", "LearnedFusionExecutionReceipt", "run_learned_cross_profile_fusion"]
