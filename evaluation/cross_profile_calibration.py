"""Qualification gates for heterogeneous-retriever score calibration artifacts.

Fitting a monotone calibrator is not itself a promotion decision.  This module evaluates
an immutable calibrator on a separately identified held-out set, records class support,
Brier/ECE, and emits a content-addressed qualification receipt.  Cross-profile fusion can
therefore require evidence that every score profile is calibrated well enough under the
same semantic calibration contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from tools.cross_profile_fusion import (
    CrossProfileFusionMode,
    CrossProfileFusionPolicy,
    IsotonicCalibrationArtifact,
    ScoreCalibrationExample,
    evaluate_isotonic_calibrator,
)
from tools.cross_profile_fusion_governance import (
    GovernedCrossProfileFusionRun,
    run_governed_cross_profile_fusion,
)
from tools.cross_profile_fusion import ProfileRankedList


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return selected


def _probability(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1].")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1].")
    return selected


@dataclass(frozen=True)
class CalibrationQualificationPolicy:
    min_examples: int = 200
    min_positive_examples: int = 20
    min_negative_examples: int = 20
    max_brier: float = 0.25
    max_ece: float = 0.10
    ece_bin_count: int = 10

    def __post_init__(self) -> None:
        for name in ("min_examples", "min_positive_examples", "min_negative_examples", "ece_bin_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        object.__setattr__(self, "max_brier", _probability(self.max_brier, "max_brier"))
        object.__setattr__(self, "max_ece", _probability(self.max_ece, "max_ece"))

    @property
    def policy_sha256(self) -> str:
        return _canonical_digest(
            {
                "schema": "rigorousrag-cross-profile-calibration-qualification-policy/v1",
                **asdict(self),
            }
        )


@dataclass(frozen=True)
class CalibrationQualificationReceipt:
    profile_id: str
    profile_sha256: str
    artifact_sha256: str
    calibration_contract_sha256: str
    evaluation_examples_sha256: str
    policy_sha256: str
    example_count: int
    positive_count: int
    negative_count: int
    brier: float
    ece: float
    eligible: bool
    reason_codes: tuple[str, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty.")
        for name in (
            "profile_sha256",
            "artifact_sha256",
            "calibration_contract_sha256",
            "evaluation_examples_sha256",
            "policy_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("example_count", "positive_count", "negative_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.positive_count + self.negative_count != self.example_count:
            raise ValueError("positive/negative counts must sum to example_count.")
        object.__setattr__(self, "brier", _probability(self.brier, "brier"))
        object.__setattr__(self, "ece", _probability(self.ece, "ece"))
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be boolean.")
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.eligible and reasons:
            raise ValueError("eligible qualification receipts may not contain failure reasons.")
        if not self.eligible and not reasons:
            raise ValueError("ineligible qualification receipts require reason codes.")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _canonical_digest(self._payload())
        provided = _sha256(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match qualification receipt content.")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-cross-profile-calibration-qualification/v1",
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "artifact_sha256": self.artifact_sha256,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "evaluation_examples_sha256": self.evaluation_examples_sha256,
            "policy_sha256": self.policy_sha256,
            "example_count": self.example_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "brier": self.brier,
            "ece": self.ece,
            "eligible": self.eligible,
            "reason_codes": self.reason_codes,
        }


def _evaluation_examples_sha256(examples: tuple[ScoreCalibrationExample, ...]) -> str:
    return _canonical_digest(
        {
            "schema": "rigorousrag-cross-profile-calibration-evaluation-examples/v1",
            "examples": sorted(
                (item.raw_score, int(item.relevant), item.weight) for item in examples
            ),
        }
    )


def qualify_calibrator(
    artifact: IsotonicCalibrationArtifact,
    examples: Iterable[ScoreCalibrationExample],
    *,
    policy: CalibrationQualificationPolicy = CalibrationQualificationPolicy(),
) -> CalibrationQualificationReceipt:
    if not isinstance(artifact, IsotonicCalibrationArtifact):
        raise ValueError("artifact must be IsotonicCalibrationArtifact.")
    values = tuple(examples)
    if not values or any(not isinstance(item, ScoreCalibrationExample) for item in values):
        raise ValueError("examples must be a non-empty collection of ScoreCalibrationExample values.")
    positive_count = sum(1 for item in values if item.relevant)
    negative_count = len(values) - positive_count
    diagnostics = evaluate_isotonic_calibrator(
        artifact,
        values,
        bin_count=policy.ece_bin_count,
    )
    reasons: list[str] = []
    if len(values) < policy.min_examples:
        reasons.append("insufficient_evaluation_examples")
    if positive_count < policy.min_positive_examples:
        reasons.append("insufficient_positive_examples")
    if negative_count < policy.min_negative_examples:
        reasons.append("insufficient_negative_examples")
    if diagnostics.brier > policy.max_brier:
        reasons.append("brier_threshold_exceeded")
    if diagnostics.ece > policy.max_ece:
        reasons.append("ece_threshold_exceeded")
    payload = {
        "schema": "rigorousrag-cross-profile-calibration-qualification/v1",
        "profile_id": artifact.profile.profile_id,
        "profile_sha256": artifact.profile.profile_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "calibration_contract_sha256": artifact.calibration_contract_sha256,
        "evaluation_examples_sha256": _evaluation_examples_sha256(values),
        "policy_sha256": policy.policy_sha256,
        "example_count": len(values),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "brier": diagnostics.brier,
        "ece": diagnostics.ece,
        "eligible": not reasons,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return CalibrationQualificationReceipt(
        **payload,
        receipt_sha256=_canonical_digest(payload),
    )


def run_qualified_calibrated_fusion(
    ranked_lists: tuple[ProfileRankedList, ...],
    *,
    calibrators: Mapping[str, IsotonicCalibrationArtifact],
    qualifications: Mapping[str, CalibrationQualificationReceipt],
    policy: CrossProfileFusionPolicy,
) -> GovernedCrossProfileFusionRun:
    if policy.mode is not CrossProfileFusionMode.CALIBRATED_LOGIT:
        raise ValueError("qualified calibrated fusion requires CALIBRATED_LOGIT mode.")
    required_profiles = {item.profile.profile_id: item.profile for item in ranked_lists}
    if set(calibrators) != set(required_profiles) or set(qualifications) != set(required_profiles):
        raise ValueError("calibrators and qualifications must exactly cover participating profiles.")
    contracts: set[str] = set()
    policies: set[str] = set()
    for profile_id, profile in required_profiles.items():
        artifact = calibrators[profile_id]
        receipt = qualifications[profile_id]
        if artifact.profile.profile_sha256 != profile.profile_sha256:
            raise ValueError("calibrator profile identity does not match ranked-list profile.")
        if receipt.profile_sha256 != profile.profile_sha256:
            raise ValueError("qualification profile identity does not match ranked-list profile.")
        if receipt.artifact_sha256 != artifact.artifact_sha256:
            raise ValueError("qualification receipt does not cover the supplied calibrator artifact.")
        if receipt.calibration_contract_sha256 != artifact.calibration_contract_sha256:
            raise ValueError("qualification receipt calibration contract mismatch.")
        if not receipt.eligible:
            raise ValueError(f"profile {profile_id!r} is not qualified for calibrated fusion.")
        contracts.add(receipt.calibration_contract_sha256)
        policies.add(receipt.policy_sha256)
    if len(contracts) != 1:
        raise ValueError("qualified calibrators must share one calibration contract.")
    if len(policies) != 1:
        raise ValueError("qualified calibrators must share one qualification policy.")
    return run_governed_cross_profile_fusion(
        ranked_lists,
        calibrators=calibrators,
        policy=policy,
    )


__all__ = [
    "CalibrationQualificationPolicy",
    "CalibrationQualificationReceipt",
    "qualify_calibrator",
    "run_qualified_calibrated_fusion",
]
