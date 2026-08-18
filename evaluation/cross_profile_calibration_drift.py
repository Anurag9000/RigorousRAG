"""Drift/requalification guards for deployed cross-profile score calibrators."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from evaluation.cross_profile_calibration import CalibrationQualificationReceipt
from tools.cross_profile_fusion import IsotonicCalibrationArtifact, ScoreCalibrationExample, evaluate_isotonic_calibrator

_EPS = 1e-9


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _nonnegative(value: float, label: str) -> float:
    selected = _finite(value, label)
    if selected < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return selected


def _probability(value: float, label: str) -> float:
    selected = _finite(value, label)
    if not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


@dataclass(frozen=True)
class CalibrationDriftPolicy:
    max_qualification_age_seconds: float = 30.0 * 24.0 * 3600.0
    min_live_scores: int = 200
    min_labeled_examples: int = 50
    max_population_stability_index: float = 0.25
    max_jensen_shannon_divergence: float = 0.10
    max_brier: float = 0.25
    max_ece: float = 0.10
    fail_closed_on_insufficient_live_scores: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_qualification_age_seconds", _nonnegative(self.max_qualification_age_seconds, "max_qualification_age_seconds"))
        for name in ("min_live_scores", "min_labeled_examples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("max_population_stability_index", "max_jensen_shannon_divergence"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(self, "max_brier", _probability(self.max_brier, "max_brier"))
        object.__setattr__(self, "max_ece", _probability(self.max_ece, "max_ece"))
        if not isinstance(self.fail_closed_on_insufficient_live_scores, bool):
            raise ValueError("fail_closed_on_insufficient_live_scores must be boolean")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-calibration-drift-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class CalibrationDriftReference:
    profile_id: str
    profile_sha256: str
    artifact_sha256: str
    calibration_contract_sha256: str
    qualification_receipt_sha256: str
    qualified_at: float
    bin_count: int
    reference_proportions: tuple[float, ...]
    reference_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("profile_id must be non-empty")
        for name in ("profile_sha256", "artifact_sha256", "calibration_contract_sha256", "qualification_receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "qualified_at", _nonnegative(self.qualified_at, "qualified_at"))
        if isinstance(self.bin_count, bool) or not isinstance(self.bin_count, int) or not 2 <= self.bin_count <= 1000:
            raise ValueError("bin_count must be in [2, 1000]")
        values = tuple(_probability(value, "reference proportion") for value in self.reference_proportions)
        if len(values) != self.bin_count or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("reference proportions must match bin_count and sum to one")
        object.__setattr__(self, "reference_proportions", values)
        expected = _digest(self._payload())
        provided = _sha(self.reference_sha256, "reference_sha256")
        if expected != provided:
            raise ValueError("reference_sha256 does not match drift reference content")
        object.__setattr__(self, "reference_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-calibration-drift-reference/v1",
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "artifact_sha256": self.artifact_sha256,
            "calibration_contract_sha256": self.calibration_contract_sha256,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "qualified_at": self.qualified_at,
            "bin_count": self.bin_count,
            "reference_proportions": self.reference_proportions,
        }


def _histogram(probabilities: Sequence[float], bin_count: int) -> tuple[float, ...]:
    if not probabilities:
        raise ValueError("probability histogram requires at least one value")
    counts = [0 for _ in range(bin_count)]
    for value in probabilities:
        probability = _probability(value, "calibrated probability")
        index = min(int(probability * bin_count), bin_count - 1)
        counts[index] += 1
    total = len(probabilities)
    return tuple(count / total for count in counts)


def build_calibration_drift_reference(
    artifact: IsotonicCalibrationArtifact,
    qualification: CalibrationQualificationReceipt,
    reference_scores: Iterable[float],
    *,
    qualified_at: float,
    bin_count: int = 10,
) -> CalibrationDriftReference:
    if not isinstance(artifact, IsotonicCalibrationArtifact):
        raise ValueError("artifact must be IsotonicCalibrationArtifact")
    if not isinstance(qualification, CalibrationQualificationReceipt):
        raise ValueError("qualification must be CalibrationQualificationReceipt")
    if not qualification.eligible or qualification.artifact_sha256 != artifact.artifact_sha256 or qualification.profile_sha256 != artifact.profile.profile_sha256:
        raise ValueError("qualification does not authorize this calibration artifact")
    scores = tuple(reference_scores)
    if not scores:
        raise ValueError("reference_scores must be non-empty")
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or not 2 <= bin_count <= 1000:
        raise ValueError("bin_count must be in [2, 1000]")
    proportions = _histogram(tuple(artifact.predict(score) for score in scores), bin_count)
    payload = {
        "schema": "rigorousrag-calibration-drift-reference/v1",
        "profile_id": artifact.profile.profile_id,
        "profile_sha256": artifact.profile.profile_sha256,
        "artifact_sha256": artifact.artifact_sha256,
        "calibration_contract_sha256": artifact.calibration_contract_sha256,
        "qualification_receipt_sha256": qualification.receipt_sha256,
        "qualified_at": _nonnegative(qualified_at, "qualified_at"),
        "bin_count": bin_count,
        "reference_proportions": proportions,
    }
    return CalibrationDriftReference(**payload, reference_sha256=_digest(payload))


def _psi(reference: Sequence[float], current: Sequence[float]) -> float:
    total = 0.0
    for left, right in zip(reference, current):
        p = max(left, _EPS)
        q = max(right, _EPS)
        total += (q - p) * math.log(q / p)
    return total


def _js(reference: Sequence[float], current: Sequence[float]) -> float:
    midpoint = [(left + right) / 2.0 for left, right in zip(reference, current)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(value * math.log(value / max(target, _EPS)) for value, target in zip(left, right) if value > 0.0)

    return 0.5 * kl(reference, midpoint) + 0.5 * kl(current, midpoint)


@dataclass(frozen=True)
class CalibrationDriftDecision:
    profile_id: str
    profile_sha256: str
    artifact_sha256: str
    reference_sha256: str
    policy_sha256: str
    observed_at: float
    qualification_age_seconds: float
    live_score_count: int
    labeled_example_count: int
    population_stability_index: float | None
    jensen_shannon_divergence: float | None
    brier: float | None
    ece: float | None
    action: str
    reason_codes: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        if self.action not in {"calibrated_ok", "requalify_rrf_only"}:
            raise ValueError("action is invalid")
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.action == "calibrated_ok" and reasons:
            raise ValueError("calibrated_ok may not contain failure reasons")
        if self.action == "requalify_rrf_only" and not reasons:
            raise ValueError("requalify_rrf_only requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match calibration drift decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-calibration-drift-decision/v1",
            "profile_id": self.profile_id,
            "profile_sha256": self.profile_sha256,
            "artifact_sha256": self.artifact_sha256,
            "reference_sha256": self.reference_sha256,
            "policy_sha256": self.policy_sha256,
            "observed_at": self.observed_at,
            "qualification_age_seconds": self.qualification_age_seconds,
            "live_score_count": self.live_score_count,
            "labeled_example_count": self.labeled_example_count,
            "population_stability_index": self.population_stability_index,
            "jensen_shannon_divergence": self.jensen_shannon_divergence,
            "brier": self.brier,
            "ece": self.ece,
            "action": self.action,
            "reason_codes": self.reason_codes,
        }


def evaluate_calibration_drift(
    reference: CalibrationDriftReference,
    artifact: IsotonicCalibrationArtifact,
    live_scores: Iterable[float],
    *,
    observed_at: float,
    labeled_examples: Iterable[ScoreCalibrationExample] = (),
    policy: CalibrationDriftPolicy = CalibrationDriftPolicy(),
) -> CalibrationDriftDecision:
    if artifact.artifact_sha256 != reference.artifact_sha256 or artifact.profile.profile_sha256 != reference.profile_sha256 or artifact.calibration_contract_sha256 != reference.calibration_contract_sha256:
        raise ValueError("runtime calibrator differs from drift reference lineage")
    observed = _nonnegative(observed_at, "observed_at")
    if observed < reference.qualified_at:
        raise ValueError("observed_at cannot precede qualified_at")
    scores = tuple(live_scores)
    labels = tuple(labeled_examples)
    reasons: list[str] = []
    age = observed - reference.qualified_at
    if age > policy.max_qualification_age_seconds:
        reasons.append("qualification_expired")
    psi = js = None
    if len(scores) < policy.min_live_scores:
        if policy.fail_closed_on_insufficient_live_scores:
            reasons.append("insufficient_live_score_evidence")
    else:
        current = _histogram(tuple(artifact.predict(score) for score in scores), reference.bin_count)
        psi = _psi(reference.reference_proportions, current)
        js = _js(reference.reference_proportions, current)
        if psi > policy.max_population_stability_index:
            reasons.append("population_stability_index_exceeded")
        if js > policy.max_jensen_shannon_divergence:
            reasons.append("jensen_shannon_divergence_exceeded")
    brier = ece = None
    if labels:
        if len(labels) < policy.min_labeled_examples:
            reasons.append("insufficient_labeled_requalification_evidence")
        else:
            diagnostics = evaluate_isotonic_calibrator(artifact, labels)
            brier, ece = diagnostics.brier, diagnostics.ece
            if brier > policy.max_brier:
                reasons.append("brier_threshold_exceeded")
            if ece > policy.max_ece:
                reasons.append("ece_threshold_exceeded")
    action = "requalify_rrf_only" if reasons else "calibrated_ok"
    payload = {
        "schema": "rigorousrag-calibration-drift-decision/v1",
        "profile_id": reference.profile_id,
        "profile_sha256": reference.profile_sha256,
        "artifact_sha256": reference.artifact_sha256,
        "reference_sha256": reference.reference_sha256,
        "policy_sha256": policy.policy_sha256,
        "observed_at": observed,
        "qualification_age_seconds": age,
        "live_score_count": len(scores),
        "labeled_example_count": len(labels),
        "population_stability_index": psi,
        "jensen_shannon_divergence": js,
        "brier": brier,
        "ece": ece,
        "action": action,
        "reason_codes": tuple(sorted(set(reasons))),
    }
    return CalibrationDriftDecision(**payload, decision_sha256=_digest(payload))


__all__ = [
    "CalibrationDriftDecision",
    "CalibrationDriftPolicy",
    "CalibrationDriftReference",
    "build_calibration_drift_reference",
    "evaluate_calibration_drift",
]
