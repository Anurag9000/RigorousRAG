from __future__ import annotations

import hashlib

import pytest

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.cross_profile_calibration_drift import CalibrationDriftPolicy, build_calibration_drift_reference, evaluate_calibration_drift
from tools.cross_profile_fusion import CalibrationContract, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile() -> RetrieverScoreProfile:
    return RetrieverScoreProfile("dense", "dense", sha("score"), sha("model"))


def contract(tag: str = "main") -> CalibrationContract:
    return CalibrationContract(sha(f"dataset:{tag}"), sha(f"split:{tag}"), sha(f"rel:{tag}"), sha(f"universe:{tag}"), "science")


def artifact(c=None):
    return fit_isotonic_calibrator(
        profile=profile(),
        contract=c or contract(),
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(0.2, False), ScoreCalibrationExample(0.8, True), ScoreCalibrationExample(1.0, True)),
    )


def qualification(value):
    return qualify_calibrator(
        value,
        (ScoreCalibrationExample(0.1, False), ScoreCalibrationExample(0.15, False), ScoreCalibrationExample(0.85, True), ScoreCalibrationExample(0.9, True)),
        policy=CalibrationQualificationPolicy(min_examples=4, min_positive_examples=2, min_negative_examples=2, max_brier=1.0, max_ece=1.0, ece_bin_count=2),
    )


def policy(**overrides):
    values = dict(max_qualification_age_seconds=1000.0, min_live_scores=4, min_labeled_examples=2, max_population_stability_index=1.0, max_jensen_shannon_divergence=1.0, max_brier=1.0, max_ece=1.0, fail_closed_on_insufficient_live_scores=True)
    values.update(overrides)
    return CalibrationDriftPolicy(**values)


def test_stable_score_distribution_keeps_calibrated_fusion_eligible() -> None:
    value = artifact()
    reference = build_calibration_drift_reference(value, qualification(value), (0.05, 0.1, 0.9, 0.95), qualified_at=10.0, bin_count=4)
    decision = evaluate_calibration_drift(reference, value, (0.05, 0.1, 0.9, 0.95), observed_at=20.0, policy=policy())
    assert decision.action == "calibrated_ok"
    assert decision.reason_codes == ()
    assert decision.population_stability_index == pytest.approx(0.0)
    assert decision.jensen_shannon_divergence == pytest.approx(0.0)


def test_expired_qualification_forces_rrf_only_requalification() -> None:
    value = artifact()
    reference = build_calibration_drift_reference(value, qualification(value), (0.05, 0.1, 0.9, 0.95), qualified_at=10.0)
    decision = evaluate_calibration_drift(reference, value, (0.05, 0.1, 0.9, 0.95), observed_at=100.0, policy=policy(max_qualification_age_seconds=20.0))
    assert decision.action == "requalify_rrf_only"
    assert "qualification_expired" in decision.reason_codes


def test_distribution_shift_forces_rrf_only() -> None:
    value = artifact()
    reference = build_calibration_drift_reference(value, qualification(value), (0.05, 0.1, 0.9, 0.95), qualified_at=1.0, bin_count=4)
    decision = evaluate_calibration_drift(reference, value, (0.99, 0.99, 0.99, 0.99), observed_at=2.0, policy=policy(max_population_stability_index=0.01, max_jensen_shannon_divergence=0.01))
    assert decision.action == "requalify_rrf_only"
    assert {"population_stability_index_exceeded", "jensen_shannon_divergence_exceeded"} & set(decision.reason_codes)


def test_degraded_labeled_calibration_quality_forces_requalification() -> None:
    value = artifact()
    reference = build_calibration_drift_reference(value, qualification(value), (0.05, 0.1, 0.9, 0.95), qualified_at=1.0)
    wrong_labels = (ScoreCalibrationExample(0.05, True), ScoreCalibrationExample(0.95, False))
    decision = evaluate_calibration_drift(reference, value, (0.05, 0.1, 0.9, 0.95), observed_at=2.0, labeled_examples=wrong_labels, policy=policy(max_brier=0.1, max_ece=0.1))
    assert decision.action == "requalify_rrf_only"
    assert {"brier_threshold_exceeded", "ece_threshold_exceeded"} & set(decision.reason_codes)


def test_insufficient_live_scores_fail_closed_when_configured() -> None:
    value = artifact()
    reference = build_calibration_drift_reference(value, qualification(value), (0.05, 0.1, 0.9, 0.95), qualified_at=1.0)
    decision = evaluate_calibration_drift(reference, value, (0.5,), observed_at=2.0, policy=policy(min_live_scores=4))
    assert decision.action == "requalify_rrf_only"
    assert "insufficient_live_score_evidence" in decision.reason_codes


def test_runtime_calibrator_must_match_reference_lineage() -> None:
    value = artifact()
    reference = build_calibration_drift_reference(value, qualification(value), (0.05, 0.1, 0.9, 0.95), qualified_at=1.0)
    replacement = artifact(contract("replacement"))
    with pytest.raises(ValueError, match="differs from drift reference lineage"):
        evaluate_calibration_drift(reference, replacement, (0.05, 0.1, 0.9, 0.95), observed_at=2.0, policy=policy())
