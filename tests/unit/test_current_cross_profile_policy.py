from __future__ import annotations

import hashlib

import pytest

from evaluation.cross_profile_calibration import CalibrationQualificationPolicy, qualify_calibrator
from evaluation.cross_profile_calibration_drift import CalibrationDriftDecision, CalibrationDriftPolicy, build_calibration_drift_reference, evaluate_calibration_drift
from tools.cross_profile_fusion import CalibrationContract, RetrieverScoreProfile, ScoreCalibrationExample, fit_isotonic_calibrator
from tools.current_cross_profile_policy import validate_current_calibrators


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def profile(name: str):
    return RetrieverScoreProfile(name, name, sha(f"score:{name}"), sha(f"model:{name}"))


def contract():
    return CalibrationContract(sha("dataset"), sha("split"), sha("relevance"), sha("universe"), "science")


def artifact(p):
    return fit_isotonic_calibrator(
        profile=p,
        contract=contract(),
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(0.2, False), ScoreCalibrationExample(0.8, True), ScoreCalibrationExample(1.0, True)),
    )


def current_decision(value):
    qualification = qualify_calibrator(
        value,
        (ScoreCalibrationExample(0.1, False), ScoreCalibrationExample(0.9, True)),
        policy=CalibrationQualificationPolicy(min_examples=2, min_positive_examples=1, min_negative_examples=1, max_brier=1.0, max_ece=1.0, ece_bin_count=2),
    )
    reference = build_calibration_drift_reference(value, qualification, (0.1, 0.1, 0.9, 0.9), qualified_at=1.0, bin_count=2)
    return evaluate_calibration_drift(
        reference,
        value,
        (0.1, 0.1, 0.9, 0.9),
        observed_at=2.0,
        policy=CalibrationDriftPolicy(max_qualification_age_seconds=100.0, min_live_scores=4, min_labeled_examples=1, max_population_stability_index=1.0, max_jensen_shannon_divergence=1.0, max_brier=1.0, max_ece=1.0),
    )


def test_current_calibrator_validation_accepts_exact_stable_lineage() -> None:
    dense, sparse = artifact(profile("dense")), artifact(profile("sparse"))
    decisions = {"dense": current_decision(dense), "sparse": current_decision(sparse)}
    rows = validate_current_calibrators({"dense": dense, "sparse": sparse}, decisions)
    assert tuple(profile_id for profile_id, _ in rows) == ("dense", "sparse")


def test_current_calibrator_validation_requires_exact_profile_coverage() -> None:
    dense, sparse = artifact(profile("dense")), artifact(profile("sparse"))
    with pytest.raises(ValueError, match="exactly cover"):
        validate_current_calibrators({"dense": dense, "sparse": sparse}, {"dense": current_decision(dense)})


def test_current_calibrator_validation_rejects_rrf_only_drift_decision() -> None:
    dense = artifact(profile("dense"))
    stable = current_decision(dense)
    stale_payload = {
        "profile_id": stable.profile_id,
        "profile_sha256": stable.profile_sha256,
        "artifact_sha256": stable.artifact_sha256,
        "reference_sha256": stable.reference_sha256,
        "policy_sha256": stable.policy_sha256,
        "observed_at": stable.observed_at,
        "qualification_age_seconds": stable.qualification_age_seconds,
        "live_score_count": stable.live_score_count,
        "labeled_example_count": stable.labeled_example_count,
        "population_stability_index": stable.population_stability_index,
        "jensen_shannon_divergence": stable.jensen_shannon_divergence,
        "brier": stable.brier,
        "ece": stable.ece,
        "action": "requalify_rrf_only",
        "reason_codes": ("qualification_expired",),
    }
    import json
    decision_digest = hashlib.sha256(json.dumps({"schema": "rigorousrag-calibration-drift-decision/v1", **stale_payload}, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    stale = CalibrationDriftDecision(**stale_payload, decision_sha256=decision_digest)
    with pytest.raises(ValueError, match="RRF-only fallback"):
        validate_current_calibrators({"dense": dense}, {"dense": stale})


def test_current_calibrator_validation_rejects_decision_for_other_artifact() -> None:
    dense = artifact(profile("dense"))
    replacement = fit_isotonic_calibrator(
        profile=profile("dense"),
        contract=contract(),
        examples=(ScoreCalibrationExample(0.0, False), ScoreCalibrationExample(0.5, True), ScoreCalibrationExample(1.0, True)),
    )
    with pytest.raises(ValueError, match="does not cover runtime calibrator"):
        validate_current_calibrators({"dense": replacement}, {"dense": current_decision(dense)})
