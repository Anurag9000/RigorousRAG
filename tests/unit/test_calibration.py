from __future__ import annotations

import pytest

from evaluation.calibration import (
    CalibrationExample,
    ConfidenceSignals,
    CorrectnessPolicy,
    HistogramCalibrator,
    brier_score,
    expected_calibration_error,
    optimize_abstention_threshold,
    reliability_bins,
    risk_coverage_curve,
)


def test_reliability_bins_ece_and_brier_are_weighted() -> None:
    examples = [
        CalibrationExample(0.9, True, 2.0),
        CalibrationExample(0.8, False, 1.0),
        CalibrationExample(0.2, False, 1.0),
    ]
    bins = reliability_bins(examples, bin_count=2)
    assert bins[0].accuracy == 0.0
    assert bins[1].count == 2
    assert expected_calibration_error(examples, bin_count=2) == pytest.approx(0.2)
    assert brier_score(examples) == pytest.approx((2 * 0.01 + 0.64 + 0.04) / 4)


def test_risk_coverage_curve_includes_full_abstention_and_full_coverage() -> None:
    examples = [CalibrationExample(0.9, True), CalibrationExample(0.2, False)]
    curve = risk_coverage_curve(examples)
    assert curve[0].coverage == 0.0
    assert curve[0].risk == 0.0
    assert curve[-1].coverage == 1.0
    assert curve[-1].risk == 0.5


def test_asymmetric_cost_prefers_abstaining_from_low_confidence_error() -> None:
    examples = [
        CalibrationExample(0.95, True),
        CalibrationExample(0.90, True),
        CalibrationExample(0.20, False),
    ]
    decision = optimize_abstention_threshold(
        examples, incorrect_answer_cost=10.0, abstention_cost=0.1
    )
    assert decision.coverage == pytest.approx(2 / 3)
    assert decision.risk == 0.0
    assert decision.threshold == 0.9


def test_correctness_policy_makes_label_contract_explicit() -> None:
    strict = CorrectnessPolicy(min_answer_score=0.8, min_citation_support=0.7)
    assert strict.label(answer_score=0.9, citation_support=0.8)
    assert not strict.label(answer_score=0.9, citation_support=0.6)
    permissive = CorrectnessPolicy(
        min_answer_score=0.8, min_citation_support=0.7, require_both=False
    )
    assert permissive.label(answer_score=0.9, citation_support=0.1)


def test_confidence_signals_keep_raw_and_calibrated_values_distinct() -> None:
    signals = ConfidenceSignals(
        raw_retrieval=0.9,
        citation_coverage=0.6,
        self_consistency=0.7,
        calibrated=0.75,
    )
    assert signals.raw_retrieval == 0.9
    assert signals.calibrated == 0.75


def test_histogram_calibrator_uses_held_out_empirical_accuracy() -> None:
    calibrator = HistogramCalibrator(bin_count=2, smoothing=0).fit(
        [
            CalibrationExample(0.1, False),
            CalibrationExample(0.2, False),
            CalibrationExample(0.8, True),
            CalibrationExample(0.9, True),
        ]
    )
    assert calibrator.predict(0.25) == 0.0
    assert calibrator.predict(0.75) == 1.0


def test_calibration_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError):
        CalibrationExample(1.1, True)
    with pytest.raises(ValueError):
        ConfidenceSignals(-0.1, 0.5, 0.5)
