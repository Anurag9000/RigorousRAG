import pytest

from tools.confidence_calibration import (
    CalibrationExample,
    fit_isotonic_calibrator,
    reliability_report,
    risk_coverage_curve,
    select_abstention_threshold,
)


def examples():
    return [
        CalibrationExample(0.9, True),
        CalibrationExample(0.8, True),
        CalibrationExample(0.7, False),
        CalibrationExample(0.4, True),
        CalibrationExample(0.2, False),
    ]


def test_reliability_report_has_brier_ece_and_bounded_bins():
    report = reliability_report(examples(), bin_count=5)
    assert report.example_count == 5
    assert 0.0 <= report.brier_score <= 1.0
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert 0.0 <= report.maximum_calibration_gap <= 1.0
    assert sum(bucket.count for bucket in report.bins) == 5
    assert all(bucket.lower < bucket.upper for bucket in report.bins)


def test_isotonic_calibrator_merges_nonmonotonic_accuracy_blocks():
    calibrator = fit_isotonic_calibrator(examples())
    probabilities = [block.calibrated_probability for block in calibrator.blocks]
    assert probabilities == sorted(probabilities)
    assert sum(block.count for block in calibrator.blocks) == 5
    calibrated = [calibrator.calibrate(value) for value in (0.1, 0.3, 0.5, 0.75, 0.95)]
    assert calibrated == sorted(calibrated)
    assert all(0.0 <= value <= 1.0 for value in calibrated)


def test_risk_coverage_groups_tied_thresholds():
    values = [
        CalibrationExample(0.9, True),
        CalibrationExample(0.9, False),
        CalibrationExample(0.5, True),
    ]
    curve = risk_coverage_curve(values)
    assert [point.threshold for point in curve] == [0.9, 0.5]
    assert curve[0].selected == 2
    assert curve[0].coverage == pytest.approx(2 / 3)
    assert curve[0].risk == 0.5
    assert curve[-1].coverage == 1.0


def test_threshold_selection_maximizes_coverage_under_risk_limit():
    selected = select_abstention_threshold(
        examples(),
        maximum_risk=0.34,
        minimum_coverage=0.4,
    )
    assert selected is not None
    assert selected.risk <= 0.34
    assert selected.coverage >= 0.4
    stricter = select_abstention_threshold(
        examples(),
        maximum_risk=0.0,
        minimum_coverage=0.8,
    )
    assert stricter is None


def test_empty_inputs_return_empty_reports_and_curves():
    report = reliability_report([])
    assert report.example_count == 0
    assert report.bins == ()
    assert fit_isotonic_calibrator([]).blocks == ()
    assert risk_coverage_curve([]) == ()
    assert select_abstention_threshold([], maximum_risk=0.1) is None


def test_invalid_probabilities_and_iterators_fail_closed():
    for value in (-0.1, 1.1, float("nan"), True, "0.5"):
        with pytest.raises(ValueError):
            CalibrationExample(value, True)

    class Hostile:
        def __iter__(self):
            yield CalibrationExample(0.5, True)
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="safely iterable"):
        reliability_report(Hostile())
    with pytest.raises(ValueError, match="bin_count"):
        reliability_report(examples(), bin_count=True)
    with pytest.raises(ValueError, match="maximum_risk"):
        select_abstention_threshold(examples(), maximum_risk=float("inf"))
