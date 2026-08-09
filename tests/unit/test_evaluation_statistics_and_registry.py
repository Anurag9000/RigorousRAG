from __future__ import annotations

import pytest

from evaluation.dataset_registry import get_dataset_spec, list_dataset_specs
from evaluation.statistics import (
    brier_score,
    expected_calibration_error,
    paired_bootstrap_difference,
    paired_permutation_test,
    selective_risk_curve,
)


def test_registry_covers_requested_beir_multihop_and_domain_sets():
    required = {
        "scifact",
        "nfcorpus",
        "fiqa",
        "trec-covid",
        "arguana",
        "cqadupstack",
        "hotpotqa",
        "musique",
        "cuad",
        "finqa",
        "pubmedqa",
    }
    assert {spec.name for spec in list_dataset_specs()} >= required
    assert {spec.name for spec in list_dataset_specs(multihop=True)} >= {
        "hotpotqa",
        "musique",
        "finqa",
    }
    assert get_dataset_spec("SciFact").format == "beir"
    assert get_dataset_spec("cuad").domain == "legal"
    with pytest.raises(KeyError):
        get_dataset_spec("unknown")


def test_paired_bootstrap_is_deterministic_and_detects_positive_improvement():
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5]
    candidate = [0.2, 0.3, 0.4, 0.5, 0.6]
    first = paired_bootstrap_difference(
        baseline,
        candidate,
        resamples=1000,
        seed=7,
    )
    second = paired_bootstrap_difference(
        baseline,
        candidate,
        resamples=1000,
        seed=7,
    )
    assert first == second
    assert first.mean_difference == pytest.approx(0.1)
    assert first.confidence_low > 0.0
    assert first.probability_positive == 1.0


def test_paired_permutation_is_deterministic_and_two_sided():
    baseline = [0.1] * 12
    candidate = [0.9] * 12
    result = paired_permutation_test(
        baseline,
        candidate,
        permutations=2000,
        seed=3,
    )
    assert result.mean_difference == pytest.approx(0.8)
    assert 0.0 < result.p_value_two_sided < 0.05
    assert result == paired_permutation_test(
        baseline,
        candidate,
        permutations=2000,
        seed=3,
    )


def test_calibration_metrics_reward_well_calibrated_predictions():
    good_confidence = [0.9, 0.8, 0.2, 0.1]
    bad_confidence = [0.1, 0.2, 0.8, 0.9]
    outcomes = [1, 1, 0, 0]
    assert brier_score(good_confidence, outcomes) < brier_score(
        bad_confidence, outcomes
    )
    good_ece = expected_calibration_error(good_confidence, outcomes, bins=4)
    bad_ece = expected_calibration_error(bad_confidence, outcomes, bins=4)
    assert good_ece == pytest.approx(0.15)
    assert bad_ece == pytest.approx(0.85)
    assert good_ece < bad_ece


def test_selective_risk_falls_when_low_confidence_high_loss_cases_are_abstained():
    curve = selective_risk_curve(
        [0.95, 0.9, 0.8, 0.3, 0.1],
        [0.0, 0.0, 0.2, 1.0, 1.0],
        coverages=(1.0, 0.6, 0.4),
    )
    assert curve[0].risk > curve[-1].risk
    assert curve[0].achieved_coverage == 1.0
    assert curve[-1].achieved_coverage == 0.4
    assert curve[-1].threshold >= curve[0].threshold


def test_statistical_tools_reject_unpaired_nonfinite_and_invalid_controls():
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_difference([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        brier_score([float("nan")], [1.0])
    with pytest.raises(ValueError, match="at least 0.5"):
        paired_bootstrap_difference(
            [1.0],
            [2.0],
            resamples=100,
            confidence=0.4,
        )
    with pytest.raises(ValueError, match="greater than zero"):
        selective_risk_curve([0.5], [0.5], coverages=(0.0,))
