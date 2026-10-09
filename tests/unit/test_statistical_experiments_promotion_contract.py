"""Executable statistical evidence and fail-closed promotion contracts.

All measurements are synthetic controlled fixtures, not research benchmark
results or an assertion that any model is eligible for production promotion.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from evaluation import statistical_experiments as stats


def _comparison(
    metric: str = "answer_f1",
    *,
    improvement: float = 0.20,
    p_value: float = 0.001,
    ci_lower: float | None = 0.10,
    ci_upper: float | None = 0.30,
    direction: stats.MetricDirection = stats.MetricDirection.HIGHER_IS_BETTER,
) -> stats.MetricComparison:
    return stats.MetricComparison(
        metric=metric,
        direction=direction,
        baseline_mean=0.3,
        candidate_mean=0.5,
        raw_delta=0.2,
        improvement=improvement,
        paired_effect=1.0,
        p_value=p_value,
        ci_lower_improvement=ci_lower,
        ci_upper_improvement=ci_upper,
    )


def _rule(
    metric: str = "answer_f1", *,
    required: bool = True,
    minimum_improvement: float = 0.05,
    require_ci_above_threshold: bool = True,
    direction: stats.MetricDirection = stats.MetricDirection.HIGHER_IS_BETTER,
) -> stats.MetricPromotionRule:
    return stats.MetricPromotionRule(
        metric=metric,
        direction=direction,
        minimum_improvement=minimum_improvement,
        require_ci_above_threshold=require_ci_above_threshold,
        required=required,
    )


@pytest.mark.parametrize("rules", (
    (),
    (_rule(required=False),),
    (_rule(), _rule()),
    (_rule(), "unchecked"),
))
def test_promotion_fails_closed_without_valid_unique_mandatory_rules(rules):
    with pytest.raises(ValueError, match="mandatory|unique|MetricPromotionRule"):
        stats.decide_promotion((_comparison(),), rules)


def test_promotion_with_one_mandatory_metric_and_valid_ci_is_governed_success():
    result = stats.decide_promotion((_comparison(),), (_rule(),))
    assert result.promoted is True
    assert result.correction == "holm"
    assert result.reasons == ("all required statistical promotion gates passed",)
    assert result.comparisons[0].adjusted_p_value == pytest.approx(0.001)


@pytest.mark.parametrize("comparison,rule,reason", (
    (_comparison(improvement=0.01), _rule(), "improvement"),
    (_comparison(p_value=0.3), _rule(), "adjusted p"),
    (_comparison(ci_lower=0.02), _rule(), "confidence interval"),
    (_comparison(ci_lower=None, ci_upper=None), _rule(), "confidence interval"),
))
def test_mandatory_promotion_gate_failure_is_reported(comparison, rule, reason):
    result = stats.decide_promotion((comparison,), (rule,), correction="none")
    assert not result.promoted
    assert any(reason in item for item in result.reasons)


def test_optional_metric_failure_does_not_block_mandatory_promotion():
    result = stats.decide_promotion(
        (_comparison(), _comparison("latency", improvement=-0.1)),
        (_rule(), _rule("latency", required=False)),
    )
    assert result.promoted is True
    assert any("latency:" in item for item in result.reasons)


def test_missing_required_metric_blocks_while_missing_optional_does_not():
    passing = stats.decide_promotion((_comparison(),), (_rule(), _rule("missing", required=False)))
    failing = stats.decide_promotion((_comparison(),), (_rule(), _rule("missing")))
    assert passing.promoted and not failing.promoted
    assert any("missing required metric" in item for item in failing.reasons)


def test_mismatched_direction_is_rejected():
    with pytest.raises(ValueError, match="metric direction mismatch"):
        stats.decide_promotion(
            (_comparison(),),
            (_rule(direction=stats.MetricDirection.LOWER_IS_BETTER),),
        )


@pytest.mark.parametrize("method", ("holm", "bh", "benjamini_hochberg", "none"))
def test_correction_names_preserve_metric_order_and_valid_probabilities(method):
    comparisons = (_comparison("a", p_value=0.01),
                   _comparison("b", p_value=0.03),
                   _comparison("c", p_value=0.04))
    values = stats.apply_multiplicity(comparisons, method=method)
    assert tuple(v.metric for v in values) == ("a", "b", "c")
    assert all(0.0 <= v.adjusted_p_value <= 1.0 for v in values)
    if method == "holm":
        assert tuple(v.adjusted_p_value for v in values) == pytest.approx((0.03, 0.06, 0.06))
    if method in {"bh", "benjamini_hochberg"}:
        assert tuple(v.adjusted_p_value for v in values) == pytest.approx((0.03, 0.04, 0.04))
    if method == "none":
        assert tuple(v.adjusted_p_value for v in values) == pytest.approx((0.01, 0.03, 0.04))


def test_duplicate_comparisons_invalid_method_and_missing_input_rejected():
    with pytest.raises(ValueError, match="unique"):
        stats.apply_multiplicity((_comparison(), _comparison()))
    with pytest.raises(ValueError, match="method"):
        stats.apply_multiplicity((_comparison(),), method="unapproved")
    with pytest.raises(ValueError, match="non-empty"):
        stats.apply_multiplicity(())


@pytest.mark.parametrize("adjuster", (stats.holm_adjust, stats.benjamini_hochberg_adjust))
def test_multiplicity_adjustments_are_bounded_monotone_and_no_smaller_than_raw(adjuster):
    source = {"b": 0.4, "a": 0.01, "c": 0.3, "d": 0.005}
    adjusted = adjuster(source)
    assert tuple(adjusted) == tuple(source)
    assert all(source[k] <= adjusted[k] <= 1.0 for k in source)
    ordered = sorted(source, key=source.get)
    assert [adjusted[k] for k in ordered] == sorted(adjusted.values())


def test_paired_bootstrap_is_deterministic_and_preserves_aligned_effect():
    baseline = (0.1, 0.2, 0.3, 0.4, 0.5)
    candidate = tuple(value + 0.2 for value in baseline)
    first = stats.paired_bootstrap_difference(
        baseline, candidate, seed=44, resamples=101, confidence=0.9
    )
    second = stats.paired_bootstrap_difference(
        baseline, candidate, seed=44, resamples=101, confidence=0.9
    )
    assert first == second
    assert first.mean_difference == pytest.approx(0.2)
    assert first.ci_lower == pytest.approx(0.2)
    assert first.ci_upper == pytest.approx(0.2)
    assert first.excludes_zero


def test_paired_permutation_exercises_all_three_alternatives_and_finite_p():
    baseline = (1,) * 12
    candidate = (2,) * 12
    outputs = [
        stats.paired_permutation_test(
            baseline, candidate, resamples=1000, seed=7, alternative=alternative,
        )
        for alternative in stats.Alternative
    ]
    assert all(0 < result.p_value <= 1 for result in outputs)
    assert all(result.observed_difference == pytest.approx(1.0) for result in outputs)
    assert outputs[0].p_value < 0.05
    assert outputs[1].p_value < 0.05
    assert outputs[2].p_value > 0.95


def test_standardized_effect_guards_zero_variance_and_single_sample():
    assert stats.paired_standardized_effect((1,), (2,)) is None
    assert stats.paired_standardized_effect((1, 2, 3), (2, 3, 4)) is None
    value = stats.paired_standardized_effect((1, 2, 3, 4), (1, 3, 4, 6))
    assert value is not None and math.isfinite(value)


@pytest.mark.parametrize("baseline,candidate", (
    ((), ()),
    ((1,), (1, 2)),
    ((1, float("nan")), (2, 3)),
    ((True, 1), (2, 3)),
))
def test_paired_input_alignment_and_finite_values_enforced(baseline, candidate):
    with pytest.raises(ValueError):
        stats.paired_bootstrap_difference(baseline, candidate, resamples=100)


@pytest.mark.parametrize("kwargs", (
    {"resamples": 99},
    {"resamples": True},
    {"confidence": 0},
    {"confidence": 1},
    {"seed": -1},
    {"seed": True},
))
def test_bootstrap_configuration_bounded(kwargs):
    with pytest.raises(ValueError):
        stats.paired_bootstrap_difference((1, 2), (2, 3), **kwargs)


def test_compare_metric_converts_lower_latency_to_positive_improvement():
    comparison = stats.compare_metric(
        "latency", (20, 24, 28, 31, 40), (18, 19, 26, 30, 34),
        direction=stats.MetricDirection.LOWER_IS_BETTER,
        bootstrap_resamples=120, permutation_resamples=150, seed=8,
    )
    assert comparison.raw_delta < 0
    assert comparison.improvement > 0
    assert comparison.ci_lower_improvement <= comparison.ci_upper_improvement
    assert 0 < comparison.p_value <= 1


def test_metric_validation_rejects_bad_bounds_and_probability():
    with pytest.raises(ValueError, match="both confidence"):
        _comparison(ci_lower=None)
    with pytest.raises(ValueError, match="inverted"):
        _comparison(ci_lower=0.4, ci_upper=0.2)
    with pytest.raises(ValueError, match="p_value"):
        _comparison(p_value=1.4)
    with pytest.raises(ValueError, match="alpha"):
        stats.MetricPromotionRule("a", stats.MetricDirection.HIGHER_IS_BETTER, alpha=1)
    with pytest.raises(ValueError, match="flags"):
        stats.MetricPromotionRule("a", stats.MetricDirection.HIGHER_IS_BETTER, required=1)
