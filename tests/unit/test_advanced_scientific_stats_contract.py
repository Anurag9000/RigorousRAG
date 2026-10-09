"""Numerical and identifiability checks for governed meta-regression/network evidence.

All observations are manufactured arithmetic fixtures, not claims about
clinical, scientific, or hydrological research results.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from tools.advanced_scientific_stats import (
    NetworkContrast, meta_regress, network_meta_analyze,
)
from tools.scientific_synthesis import EffectEstimate


def _effect(i, value, *, outcome="flow", effect_type="mean_difference"):
    return EffectEstimate(
        study_id=f"study-{i}",
        outcome=outcome,
        effect_type=effect_type,
        estimate=value,
        standard_error=0.1,
        unit="m3/s" if effect_type == "mean_difference" else "",
    )


def _contrast(i, a, b, estimate, *, effect_type="mean_difference", outcome="flow"):
    return NetworkContrast(
        study_id=f"study-{i}",
        treatment_a=a, treatment_b=b,
        effect_type=effect_type,
        estimate=estimate, standard_error=0.1,
        outcome=outcome,
        unit="m3/s" if effect_type == "mean_difference" else "",
    )


def test_linear_meta_regression_recovers_intercept_and_slope_without_noise():
    rows = tuple(_effect(i, 1 + 2 * i) for i in range(7))
    moderators = {row.study_id: {"altitude": i} for i, row in enumerate(rows)}
    fit = meta_regress(rows, moderators, random_effects=False)
    assert fit.outcome == "flow"
    assert fit.studies == 7
    assert fit.moderators == ("altitude",)
    assert fit.tau_squared == 0
    assert fit.residual_df == 5
    assert fit.residual_q == pytest.approx(0, abs=1e-8)
    assert fit.coefficients[0].term == "intercept"
    assert fit.coefficients[0].estimate_analysis_scale == pytest.approx(1)
    assert fit.coefficients[1].term == "altitude"
    assert fit.coefficients[1].estimate_analysis_scale == pytest.approx(2)
    assert fit.fingerprint == meta_regress(rows, moderators, random_effects=False).fingerprint


def test_meta_regression_random_effects_and_two_moderator_fit():
    rows = tuple(_effect(i, 3 + 2 * i + 0.5 * (i % 2)) for i in range(10))
    moderators = {
        row.study_id: {"x": i, "group": (i % 2)}
        for i, row in enumerate(rows)
    }
    fit = meta_regress(rows, moderators, random_effects=True)
    estimates = {entry.term: entry.estimate_analysis_scale for entry in fit.coefficients}
    assert estimates["intercept"] == pytest.approx(3)
    assert estimates["x"] == pytest.approx(2)
    assert estimates["group"] == pytest.approx(0.5)
    assert fit.tau_squared >= 0
    assert fit.residual_df == 7


def test_singular_meta_regression_is_rejected():
    rows = tuple(_effect(i, i) for i in range(4))
    moderators = {row.study_id: {"constant": 1.0} for row in rows}
    with pytest.raises(ValueError, match="singular"):
        meta_regress(rows, moderators, random_effects=False)


@pytest.mark.parametrize("moderators,names,pattern", (
    ({}, None, "moderators"),
    ({"study-0": {"x": 1}}, ("x",), "more studies|missing"),
))
def test_missing_or_unidentified_meta_regression_fails_closed(moderators, names, pattern):
    rows = tuple(_effect(i, i) for i in range(5))
    with pytest.raises(ValueError, match=pattern):
        meta_regress(rows, moderators, moderator_names=names, random_effects=False)


def test_meta_regression_requires_identifiable_design_and_compatible_effects():
    small = tuple(_effect(i, i) for i in range(2))
    with pytest.raises(ValueError, match="more studies"):
        meta_regress(small, {item.study_id: {"x": i} for i, item in enumerate(small)})
    incompatible = (_effect(0, 1), _effect(1, 2, outcome="unrelated"), _effect(2, 3))
    with pytest.raises(ValueError, match="incompatible"):
        meta_regress(incompatible, {row.study_id: {"x": i} for i, row in enumerate(incompatible)})


def test_connected_network_recovers_transitive_consistency_without_residual():
    rows = (
        _contrast(0, "A", "B", 1.0),
        _contrast(1, "B", "C", 1.5),
        _contrast(2, "A", "C", 2.5),
        _contrast(3, "C", "A", -2.5),
    )
    result = network_meta_analyze(rows, reference_treatment="A")
    assert result.connected
    assert result.treatments == ("A", "B", "C")
    assert result.residual_q == pytest.approx(0, abs=1e-8)
    assert result.residual_df == 2
    estimates = {item.treatment: item.versus_reference_estimate
                 for item in result.treatment_effects}
    assert estimates == pytest.approx({"A": 0, "B": 1, "C": 2.5})
    assert result.pairwise_estimates["C vs B"] == pytest.approx(1.5)
    assert result.fingerprint == network_meta_analyze(rows, reference_treatment="A").fingerprint


def test_network_reference_change_preserves_pairwise_results():
    rows = (_contrast(0, "A", "B", 1), _contrast(1, "B", "C", 2))
    from_a = network_meta_analyze(rows, reference_treatment="A")
    from_c = network_meta_analyze(rows, reference_treatment="C")
    assert from_a.pairwise_estimates == pytest.approx(from_c.pairwise_estimates)
    assert from_a.reference_treatment == "A"
    assert from_c.reference_treatment == "C"


def test_log_ratio_network_predictions_are_multiplicative():
    rows = (
        _contrast(0, "A", "B", 2.0, effect_type="risk_ratio", outcome="risk"),
        _contrast(1, "B", "C", 3.0, effect_type="risk_ratio", outcome="risk"),
    )
    result = network_meta_analyze(rows, reference_treatment="A")
    assert result.pairwise_estimates["C vs A"] == pytest.approx(6)
    assert result.pairwise_estimates["C vs B"] == pytest.approx(3)
    estimates = {item.treatment: item.versus_reference_estimate for item in result.treatment_effects}
    assert estimates["A"] == pytest.approx(1)
    assert estimates["B"] == pytest.approx(2)


def test_correlation_network_fisher_transform_round_trips():
    contrast = _contrast(0, "A", "B", 0.5, effect_type="correlation", outcome="association")
    assert contrast.analysis_estimate == pytest.approx(math.atanh(0.5))
    assert contrast.from_analysis_scale(contrast.analysis_estimate) == pytest.approx(0.5)
    assert network_meta_analyze((contrast,)).pairwise_estimates["B vs A"] == pytest.approx(0.5)


@pytest.mark.parametrize("rows,reason", (
    ((), "non-empty"),
    ((_contrast(0, "A", "B", 1), _contrast(1, "C", "D", 2)), "disconnected"),
    ((_contrast(0, "A", "B", 1), _contrast(1, "B", "C", 2, outcome="other")), "share outcome"),
    ((_contrast(0, "A", "B", 1), _contrast(1, "B", "C", 2, effect_type="risk_ratio")), "share outcome"),
))
def test_invalid_or_disconnected_networks_fail_closed(rows, reason):
    with pytest.raises(ValueError, match=reason):
        network_meta_analyze(rows)


def test_reference_must_exist_in_network():
    with pytest.raises(ValueError, match="absent"):
        network_meta_analyze((_contrast(0, "A", "B", 1),), reference_treatment="Z")


@pytest.mark.parametrize("override", (
    {"treatment_b": "A"},
    {"standard_error": 0},
    {"estimate": float("nan")},
    {"effect_type": "unknown"},
))
def test_invalid_network_contrast_values_are_rejected(override):
    with pytest.raises(ValueError):
        replace(_contrast(0, "A", "B", 1), **override)


def test_network_contrast_rejects_invalid_ratio_and_correlation():
    with pytest.raises(ValueError, match="positive"):
        _contrast(0, "A", "B", 0, effect_type="risk_ratio", outcome="risk")
    with pytest.raises(ValueError, match="correlation"):
        _contrast(0, "A", "B", 1, effect_type="correlation", outcome="association")
