from __future__ import annotations

import pytest

from tools.index_drift import (
    IndexDriftSnapshot,
    decide_index_adaptation,
    population_stability_index,
)


def test_population_stability_index_is_zero_for_identical_distribution_and_positive_for_shift():
    baseline = {"simple": 0.7, "complex": 0.3}
    shifted = {"simple": 0.2, "complex": 0.8}
    assert population_stability_index(baseline, baseline) == pytest.approx(0.0)
    assert population_stability_index(baseline, shifted) > 0.0


def test_stable_shadow_and_urgent_index_adaptation_paths():
    stable = decide_index_adaptation(IndexDriftSnapshot(0.01, 0.02, 0.01, 0.01, 0.0))
    shadow = decide_index_adaptation(IndexDriftSnapshot(0.25, 0.02, 0.01, 0.01, 0.0))
    urgent = decide_index_adaptation(IndexDriftSnapshot(0.01, 0.02, 0.01, 0.01, 0.2))
    assert stable.action == "stable" and stable.reasons == ()
    assert shadow.action == "shadow_rebuild"
    assert "distribution_shift_detected" in shadow.reasons
    assert urgent.action == "urgent_rebuild"
    assert "index_update_failures_high" in urgent.reasons


def test_quality_and_staleness_can_trigger_shadow_rebuild_without_distribution_shift():
    decision = decide_index_adaptation(
        IndexDriftSnapshot(0.0, 0.0, 0.2, 0.3, 0.0)
    )
    assert decision.action == "shadow_rebuild"
    assert decision.reasons == (
        "retrieval_quality_dropped",
        "stale_document_fraction_high",
    )
