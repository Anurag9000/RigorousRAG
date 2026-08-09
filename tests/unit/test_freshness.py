from __future__ import annotations

import pytest

from tools.freshness import (
    EvidenceFreshness,
    freshness_adjusted_score,
    freshness_score,
    summarize_freshness,
)


def test_half_life_freshness_and_floor_are_deterministic():
    assert freshness_score(observed_at=0, as_of=10, half_life_seconds=10) == pytest.approx(0.5)
    assert freshness_score(
        observed_at=0,
        as_of=10,
        half_life_seconds=10,
        floor=0.2,
    ) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="future"):
        freshness_score(observed_at=11, as_of=10, half_life_seconds=10)


def test_freshness_adjustment_respects_temporal_importance_and_generation_staleness():
    current = EvidenceFreshness("current", 0, 2, 2, 0.8)
    stale = EvidenceFreshness("stale", 0, 1, 2, 0.8)
    current_score = freshness_adjusted_score(
        current,
        as_of=10,
        half_life_seconds=10,
        temporal_importance=1.0,
    )
    stale_score = freshness_adjusted_score(
        stale,
        as_of=10,
        half_life_seconds=10,
        temporal_importance=1.0,
        stale_generation_penalty=0.5,
    )
    timeless = freshness_adjusted_score(
        current,
        as_of=10,
        half_life_seconds=10,
        temporal_importance=0.0,
    )
    assert current_score == pytest.approx(0.4)
    assert stale_score == pytest.approx(0.2)
    assert timeless == pytest.approx(0.8)


def test_freshness_summary_exposes_current_fraction_and_stale_ids():
    values = [
        EvidenceFreshness("a", 90, 2, 2, 0.8),
        EvidenceFreshness("b", 80, 1, 2, 0.7),
    ]
    summary = summarize_freshness(values, as_of=100, half_life_seconds=10)
    assert summary.count == 2
    assert summary.current_generation_fraction == 0.5
    assert summary.stale_evidence_ids == ("b",)
    assert summary.minimum_freshness <= summary.mean_freshness <= 1.0


def test_evidence_freshness_rejects_future_generation():
    with pytest.raises(ValueError, match="may not exceed"):
        EvidenceFreshness("bad", 1, 3, 2, 0.5)
