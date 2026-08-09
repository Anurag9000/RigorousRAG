"""Distribution-drift signals for continual index adaptation and shadow rebuilds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

_MAX_BUCKETS = 10_000
_EPS = 1e-12


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _distribution(value: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value or len(value) > _MAX_BUCKETS:
        raise ValueError(f"{label} must be a non-empty bounded mapping.")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip() or len(key) > 200:
            raise ValueError(f"{label} contains an invalid bucket.")
        result[key.strip()] = _unit(raw, f"{label} mass")
    total = sum(result.values())
    if total <= 0.0:
        raise ValueError(f"{label} must contain positive mass.")
    return {key: mass / total for key, mass in result.items()}


def population_stability_index(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> float:
    """Compute PSI over named buckets with epsilon smoothing for missing buckets."""

    left = _distribution(baseline, "baseline")
    right = _distribution(current, "current")
    keys = set(left) | set(right)
    value = 0.0
    for key in keys:
        expected = max(left.get(key, 0.0), _EPS)
        actual = max(right.get(key, 0.0), _EPS)
        value += (actual - expected) * math.log(actual / expected)
    return max(0.0, value)


@dataclass(frozen=True)
class IndexDriftSnapshot:
    query_distribution_psi: float
    language_distribution_psi: float
    mean_retrieval_score_drop: float
    stale_document_fraction: float
    failed_update_fraction: float

    def __post_init__(self) -> None:
        for name in ("query_distribution_psi", "language_distribution_psi"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be finite and non-negative.")
            selected = float(value)
            if not math.isfinite(selected) or selected < 0.0 or selected > 1.0e6:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, selected)
        for name in (
            "mean_retrieval_score_drop",
            "stale_document_fraction",
            "failed_update_fraction",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))


@dataclass(frozen=True)
class IndexAdaptationDecision:
    action: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in {"stable", "shadow_rebuild", "urgent_rebuild"}:
            raise ValueError("index adaptation action is unsupported.")
        if self.action == "stable" and self.reasons:
            raise ValueError("stable decision may not contain reasons.")
        if self.action != "stable" and not self.reasons:
            raise ValueError("rebuild decisions require reasons.")


def decide_index_adaptation(
    snapshot: IndexDriftSnapshot,
    *,
    psi_shadow_threshold: float = 0.20,
    psi_urgent_threshold: float = 0.40,
    score_drop_threshold: float = 0.10,
    stale_fraction_threshold: float = 0.15,
    failure_fraction_threshold: float = 0.05,
) -> IndexAdaptationDecision:
    if not isinstance(snapshot, IndexDriftSnapshot):
        raise ValueError("snapshot must be IndexDriftSnapshot.")
    shadow_psi = float(psi_shadow_threshold)
    urgent_psi = float(psi_urgent_threshold)
    if not math.isfinite(shadow_psi) or not math.isfinite(urgent_psi) or not 0 <= shadow_psi <= urgent_psi:
        raise ValueError("PSI thresholds are invalid.")
    score = _unit(score_drop_threshold, "score_drop_threshold")
    stale = _unit(stale_fraction_threshold, "stale_fraction_threshold")
    failure = _unit(failure_fraction_threshold, "failure_fraction_threshold")
    reasons: list[str] = []
    maximum_psi = max(snapshot.query_distribution_psi, snapshot.language_distribution_psi)
    urgent = False
    if maximum_psi >= urgent_psi:
        reasons.append("distribution_shift_critical")
        urgent = True
    elif maximum_psi >= shadow_psi:
        reasons.append("distribution_shift_detected")
    if snapshot.mean_retrieval_score_drop >= score:
        reasons.append("retrieval_quality_dropped")
    if snapshot.stale_document_fraction >= stale:
        reasons.append("stale_document_fraction_high")
    if snapshot.failed_update_fraction >= failure:
        reasons.append("index_update_failures_high")
        urgent = True
    if not reasons:
        return IndexAdaptationDecision("stable", ())
    return IndexAdaptationDecision("urgent_rebuild" if urgent else "shadow_rebuild", tuple(reasons))


__all__ = [
    "IndexAdaptationDecision",
    "IndexDriftSnapshot",
    "decide_index_adaptation",
    "population_stability_index",
]
