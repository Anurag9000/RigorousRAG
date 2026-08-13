"""Deterministic drift metrics and alert evaluation for RAG operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

_MAX_VALUES = 1_000_000
_EPSILON = 1e-12


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _values(values: Iterable[Any], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric iterable.")
    result: list[float] = []
    for value in values:
        if len(result) >= _MAX_VALUES:
            raise ValueError(f"{label} exceeds the value limit.")
        result.append(_finite(value, label))
    if not result:
        raise ValueError(f"{label} must not be empty.")
    return tuple(result)


def histogram(values: Iterable[Any], edges: Sequence[Any]) -> tuple[float, ...]:
    rows = _values(values, "values")
    if isinstance(edges, (str, bytes, bytearray)) or len(edges) < 2 or len(edges) > 10_000:
        raise ValueError("edges must contain 2-10,000 finite boundaries.")
    boundaries = tuple(_finite(value, "edge") for value in edges)
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("edges must be strictly increasing.")
    counts = [0] * (len(boundaries) - 1)
    for value in rows:
        if value < boundaries[0] or value > boundaries[-1]:
            continue
        index = len(counts) - 1
        for candidate in range(len(counts)):
            if boundaries[candidate] <= value < boundaries[candidate + 1]:
                index = candidate
                break
        counts[index] += 1
    total = sum(counts)
    if total == 0:
        raise ValueError("no values fall inside the histogram edges.")
    return tuple(count / total for count in counts)


def population_stability_index(reference: Sequence[Any], current: Sequence[Any]) -> float:
    if len(reference) != len(current) or not reference:
        raise ValueError("reference/current histograms must have equal non-zero length.")
    total = 0.0
    for raw_ref, raw_cur in zip(reference, current):
        ref = max(_finite(raw_ref, "reference probability"), _EPSILON)
        cur = max(_finite(raw_cur, "current probability"), _EPSILON)
        total += (cur - ref) * math.log(cur / ref)
    return total


def jensen_shannon_divergence(reference: Mapping[str, Any], current: Mapping[str, Any]) -> float:
    if not isinstance(reference, Mapping) or not isinstance(current, Mapping):
        raise ValueError("reference and current must be mappings.")
    keys = sorted(set(reference) | set(current))
    if not keys:
        return 0.0
    ref_values = [max(0.0, _finite(reference.get(key, 0.0), "reference weight")) for key in keys]
    cur_values = [max(0.0, _finite(current.get(key, 0.0), "current weight")) for key in keys]
    ref_total, cur_total = sum(ref_values), sum(cur_values)
    if ref_total <= 0.0 or cur_total <= 0.0:
        raise ValueError("both distributions must contain positive mass.")
    p = [value / ref_total for value in ref_values]
    q = [value / cur_total for value in cur_values]
    m = [(left + right) / 2.0 for left, right in zip(p, q)]

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(value * math.log2(value / target) for value, target in zip(left, right) if value > 0.0 and target > 0.0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean_relative_shift(reference: Iterable[Any], current: Iterable[Any]) -> float:
    ref, cur = _values(reference, "reference"), _values(current, "current")
    ref_mean, cur_mean = fmean(ref), fmean(cur)
    denominator = max(abs(ref_mean), _EPSILON)
    return (cur_mean - ref_mean) / denominator


@dataclass(frozen=True)
class DriftThresholds:
    score_psi: float = 0.20
    route_jsd: float = 0.10
    calibration_shift: float = 0.05
    latency_relative: float = 0.25
    cost_relative: float = 0.25

    def __post_init__(self) -> None:
        for name in ("score_psi", "route_jsd", "calibration_shift", "latency_relative", "cost_relative"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DriftReport:
    score_psi: float
    route_jsd: float
    calibration_shift: float
    latency_relative: float
    cost_relative: float
    alerts: tuple[str, ...]


def evaluate_drift(
    *,
    reference_scores: Iterable[Any],
    current_scores: Iterable[Any],
    score_edges: Sequence[Any],
    reference_routes: Mapping[str, Any],
    current_routes: Mapping[str, Any],
    reference_calibration_error: Any,
    current_calibration_error: Any,
    reference_latency: Iterable[Any],
    current_latency: Iterable[Any],
    reference_cost: Iterable[Any],
    current_cost: Iterable[Any],
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    selected = thresholds or DriftThresholds()
    if not isinstance(selected, DriftThresholds):
        raise ValueError("thresholds must be DriftThresholds.")
    psi = population_stability_index(histogram(reference_scores, score_edges), histogram(current_scores, score_edges))
    jsd = jensen_shannon_divergence(reference_routes, current_routes)
    calibration = _finite(current_calibration_error, "current_calibration_error") - _finite(reference_calibration_error, "reference_calibration_error")
    latency = mean_relative_shift(reference_latency, current_latency)
    cost = mean_relative_shift(reference_cost, current_cost)
    alerts: list[str] = []
    if psi >= selected.score_psi:
        alerts.append("score_distribution_drift")
    if jsd >= selected.route_jsd:
        alerts.append("route_mix_drift")
    if calibration >= selected.calibration_shift:
        alerts.append("calibration_drift")
    if latency >= selected.latency_relative:
        alerts.append("latency_drift")
    if cost >= selected.cost_relative:
        alerts.append("cost_drift")
    return DriftReport(psi, jsd, calibration, latency, cost, tuple(alerts))


__all__ = [
    "DriftReport",
    "DriftThresholds",
    "evaluate_drift",
    "histogram",
    "jensen_shannon_divergence",
    "mean_relative_shift",
    "population_stability_index",
]
