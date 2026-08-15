"""Deterministic load-capacity summaries and multi-objective Pareto analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LoadSample:
    latency_seconds: float
    succeeded: bool
    rejected: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.latency_seconds, (int, float)) or isinstance(self.latency_seconds, bool):
            raise ValueError("latency_seconds must be numeric")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative and finite")
        if not isinstance(self.succeeded, bool) or not isinstance(self.rejected, bool):
            raise ValueError("succeeded and rejected must be booleans")
        if self.succeeded and self.rejected:
            raise ValueError("a successful sample cannot be rejected")


@dataclass(frozen=True)
class CapacitySummary:
    requests: int
    throughput_rps: float
    success_rate: float
    error_rate: float
    rejection_rate: float
    p50_latency_seconds: float
    p95_latency_seconds: float
    p99_latency_seconds: float


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return float(ordered[rank - 1])


def summarize_capacity(samples: Sequence[LoadSample], *, wall_seconds: float) -> CapacitySummary:
    if not isinstance(wall_seconds, (int, float)) or isinstance(wall_seconds, bool):
        raise ValueError("wall_seconds must be numeric")
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive and finite")
    count = len(samples)
    if count == 0:
        return CapacitySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    success = sum(1 for sample in samples if sample.succeeded)
    rejected = sum(1 for sample in samples if sample.rejected)
    latencies = [float(sample.latency_seconds) for sample in samples]
    return CapacitySummary(
        requests=count,
        throughput_rps=count / float(wall_seconds),
        success_rate=success / count,
        error_rate=(count - success) / count,
        rejection_rate=rejected / count,
        p50_latency_seconds=_percentile(latencies, 0.50),
        p95_latency_seconds=_percentile(latencies, 0.95),
        p99_latency_seconds=_percentile(latencies, 0.99),
    )


@dataclass(frozen=True)
class CapacityGate:
    minimum_throughput_rps: float = 0.0
    minimum_success_rate: float = 1.0
    maximum_p95_latency_seconds: float = float("inf")
    maximum_rejection_rate: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_throughput_rps,
            self.minimum_success_rate,
            self.maximum_p95_latency_seconds,
            self.maximum_rejection_rate,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or math.isnan(value) for value in values):
            raise ValueError("capacity gate values must be numeric")
        if self.minimum_throughput_rps < 0 or self.maximum_p95_latency_seconds < 0:
            raise ValueError("throughput and latency limits must be non-negative")
        if not 0 <= self.minimum_success_rate <= 1 or not 0 <= self.maximum_rejection_rate <= 1:
            raise ValueError("rate limits must be between zero and one")

    def evaluate(self, summary: CapacitySummary) -> tuple[bool, tuple[str, ...]]:
        reasons = []
        if summary.throughput_rps < self.minimum_throughput_rps:
            reasons.append("throughput")
        if summary.success_rate < self.minimum_success_rate:
            reasons.append("success_rate")
        if summary.p95_latency_seconds > self.maximum_p95_latency_seconds:
            reasons.append("p95_latency")
        if summary.rejection_rate > self.maximum_rejection_rate:
            reasons.append("rejection_rate")
        return not reasons, tuple(reasons)


@dataclass(frozen=True)
class ArchitecturePoint:
    name: str
    quality: float
    latency_seconds: float
    cost: float
    memory_bytes: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty")
        values = (self.quality, self.latency_seconds, self.cost, self.memory_bytes)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("architecture metrics must be finite numbers")
        if self.latency_seconds < 0 or self.cost < 0 or self.memory_bytes < 0:
            raise ValueError("resource metrics must be non-negative")


def _dominates(a: ArchitecturePoint, b: ArchitecturePoint) -> bool:
    no_worse = (
        a.quality >= b.quality
        and a.latency_seconds <= b.latency_seconds
        and a.cost <= b.cost
        and a.memory_bytes <= b.memory_bytes
    )
    strictly_better = (
        a.quality > b.quality
        or a.latency_seconds < b.latency_seconds
        or a.cost < b.cost
        or a.memory_bytes < b.memory_bytes
    )
    return no_worse and strictly_better


def pareto_frontier(points: Sequence[ArchitecturePoint]) -> tuple[ArchitecturePoint, ...]:
    names = [point.name for point in points]
    if len(names) != len(set(names)):
        raise ValueError("architecture names must be unique")
    frontier = [point for point in points if not any(_dominates(other, point) for other in points if other is not point)]
    return tuple(sorted(frontier, key=lambda point: (-point.quality, point.latency_seconds, point.cost, point.name)))
