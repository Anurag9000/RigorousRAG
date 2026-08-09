"""Deterministic production backpressure, circuit-breaker and SLO decisions."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_MAX_OBSERVATIONS = 1_000_000


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _unit(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _nonnegative(value: Any, label: str, maximum: float = 1.0e12) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= maximum:
        raise ValueError(f"{label} must be non-negative and bounded.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires observations.")
    selected = _unit(probability, "probability")
    ordered = sorted(values)
    position = selected * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class BackpressureSnapshot:
    inflight: int
    worker_capacity: int
    queue_depth: int
    queue_capacity: int
    error_rate: float
    p95_latency_ms: float
    latency_slo_ms: float
    circuit_open: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "inflight", _integer(self.inflight, "inflight", 0, 100_000_000))
        object.__setattr__(
            self,
            "worker_capacity",
            _integer(self.worker_capacity, "worker_capacity", 1, 100_000_000),
        )
        object.__setattr__(self, "queue_depth", _integer(self.queue_depth, "queue_depth", 0, 1_000_000_000))
        object.__setattr__(
            self,
            "queue_capacity",
            _integer(self.queue_capacity, "queue_capacity", 1, 1_000_000_000),
        )
        object.__setattr__(self, "error_rate", _unit(self.error_rate, "error_rate"))
        object.__setattr__(self, "p95_latency_ms", _nonnegative(self.p95_latency_ms, "p95_latency_ms"))
        latency_slo = _nonnegative(self.latency_slo_ms, "latency_slo_ms")
        if latency_slo <= 0.0:
            raise ValueError("latency_slo_ms must be positive.")
        object.__setattr__(self, "latency_slo_ms", latency_slo)
        if not isinstance(self.circuit_open, bool):
            raise ValueError("circuit_open must be boolean.")


@dataclass(frozen=True)
class AdmissionDecision:
    action: str
    pressure: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.action not in {"admit", "defer", "shed"}:
            raise ValueError("admission action is unsupported.")
        object.__setattr__(self, "pressure", _unit(self.pressure, "pressure"))
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason for reason in self.reasons
        ):
            raise ValueError("admission reasons are invalid.")


def backpressure_decision(snapshot: BackpressureSnapshot) -> AdmissionDecision:
    if not isinstance(snapshot, BackpressureSnapshot):
        raise ValueError("snapshot must be BackpressureSnapshot.")
    worker_pressure = min(snapshot.inflight / snapshot.worker_capacity, 1.0)
    queue_pressure = min(snapshot.queue_depth / snapshot.queue_capacity, 1.0)
    latency_pressure = min(snapshot.p95_latency_ms / snapshot.latency_slo_ms, 2.0) / 2.0
    pressure = max(
        worker_pressure,
        queue_pressure,
        snapshot.error_rate,
        latency_pressure,
    )
    reasons: list[str] = []
    if snapshot.circuit_open:
        reasons.append("circuit_open")
    if snapshot.error_rate >= 0.50:
        reasons.append("error_rate_critical")
    if snapshot.queue_depth >= snapshot.queue_capacity:
        reasons.append("queue_full")
    if snapshot.p95_latency_ms >= snapshot.latency_slo_ms * 2.0:
        reasons.append("latency_critical")
    if reasons:
        return AdmissionDecision("shed", min(pressure, 1.0), tuple(reasons))
    if snapshot.inflight >= snapshot.worker_capacity:
        reasons.append("workers_saturated")
    if queue_pressure >= 0.80:
        reasons.append("queue_high")
    if snapshot.error_rate >= 0.20:
        reasons.append("error_rate_high")
    if snapshot.p95_latency_ms > snapshot.latency_slo_ms:
        reasons.append("latency_above_slo")
    if reasons or pressure >= 0.80:
        return AdmissionDecision("defer", min(pressure, 1.0), tuple(reasons or ["pressure_high"]))
    return AdmissionDecision("admit", min(pressure, 1.0), ())


@dataclass(frozen=True)
class CircuitBreakerState:
    state: str = "closed"
    consecutive_failures: int = 0
    opened_at: float | None = None
    half_open_successes: int = 0

    def __post_init__(self) -> None:
        if self.state not in {"closed", "open", "half_open"}:
            raise ValueError("circuit breaker state is invalid.")
        object.__setattr__(
            self,
            "consecutive_failures",
            _integer(self.consecutive_failures, "consecutive_failures", 0, 1_000_000),
        )
        object.__setattr__(
            self,
            "half_open_successes",
            _integer(self.half_open_successes, "half_open_successes", 0, 1_000_000),
        )
        if self.opened_at is not None:
            object.__setattr__(self, "opened_at", _nonnegative(self.opened_at, "opened_at"))
        if self.state == "open" and self.opened_at is None:
            raise ValueError("open circuit requires opened_at.")


def circuit_breaker_transition(
    state: CircuitBreakerState,
    *,
    success: bool | None,
    now: float,
    failure_threshold: int = 5,
    cooldown_seconds: float = 30.0,
    half_open_successes: int = 2,
) -> CircuitBreakerState:
    """Advance breaker state; success=None is a timer/probe eligibility tick."""

    if not isinstance(state, CircuitBreakerState):
        raise ValueError("state must be CircuitBreakerState.")
    if success is not None and not isinstance(success, bool):
        raise ValueError("success must be boolean or null.")
    current = _nonnegative(now, "now")
    threshold = _integer(failure_threshold, "failure_threshold", 1, 1_000_000)
    required_successes = _integer(half_open_successes, "half_open_successes", 1, 1_000_000)
    cooldown = _nonnegative(cooldown_seconds, "cooldown_seconds")

    if state.state == "open":
        if current < (state.opened_at or 0.0) + cooldown:
            return state
        if success is None:
            return CircuitBreakerState("half_open", 0, state.opened_at, 0)
        state = CircuitBreakerState("half_open", 0, state.opened_at, 0)

    if success is None:
        return state
    if state.state == "closed":
        if success:
            return CircuitBreakerState("closed", 0, None, 0)
        failures = state.consecutive_failures + 1
        if failures >= threshold:
            return CircuitBreakerState("open", failures, current, 0)
        return CircuitBreakerState("closed", failures, None, 0)

    if not success:
        return CircuitBreakerState("open", 1, current, 0)
    successes = state.half_open_successes + 1
    if successes >= required_successes:
        return CircuitBreakerState("closed", 0, None, 0)
    return CircuitBreakerState("half_open", 0, state.opened_at, successes)


@dataclass(frozen=True)
class SLOObservation:
    success: bool
    latency_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("success must be boolean.")
        object.__setattr__(self, "latency_ms", _nonnegative(self.latency_ms, "latency_ms"))


@dataclass(frozen=True)
class SLOReport:
    observations: int
    availability: float
    error_rate: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    latency_slo_fraction: float
    error_budget_burn_rate: float


def compute_slo_report(
    observations: Iterable[SLOObservation],
    *,
    latency_slo_ms: float,
    availability_slo: float = 0.99,
) -> SLOReport:
    if isinstance(observations, (str, bytes, bytearray)):
        raise ValueError("observations must be an iterable of SLOObservation values.")
    values: list[SLOObservation] = []
    for item in observations:
        if len(values) >= _MAX_OBSERVATIONS:
            raise ValueError("SLO observations exceed the limit.")
        if not isinstance(item, SLOObservation):
            raise ValueError("every observation must be SLOObservation.")
        values.append(item)
    if not values:
        raise ValueError("at least one SLO observation is required.")
    latency_slo = _nonnegative(latency_slo_ms, "latency_slo_ms")
    if latency_slo <= 0.0:
        raise ValueError("latency_slo_ms must be positive.")
    target_availability = _unit(availability_slo, "availability_slo")
    if target_availability >= 1.0:
        raise ValueError("availability_slo must be below 1.")
    availability = sum(item.success for item in values) / len(values)
    error_rate = 1.0 - availability
    latencies = [item.latency_ms for item in values]
    allowed_error_rate = 1.0 - target_availability
    burn = error_rate / allowed_error_rate if allowed_error_rate > 0.0 else math.inf
    return SLOReport(
        observations=len(values),
        availability=availability,
        error_rate=error_rate,
        p50_latency_ms=_quantile(latencies, 0.50),
        p95_latency_ms=_quantile(latencies, 0.95),
        p99_latency_ms=_quantile(latencies, 0.99),
        latency_slo_fraction=sum(item.latency_ms <= latency_slo for item in values) / len(values),
        error_budget_burn_rate=burn,
    )


__all__ = [
    "AdmissionDecision",
    "BackpressureSnapshot",
    "CircuitBreakerState",
    "SLOObservation",
    "SLOReport",
    "backpressure_decision",
    "circuit_breaker_transition",
    "compute_slo_report",
]
