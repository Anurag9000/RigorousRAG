"""Dependency-light service telemetry, SLOs, and error-budget accounting."""

from __future__ import annotations

import json
import math
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol


def _finite(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or parsed < minimum:
        raise ValueError(f"{label} is invalid.")
    return parsed


@dataclass(frozen=True)
class StageObservation:
    trace_id: str
    stage: str
    duration_ms: float
    success: bool
    tokens: int = 0
    estimated_cost: float = 0.0
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SLOObjective:
    availability_target: float = 0.99
    latency_target_ms: float = 1_000.0
    latency_success_fraction: float = 0.95
    window_requests: int = 100

    def __post_init__(self) -> None:
        for name in ("availability_target", "latency_success_fraction"):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1].")
        _finite(self.latency_target_ms, "latency_target_ms", 0.000001)
        if isinstance(self.window_requests, bool) or not isinstance(self.window_requests, int):
            raise ValueError("window_requests must be an integer.")
        if self.window_requests < 1:
            raise ValueError("window_requests must be positive.")


@dataclass(frozen=True)
class SLOReport:
    request_count: int
    availability: float
    latency_success_fraction: float
    p95_latency_ms: float
    error_budget_total: float
    error_budget_consumed: float
    error_budget_remaining: float
    within_slo: bool


class SpanSink(Protocol):
    """Bridge for OpenTelemetry/OTLP or any other span exporter."""

    def emit_span(self, observation: StageObservation) -> None: ...


class CallbackSpanSink:
    """Concrete adapter for SDK exporters without taking a hard dependency.

    An OpenTelemetry integration can pass a callback that starts/ends the SDK span and maps
    the canonical observation fields into attributes. This keeps core operation deterministic
    and import-safe when OpenTelemetry is not installed.
    """

    def __init__(self, callback: Callable[[StageObservation], None]) -> None:
        self._callback = callback

    def emit_span(self, observation: StageObservation) -> None:
        self._callback(observation)


class TelemetryRecorder:
    def __init__(self, *, sink: SpanSink | None = None, max_observations: int = 10_000) -> None:
        if isinstance(max_observations, bool) or not isinstance(max_observations, int):
            raise ValueError("max_observations must be an integer.")
        if max_observations < 1:
            raise ValueError("max_observations must be positive.")
        self._sink = sink
        self._max = max_observations
        self._observations: list[StageObservation] = []

    @property
    def observations(self) -> tuple[StageObservation, ...]:
        return tuple(self._observations)

    def record(self, observation: StageObservation) -> None:
        _finite(observation.duration_ms, "duration_ms")
        if observation.tokens < 0:
            raise ValueError("tokens must be non-negative.")
        _finite(observation.estimated_cost, "estimated_cost")
        self._observations.append(observation)
        if len(self._observations) > self._max:
            del self._observations[: len(self._observations) - self._max]
        if self._sink is not None:
            self._sink.emit_span(observation)

    @contextmanager
    def stage(
        self,
        stage: str,
        *,
        trace_id: str | None = None,
        attributes: Mapping[str, str] | None = None,
    ) -> Iterator[dict[str, object]]:
        selected_trace = trace_id or uuid.uuid4().hex
        started = time.perf_counter()
        state: dict[str, object] = {"tokens": 0, "estimated_cost": 0.0}
        success = False
        try:
            yield state
            success = True
        finally:
            elapsed = (time.perf_counter() - started) * 1_000.0
            self.record(
                StageObservation(
                    trace_id=selected_trace,
                    stage=str(stage),
                    duration_ms=elapsed,
                    success=success,
                    tokens=int(state.get("tokens", 0)),
                    estimated_cost=float(state.get("estimated_cost", 0.0)),
                    attributes=dict(attributes or {}),
                )
            )

    def export_jsonl(self, path: str | Path) -> int:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for observation in self._observations:
                handle.write(json.dumps(asdict(observation), sort_keys=True, allow_nan=False) + "\n")
        return len(self._observations)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def evaluate_slo(
    observations: list[StageObservation] | tuple[StageObservation, ...],
    objective: SLOObjective | None = None,
) -> SLOReport:
    selected = objective or SLOObjective()
    window = list(observations)[-selected.window_requests :]
    if not window:
        return SLOReport(0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, True)
    availability = sum(1 for item in window if item.success) / len(window)
    latency_fraction = sum(
        1 for item in window if item.success and item.duration_ms <= selected.latency_target_ms
    ) / len(window)
    p95 = _percentile([item.duration_ms for item in window], 0.95)
    budget_total = len(window) * (1.0 - selected.availability_target)
    failures = sum(1 for item in window if not item.success)
    remaining = budget_total - failures
    within = availability >= selected.availability_target and latency_fraction >= selected.latency_success_fraction
    return SLOReport(
        request_count=len(window),
        availability=availability,
        latency_success_fraction=latency_fraction,
        p95_latency_ms=p95,
        error_budget_total=budget_total,
        error_budget_consumed=float(failures),
        error_budget_remaining=remaining,
        within_slo=within,
    )


__all__ = [
    "CallbackSpanSink",
    "SLOObjective",
    "SLOReport",
    "SpanSink",
    "StageObservation",
    "TelemetryRecorder",
    "evaluate_slo",
]
