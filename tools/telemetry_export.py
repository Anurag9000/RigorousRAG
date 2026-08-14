"""Privacy-safe, bounded-cardinality export bridges for service telemetry and SLOs."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Mapping

from tools.service_slo import SLOReport, SpanSink, StageObservation
from tools.slo_alerts import BurnRateReport


@dataclass(frozen=True)
class TelemetryExportPolicy:
    """Controls what request attributes may leave the process telemetry boundary."""

    allowed_attribute_keys: tuple[str, ...] = (
        "model",
        "provider",
        "retriever",
        "route",
        "tenant_class",
    )
    max_value_length: int = 96
    max_distinct_values_per_key: int = 64

    def __post_init__(self) -> None:
        if self.max_value_length < 1 or self.max_distinct_values_per_key < 1:
            raise ValueError("telemetry export bounds must be positive")
        if len(set(self.allowed_attribute_keys)) != len(self.allowed_attribute_keys):
            raise ValueError("allowed telemetry attribute keys must be unique")


def _safe_text(value: object, maximum: int) -> str:
    rendered = str(value)
    cleaned = "".join(ch if 32 <= ord(ch) != 127 else "_" for ch in rendered)
    return cleaned[:maximum]


class AttributeCardinalityGuard:
    """Allowlist and bound exported attribute values without retaining raw request text."""

    def __init__(self, policy: TelemetryExportPolicy | None = None) -> None:
        self.policy = policy or TelemetryExportPolicy()
        self._seen: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def sanitize(self, attributes: Mapping[str, str]) -> dict[str, str]:
        allowed = set(self.policy.allowed_attribute_keys)
        sanitized: dict[str, str] = {}
        with self._lock:
            for key in sorted(attributes):
                if key not in allowed:
                    continue
                value = _safe_text(attributes[key], self.policy.max_value_length)
                values = self._seen.setdefault(key, set())
                if value not in values and len(values) >= self.policy.max_distinct_values_per_key:
                    value = "_other"
                values.add(value)
                sanitized[key] = value
        return sanitized


def sanitize_observation(
    observation: StageObservation,
    guard: AttributeCardinalityGuard,
) -> StageObservation:
    return StageObservation(
        trace_id=_safe_text(observation.trace_id, 64),
        stage=_safe_text(observation.stage, 96),
        duration_ms=observation.duration_ms,
        success=observation.success,
        tokens=observation.tokens,
        estimated_cost=observation.estimated_cost,
        attributes=guard.sanitize(observation.attributes),
    )


class SanitizingSpanSink:
    """Wrap any span sink with the repository telemetry privacy boundary."""

    def __init__(self, sink: SpanSink, *, policy: TelemetryExportPolicy | None = None) -> None:
        self._sink = sink
        self._guard = AttributeCardinalityGuard(policy)

    def emit_span(self, observation: StageObservation) -> None:
        self._sink.emit_span(sanitize_observation(observation, self._guard))


class OpenTelemetrySpanSink:
    """Duck-typed OpenTelemetry tracer adapter with no SDK hard dependency.

    The supplied tracer is expected to provide ``start_span`` and returned spans must provide
    ``set_attribute`` and ``end``.  Export/collector configuration remains the responsibility
    of the production process, so core tests never require a live OTLP endpoint.
    """

    def __init__(self, tracer: Any, *, policy: TelemetryExportPolicy | None = None) -> None:
        self._tracer = tracer
        self._guard = AttributeCardinalityGuard(policy)

    def emit_span(self, observation: StageObservation) -> None:
        clean = sanitize_observation(observation, self._guard)
        span = self._tracer.start_span(clean.stage)
        try:
            span.set_attribute("rigorousrag.trace_id", clean.trace_id)
            span.set_attribute("rigorousrag.duration_ms", float(clean.duration_ms))
            span.set_attribute("rigorousrag.success", bool(clean.success))
            span.set_attribute("rigorousrag.tokens", int(clean.tokens))
            span.set_attribute("rigorousrag.estimated_cost", float(clean.estimated_cost))
            for key, value in clean.attributes.items():
                span.set_attribute(f"rigorousrag.{key}", value)
        finally:
            span.end()


def prometheus_slo_text(
    slo: SLOReport,
    burn: BurnRateReport,
    *,
    prefix: str = "rigorousrag",
) -> str:
    """Render bounded-cardinality SLO/error-budget metrics for Prometheus scraping."""

    if not prefix or not prefix.replace("_", "a").isalnum():
        raise ValueError("Prometheus prefix must contain only letters, digits, and underscores")
    values = (
        slo.availability,
        slo.latency_success_fraction,
        slo.p95_latency_ms,
        slo.error_budget_total,
        slo.error_budget_consumed,
        slo.error_budget_remaining,
        burn.short_burn_rate,
        burn.long_burn_rate,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("SLO metrics must be finite")
    lines = [
        f"{prefix}_slo_requests {slo.request_count}",
        f"{prefix}_slo_availability {slo.availability:.12f}",
        f"{prefix}_slo_latency_success_fraction {slo.latency_success_fraction:.12f}",
        f"{prefix}_slo_p95_latency_milliseconds {slo.p95_latency_ms:.6f}",
        f"{prefix}_error_budget_total {slo.error_budget_total:.12f}",
        f"{prefix}_error_budget_consumed {slo.error_budget_consumed:.12f}",
        f"{prefix}_error_budget_remaining {slo.error_budget_remaining:.12f}",
        f"{prefix}_slo_within_target {1 if slo.within_slo else 0}",
        f"{prefix}_burn_rate_short {burn.short_burn_rate:.12f}",
        f"{prefix}_burn_rate_long {burn.long_burn_rate:.12f}",
        f"{prefix}_burn_rate_alert {1 if burn.alert else 0}",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "AttributeCardinalityGuard",
    "OpenTelemetrySpanSink",
    "SanitizingSpanSink",
    "TelemetryExportPolicy",
    "prometheus_slo_text",
    "sanitize_observation",
]
