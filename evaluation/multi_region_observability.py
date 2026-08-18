"""Privacy-safe observability for multi-region authority and routing state."""

from __future__ import annotations

import math
from typing import Sequence

from evaluation.quality_observability import MetricObservation
from orchestration.multi_region_authority import RegionAuthorityRecord, RegionHealthObservation
from orchestration.multi_region_routing import RegionRoutePublicationReceipt


def _now(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("now must be finite and non-negative")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError("now must be finite and non-negative")
    return selected


def observations_from_region_health(
    observations: Sequence[RegionHealthObservation],
    *,
    now: float,
) -> tuple[MetricObservation, ...]:
    timestamp = _now(now)
    rows = tuple(observations)
    if any(not isinstance(row, RegionHealthObservation) for row in rows):
        raise ValueError("observations contains invalid values")
    output = []
    for row in sorted(rows, key=lambda item: item.region_id):
        health_age = max(0.0, timestamp - row.observed_at)
        rpo_age = max(0.0, timestamp - row.recovery_point_at)
        tags = (("metric_family", "multi_region"), ("region_id", row.region_id), ("variant", "health"))
        source = "evaluation.multi_region_observability"
        output.extend(
            (
                MetricObservation("multi_region.read_ready", float(row.ready_for_reads), "higher", "ratio", 1, source, tags),
                MetricObservation("multi_region.write_ready", float(row.ready_for_writes), "higher", "ratio", 1, source, tags),
                MetricObservation("multi_region.replication_lag_seconds", row.replication_lag_seconds, "lower", "seconds", 1, source, tags),
                MetricObservation("multi_region.health_age_seconds", health_age, "lower", "seconds", 1, source, tags),
                MetricObservation("multi_region.recovery_point_age_seconds", rpo_age, "lower", "seconds", 1, source, tags),
            )
        )
    return tuple(output)


def observations_from_region_authority(record: RegionAuthorityRecord) -> tuple[MetricObservation, ...]:
    if not isinstance(record, RegionAuthorityRecord):
        raise ValueError("record must be RegionAuthorityRecord")
    tags = (("metric_family", "multi_region"), ("region_id", record.authority_region), ("service_id", record.service_id), ("variant", "authority"))
    source = "evaluation.multi_region_observability"
    return (
        MetricObservation("multi_region.authority_revision", float(record.revision), "neutral", "revision", 1, source, tags),
        MetricObservation("multi_region.authority_fencing_token", float(record.fencing_token), "neutral", "token", 1, source, tags),
    )


def observations_from_region_route_publication(receipt: RegionRoutePublicationReceipt) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, RegionRoutePublicationReceipt):
        raise ValueError("receipt must be RegionRoutePublicationReceipt")
    tags = (("metric_family", "multi_region"), ("region_id", receipt.authority_region), ("service_id", receipt.service_id), ("variant", "route_publication"))
    source = "evaluation.multi_region_observability"
    return (
        MetricObservation("multi_region.route_publication_performed", float(receipt.publication_performed), "neutral", "ratio", 1, source, tags),
        MetricObservation("multi_region.provider_route_revision", float(receipt.provider_revision), "neutral", "revision", 1, source, tags),
    )


__all__ = [
    "observations_from_region_authority",
    "observations_from_region_health",
    "observations_from_region_route_publication",
]
