"""Privacy-safe aggregate observations for generation-scoped cache outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from evaluation.quality_observability import MetricObservation
from orchestration.generation_scoped_cache import CacheLookup


@dataclass(frozen=True)
class CacheLookupSummary:
    count: int
    hit_count: int
    miss_count: int
    expired_count: int
    revoked_count: int

    def __post_init__(self) -> None:
        for name in ("count", "hit_count", "miss_count", "expired_count", "revoked_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hit_count + self.miss_count + self.expired_count + self.revoked_count != self.count:
            raise ValueError("cache status counts must sum to count")


def summarize_cache_lookups(lookups: Sequence[CacheLookup]) -> CacheLookupSummary:
    rows = tuple(lookups)
    if any(not isinstance(row, CacheLookup) for row in rows):
        raise ValueError("lookups contains invalid values")
    return CacheLookupSummary(
        count=len(rows),
        hit_count=sum(row.status == "hit" for row in rows),
        miss_count=sum(row.status == "miss" for row in rows),
        expired_count=sum(row.status == "expired" for row in rows),
        revoked_count=sum(row.status == "revoked" for row in rows),
    )


def observations_from_cache_summary(summary: CacheLookupSummary) -> tuple[MetricObservation, ...]:
    if not isinstance(summary, CacheLookupSummary):
        raise ValueError("summary must be CacheLookupSummary")
    count = summary.count
    denominator = count if count else 1
    tags = (("metric_family", "generation_cache"), ("variant", "lookup"))
    source = "evaluation.cache_observability"
    return (
        MetricObservation("cache.hit_rate", summary.hit_count / denominator if count else 0.0, "higher", "ratio", count, source, tags),
        MetricObservation("cache.miss_rate", summary.miss_count / denominator if count else 0.0, "neutral", "ratio", count, source, tags),
        MetricObservation("cache.expired_rate", summary.expired_count / denominator if count else 0.0, "lower", "ratio", count, source, tags),
        MetricObservation("cache.revoked_rate", summary.revoked_count / denominator if count else 0.0, "lower", "ratio", count, source, tags),
    )


__all__ = ["CacheLookupSummary", "observations_from_cache_summary", "summarize_cache_lookups"]
