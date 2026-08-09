"""Deterministic freshness and stale-evidence scoring for time-sensitive retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

_MAX_AGE_SECONDS = 100 * 365.25 * 24 * 3600


def _finite(value: Any, label: str, minimum: float = 0.0, maximum: float = _MAX_AGE_SECONDS * 10) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside the supported range.")
    return result


def _unit(value: Any, label: str) -> float:
    result = _finite(value, label, 0.0, 1.0)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def freshness_score(
    *,
    observed_at: float,
    as_of: float,
    half_life_seconds: float,
    floor: float = 0.0,
) -> float:
    """Exponential half-life freshness score in [floor, 1]."""

    observed = _finite(observed_at, "observed_at", 0.0, 10**12)
    current = _finite(as_of, "as_of", 0.0, 10**12)
    half_life = _finite(half_life_seconds, "half_life_seconds", 1e-9, _MAX_AGE_SECONDS)
    selected_floor = _unit(floor, "floor")
    if observed > current:
        raise ValueError("observed_at may not be in the future relative to as_of.")
    decay = 0.5 ** ((current - observed) / half_life)
    return selected_floor + (1.0 - selected_floor) * decay


@dataclass(frozen=True)
class EvidenceFreshness:
    evidence_id: str
    observed_at: float
    generation_sequence: int
    current_generation_sequence: int
    base_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip() or len(self.evidence_id) > 500:
            raise ValueError("evidence_id is invalid.")
        object.__setattr__(self, "evidence_id", self.evidence_id.strip())
        object.__setattr__(self, "observed_at", _finite(self.observed_at, "observed_at", 0.0, 10**12))
        for name in ("generation_sequence", "current_generation_sequence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**63 - 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.generation_sequence > self.current_generation_sequence:
            raise ValueError("evidence generation may not exceed current generation.")
        object.__setattr__(self, "base_score", _unit(self.base_score, "base_score"))

    @property
    def stale_generation(self) -> bool:
        return self.generation_sequence < self.current_generation_sequence


def freshness_adjusted_score(
    evidence: EvidenceFreshness,
    *,
    as_of: float,
    half_life_seconds: float,
    temporal_importance: float = 1.0,
    stale_generation_penalty: float = 0.5,
) -> float:
    """Blend base relevance with time decay and explicit generation staleness."""

    if not isinstance(evidence, EvidenceFreshness):
        raise ValueError("evidence must be EvidenceFreshness.")
    importance = _unit(temporal_importance, "temporal_importance")
    stale_penalty = _unit(stale_generation_penalty, "stale_generation_penalty")
    fresh = freshness_score(
        observed_at=evidence.observed_at,
        as_of=as_of,
        half_life_seconds=half_life_seconds,
    )
    temporal_factor = (1.0 - importance) + importance * fresh
    generation_factor = stale_penalty if evidence.stale_generation else 1.0
    return max(0.0, min(evidence.base_score * temporal_factor * generation_factor, 1.0))


@dataclass(frozen=True)
class FreshnessSummary:
    count: int
    current_generation_fraction: float
    mean_freshness: float
    minimum_freshness: float
    stale_evidence_ids: tuple[str, ...]


def summarize_freshness(
    evidence: Sequence[EvidenceFreshness],
    *,
    as_of: float,
    half_life_seconds: float,
) -> FreshnessSummary:
    if isinstance(evidence, (str, bytes, bytearray)) or len(evidence) > 100_000:
        raise ValueError("evidence must be a bounded sequence.")
    values = tuple(evidence)
    if not values or any(not isinstance(item, EvidenceFreshness) for item in values):
        raise ValueError("evidence must contain EvidenceFreshness values.")
    scores = [
        freshness_score(
            observed_at=item.observed_at,
            as_of=as_of,
            half_life_seconds=half_life_seconds,
        )
        for item in values
    ]
    current_count = sum(not item.stale_generation for item in values)
    return FreshnessSummary(
        count=len(values),
        current_generation_fraction=current_count / len(values),
        mean_freshness=sum(scores) / len(scores),
        minimum_freshness=min(scores),
        stale_evidence_ids=tuple(sorted(item.evidence_id for item in values if item.stale_generation)),
    )


__all__ = [
    "EvidenceFreshness",
    "FreshnessSummary",
    "freshness_adjusted_score",
    "freshness_score",
    "summarize_freshness",
]
