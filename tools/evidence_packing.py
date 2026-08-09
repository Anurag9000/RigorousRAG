"""Risk-aware evidence packing and deterministic quality/cost Pareto utilities."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Sequence

_MAX_CANDIDATES = 10_000
_MAX_TOKENS = 10_000_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid.")
    return result


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


def _nonnegative(value: Any, label: str, maximum: float = 1.0e12) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be non-negative and finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be non-negative and finite.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= maximum:
        raise ValueError(f"{label} must be non-negative and finite.")
    return result


@dataclass(frozen=True)
class EvidencePackingCandidate:
    evidence_id: str
    source_id: str
    token_cost: int
    relevance: float
    evidence_strength: float = 1.0
    retraction_risk: float = 0.0
    redundancy_group: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        object.__setattr__(self, "token_cost", _integer(self.token_cost, "token_cost", 1, _MAX_TOKENS))
        object.__setattr__(self, "relevance", _unit(self.relevance, "relevance"))
        object.__setattr__(self, "evidence_strength", _unit(self.evidence_strength, "evidence_strength"))
        object.__setattr__(self, "retraction_risk", _unit(self.retraction_risk, "retraction_risk"))
        if self.redundancy_group is not None:
            object.__setattr__(
                self,
                "redundancy_group",
                _identifier(self.redundancy_group, "redundancy_group"),
            )

    @property
    def base_utility(self) -> float:
        return self.relevance * self.evidence_strength * (1.0 - self.retraction_risk)


@dataclass(frozen=True)
class EvidencePackingPlan:
    selected: tuple[EvidencePackingCandidate, ...]
    token_budget: int
    tokens_used: int
    source_count: int
    objective: float
    excluded_high_risk: tuple[str, ...]

    def __post_init__(self) -> None:
        budget = _integer(self.token_budget, "token_budget", 1, _MAX_TOKENS)
        used = _integer(self.tokens_used, "tokens_used", 0, budget)
        object.__setattr__(self, "token_budget", budget)
        object.__setattr__(self, "tokens_used", used)
        if self.source_count != len({item.source_id for item in self.selected}):
            raise ValueError("source_count does not match selected evidence.")
        object.__setattr__(self, "objective", _nonnegative(self.objective, "objective"))


def pack_evidence(
    candidates: Sequence[EvidencePackingCandidate],
    *,
    token_budget: int,
    max_per_source: int = 3,
    max_retraction_risk: float = 0.50,
    source_diversity_bonus: float = 0.15,
    redundancy_penalty: float = 0.30,
) -> EvidencePackingPlan:
    """Greedily maximize marginal utility/token with source diversity and redundancy penalties."""

    if isinstance(candidates, (str, bytes, bytearray)) or len(candidates) > _MAX_CANDIDATES:
        raise ValueError("candidates must be a bounded sequence.")
    values = tuple(candidates)
    if any(not isinstance(item, EvidencePackingCandidate) for item in values):
        raise ValueError("every candidate must be EvidencePackingCandidate.")
    if len({item.evidence_id for item in values}) != len(values):
        raise ValueError("evidence IDs must be unique.")
    budget = _integer(token_budget, "token_budget", 1, _MAX_TOKENS)
    source_limit = _integer(max_per_source, "max_per_source", 1, _MAX_CANDIDATES)
    risk_limit = _unit(max_retraction_risk, "max_retraction_risk")
    diversity = _unit(source_diversity_bonus, "source_diversity_bonus")
    redundancy = _unit(redundancy_penalty, "redundancy_penalty")
    excluded = tuple(sorted(item.evidence_id for item in values if item.retraction_risk > risk_limit))
    remaining = [item for item in values if item.retraction_risk <= risk_limit]
    selected: list[EvidencePackingCandidate] = []
    tokens = 0
    source_counts: dict[str, int] = {}
    used_groups: set[str] = set()
    objective = 0.0

    while remaining:
        ranked: list[tuple[float, float, str, EvidencePackingCandidate]] = []
        for item in remaining:
            if tokens + item.token_cost > budget or source_counts.get(item.source_id, 0) >= source_limit:
                continue
            marginal = item.base_utility
            if item.source_id not in source_counts:
                marginal *= 1.0 + diversity
            if item.redundancy_group is not None and item.redundancy_group in used_groups:
                marginal *= 1.0 - redundancy
            density = marginal / item.token_cost
            ranked.append((density, marginal, item.evidence_id, item))
        if not ranked:
            break
        _, marginal, _, winner = max(ranked, key=lambda row: (row[0], row[1], row[2]))
        selected.append(winner)
        tokens += winner.token_cost
        source_counts[winner.source_id] = source_counts.get(winner.source_id, 0) + 1
        if winner.redundancy_group is not None:
            used_groups.add(winner.redundancy_group)
        objective += marginal
        remaining.remove(winner)

    return EvidencePackingPlan(
        selected=tuple(selected),
        token_budget=budget,
        tokens_used=tokens,
        source_count=len(source_counts),
        objective=objective,
        excluded_high_risk=excluded,
    )


@dataclass(frozen=True)
class QualityCostPoint:
    candidate_id: str
    quality: float
    cost: float
    latency_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "quality", _unit(self.quality, "quality"))
        object.__setattr__(self, "cost", _nonnegative(self.cost, "cost"))
        object.__setattr__(self, "latency_ms", _nonnegative(self.latency_ms, "latency_ms"))


def pareto_frontier(points: Sequence[QualityCostPoint]) -> tuple[QualityCostPoint, ...]:
    """Return non-dominated points where quality is maximized and cost/latency minimized."""

    if isinstance(points, (str, bytes, bytearray)) or len(points) > _MAX_CANDIDATES:
        raise ValueError("points must be a bounded sequence.")
    values = tuple(points)
    if any(not isinstance(item, QualityCostPoint) for item in values):
        raise ValueError("every point must be QualityCostPoint.")
    if len({item.candidate_id for item in values}) != len(values):
        raise ValueError("candidate IDs must be unique.")
    frontier = []
    for point in values:
        dominated = any(
            other.candidate_id != point.candidate_id
            and other.quality >= point.quality
            and other.cost <= point.cost
            and other.latency_ms <= point.latency_ms
            and (
                other.quality > point.quality
                or other.cost < point.cost
                or other.latency_ms < point.latency_ms
            )
            for other in values
        )
        if not dominated:
            frontier.append(point)
    return tuple(sorted(frontier, key=lambda item: (-item.quality, item.cost, item.latency_ms, item.candidate_id)))


def choose_pareto_candidate(
    points: Sequence[QualityCostPoint],
    *,
    max_cost: float,
    max_latency_ms: float,
    min_quality: float = 0.0,
) -> QualityCostPoint | None:
    """Choose highest-quality frontier point satisfying explicit deployment budgets."""

    cost_limit = _nonnegative(max_cost, "max_cost")
    latency_limit = _nonnegative(max_latency_ms, "max_latency_ms")
    quality_floor = _unit(min_quality, "min_quality")
    eligible = [
        item
        for item in pareto_frontier(points)
        if item.cost <= cost_limit
        and item.latency_ms <= latency_limit
        and item.quality >= quality_floor
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: (-item.quality, item.cost, item.latency_ms, item.candidate_id))


__all__ = [
    "EvidencePackingCandidate",
    "EvidencePackingPlan",
    "QualityCostPoint",
    "choose_pareto_candidate",
    "pack_evidence",
    "pareto_frontier",
]
