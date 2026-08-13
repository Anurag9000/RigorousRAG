"""Human-review routing driven by uncertainty, conflicts, proof gaps, and safety gates."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

_MAX_QUEUE = 100_000


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > 500:
        raise ValueError(f"{label} is invalid.")
    return text


@dataclass(frozen=True)
class ReviewPolicy:
    review_uncertainty: float = 0.35
    block_uncertainty: float = 0.75
    review_conflict: float = 0.30
    minimum_proof_completeness: float = 0.80
    minimum_independent_sources: int = 1

    def __post_init__(self) -> None:
        for name in ("review_uncertainty", "block_uncertainty", "review_conflict", "minimum_proof_completeness"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if self.block_uncertainty < self.review_uncertainty:
            raise ValueError("block_uncertainty cannot be below review_uncertainty.")
        if isinstance(self.minimum_independent_sources, bool) or not isinstance(self.minimum_independent_sources, int) or not 0 <= self.minimum_independent_sources <= 1000:
            raise ValueError("minimum_independent_sources is invalid.")


@dataclass(frozen=True)
class ReviewDecision:
    route: Literal["automatic", "human_review", "block"]
    priority: float
    reasons: tuple[str, ...]


def route_for_review(
    *,
    aggregate_uncertainty: Any,
    evidence_conflict: Any = 0.0,
    proof_completeness: Any = 1.0,
    independent_sources: int = 1,
    security_violation: bool = False,
    policy: ReviewPolicy | None = None,
) -> ReviewDecision:
    """Choose automatic answer, human review, or hard block with explicit reasons."""

    selected = policy or ReviewPolicy()
    if not isinstance(selected, ReviewPolicy):
        raise ValueError("policy must be ReviewPolicy.")
    uncertainty = _unit(aggregate_uncertainty, "aggregate_uncertainty")
    conflict = _unit(evidence_conflict, "evidence_conflict")
    completeness = _unit(proof_completeness, "proof_completeness")
    if isinstance(independent_sources, bool) or not isinstance(independent_sources, int) or independent_sources < 0:
        raise ValueError("independent_sources must be a non-negative integer.")
    if not isinstance(security_violation, bool):
        raise ValueError("security_violation must be boolean.")
    reasons: list[str] = []
    if security_violation:
        reasons.append("security_violation")
        return ReviewDecision("block", 1.0, tuple(reasons))
    if uncertainty >= selected.block_uncertainty:
        reasons.append("uncertainty_block_threshold")
        return ReviewDecision("block", uncertainty, tuple(reasons))
    if uncertainty >= selected.review_uncertainty:
        reasons.append("high_uncertainty")
    if conflict >= selected.review_conflict:
        reasons.append("evidence_conflict")
    if completeness < selected.minimum_proof_completeness:
        reasons.append("proof_gap")
    if independent_sources < selected.minimum_independent_sources:
        reasons.append("insufficient_independent_sources")
    priority = max(
        uncertainty,
        conflict,
        1.0 - completeness,
        1.0 if independent_sources < selected.minimum_independent_sources else 0.0,
    )
    return ReviewDecision("human_review" if reasons else "automatic", priority, tuple(reasons))


@dataclass(order=True)
class _Queued:
    sort_key: tuple[float, int]
    request_id: str = field(compare=False)
    decision: ReviewDecision = field(compare=False)
    metadata: Mapping[str, Any] = field(compare=False, default_factory=dict)


class ReviewQueue:
    """Bounded in-memory priority queue for review-worthy requests."""

    def __init__(self, *, max_items: int = 10_000) -> None:
        if isinstance(max_items, bool) or not isinstance(max_items, int) or not 1 <= max_items <= _MAX_QUEUE:
            raise ValueError("max_items is invalid.")
        self._max_items = max_items
        self._heap: list[_Queued] = []
        self._sequence = 0

    def __len__(self) -> int:
        return len(self._heap)

    def push(self, request_id: str, decision: ReviewDecision, *, metadata: Mapping[str, Any] | None = None) -> None:
        identifier = _identifier(request_id, "request_id")
        if not isinstance(decision, ReviewDecision):
            raise ValueError("decision must be ReviewDecision.")
        if decision.route != "human_review":
            raise ValueError("only human_review decisions may enter the review queue.")
        values = metadata or {}
        if not isinstance(values, Mapping) or len(values) > 100:
            raise ValueError("metadata must be a bounded mapping.")
        self._sequence += 1
        queued = _Queued((-decision.priority, self._sequence), identifier, decision, dict(values))
        heapq.heappush(self._heap, queued)
        if len(self._heap) > self._max_items:
            worst_index = max(range(len(self._heap)), key=lambda index: self._heap[index].sort_key)
            self._heap[worst_index] = self._heap[-1]
            self._heap.pop()
            heapq.heapify(self._heap)

    def pop(self) -> tuple[str, ReviewDecision, Mapping[str, Any]] | None:
        if not self._heap:
            return None
        queued = heapq.heappop(self._heap)
        return queued.request_id, queued.decision, queued.metadata


__all__ = ["ReviewDecision", "ReviewPolicy", "ReviewQueue", "route_for_review"]
