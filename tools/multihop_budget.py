"""Deterministic global estimated-cost allocation for multi-hop retrieval."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any

from tools.adaptive_retrieval import initial_attempt
from tools.query_decomposition import DecompositionPlan, Subquestion

_MAX_TOTAL_COST = 100_000
_MAX_PER_HOP_COST = 10_000


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


@dataclass(frozen=True)
class HopBudget:
    question_id: str
    minimum_cost: int
    max_estimated_cost: int
    weight: float

    def __post_init__(self) -> None:
        if self.minimum_cost < 1 or self.max_estimated_cost < self.minimum_cost:
            raise ValueError("hop budget is below its minimum cost.")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("hop budget weight must be finite and positive.")


@dataclass(frozen=True)
class MultiHopBudget:
    total_limit: int
    allocated_cost: int
    unallocated_cost: int
    per_hop_limit: int
    budgets: tuple[HopBudget, ...]

    def __post_init__(self) -> None:
        if self.allocated_cost != sum(item.max_estimated_cost for item in self.budgets):
            raise ValueError("allocated cost does not match hop budgets.")
        if self.allocated_cost + self.unallocated_cost != self.total_limit:
            raise ValueError("budget accounting does not match the total limit.")
        if any(item.max_estimated_cost > self.per_hop_limit for item in self.budgets):
            raise ValueError("a hop budget exceeds the per-hop limit.")

    def by_id(self) -> dict[str, HopBudget]:
        return {item.question_id: item for item in self.budgets}


def _weight(node: Subquestion) -> float:
    value = 1.0 + 0.18 * len(node.depends_on)
    if node.relation in {"compare", "synthesize"}:
        value += 0.45
    elif node.relation in {"explain", "temporal"}:
        value += 0.25
    value += min(len(node.entities), 5) * 0.04
    value += min(len(node.temporal_constraints), 5) * 0.04
    return value


def allocate_multihop_budget(
    plan: DecompositionPlan,
    *,
    top_k: int = 8,
    total_limit: int = 1_200,
    per_hop_limit: int = 500,
) -> MultiHopBudget:
    """Allocate a hard global ceiling while preserving each hop's initial attempt."""

    if not isinstance(plan, DecompositionPlan):
        raise ValueError("plan must be a DecompositionPlan.")
    requested = _integer(top_k, "top_k", 1, 50)
    total = _integer(total_limit, "total_limit", 1, _MAX_TOTAL_COST)
    per_hop = _integer(per_hop_limit, "per_hop_limit", 1, _MAX_PER_HOP_COST)
    nodes = plan.subquestions
    minima = {
        node.question_id: initial_attempt(node.text, top_k=requested).estimated_cost
        for node in nodes
    }
    if any(cost > per_hop for cost in minima.values()):
        raise ValueError("per_hop_limit is below a required initial retrieval attempt.")
    minimum_total = sum(minima.values())
    if minimum_total > total:
        raise ValueError(
            "total_limit is below the minimum required for all decomposition hops."
        )

    allocations = dict(minima)
    capacities = {
        node.question_id: per_hop - minima[node.question_id] for node in nodes
    }
    weights = {node.question_id: _weight(node) for node in nodes}
    remaining = total - minimum_total
    while remaining > 0:
        active = [
            node.question_id
            for node in nodes
            if allocations[node.question_id] - minima[node.question_id]
            < capacities[node.question_id]
        ]
        if not active:
            break
        weight_sum = sum(weights[identifier] for identifier in active)
        exact = {
            identifier: remaining * weights[identifier] / weight_sum
            for identifier in active
        }
        distributed = 0
        for identifier in active:
            capacity = (
                minima[identifier]
                + capacities[identifier]
                - allocations[identifier]
            )
            grant = min(capacity, int(math.floor(exact[identifier])))
            allocations[identifier] += grant
            distributed += grant
        remaining -= distributed
        if remaining <= 0:
            break
        ranked = sorted(
            active,
            key=lambda identifier: (
                exact[identifier] - math.floor(exact[identifier]),
                weights[identifier],
                identifier,
            ),
            reverse=True,
        )
        progress = False
        for identifier in ranked:
            if remaining <= 0:
                break
            if allocations[identifier] >= minima[identifier] + capacities[identifier]:
                continue
            allocations[identifier] += 1
            remaining -= 1
            progress = True
        if not progress and distributed == 0:
            break

    budgets = tuple(
        HopBudget(
            question_id=node.question_id,
            minimum_cost=minima[node.question_id],
            max_estimated_cost=allocations[node.question_id],
            weight=round(weights[node.question_id], 6),
        )
        for node in nodes
    )
    allocated = sum(item.max_estimated_cost for item in budgets)
    return MultiHopBudget(
        total_limit=total,
        allocated_cost=allocated,
        unallocated_cost=total - allocated,
        per_hop_limit=per_hop,
        budgets=budgets,
    )


__all__ = ["HopBudget", "MultiHopBudget", "allocate_multihop_budget"]
