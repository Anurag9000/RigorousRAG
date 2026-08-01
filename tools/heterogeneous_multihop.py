"""Execute heterogeneous multi-hop retrieval through injected bounded route adapters."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from tools.adaptive_route_experiments import ROUTES
from tools.heterogeneous_budget import (
    allocate_heterogeneous_budget,
    normalize_profiles,
    select_subquestion_route,
)
from tools.heterogeneous_route_types import (
    DEFAULT_ROUTE_PROFILES,
    HeterogeneousHopBudget,
    HeterogeneousMultiHopBudget,
    HeterogeneousMultiHopResult,
    HeterogeneousRouteRequest,
    RouteCostProfile,
    route_name,
)
from tools.multihop_retrieval import (
    HopEvidence,
    run_multihop_retrieval,
)
from tools.query_decomposition import DecompositionPlan, Subquestion


def _adapter_map(
    adapters: Mapping[
        str,
        Callable[[HeterogeneousRouteRequest], Sequence[Any] | Iterable[Any]],
    ],
) -> dict[str, Callable[[HeterogeneousRouteRequest], Any]]:
    if not isinstance(adapters, Mapping):
        raise ValueError("adapters must be a mapping.")
    try:
        rows = list(itertools.islice(adapters.items(), len(ROUTES) + 1))
    except Exception as exc:
        raise ValueError("adapters is not safely iterable.") from exc
    if not rows or len(rows) > len(ROUTES):
        raise ValueError("adapters are empty or exceed the route limit.")
    result: dict[str, Callable[[HeterogeneousRouteRequest], Any]] = {}
    for raw_route, adapter in rows:
        route = route_name(raw_route)
        if route in result:
            raise ValueError("adapters contains duplicate routes.")
        if not callable(adapter):
            raise ValueError("every route adapter must be callable.")
        result[route] = adapter
    return result


def _validate_budget(
    plan: DecompositionPlan,
    budget: HeterogeneousMultiHopBudget,
    adapters: Mapping[str, Callable[..., Any]],
    profiles: Mapping[str, RouteCostProfile] | None,
) -> dict[str, HeterogeneousHopBudget]:
    if not isinstance(budget, HeterogeneousMultiHopBudget):
        raise ValueError("budget must be a HeterogeneousMultiHopBudget.")
    by_id = budget.by_id()
    plan_ids = tuple(question.question_id for question in plan.subquestions)
    if set(by_id) != set(plan_ids):
        raise ValueError("budget question IDs do not match the plan.")
    if any(item.route not in adapters for item in by_id.values()):
        raise ValueError("budget references an unavailable route adapter.")
    profile_map = normalize_profiles(profiles)
    for question in plan.subquestions:
        hop_budget = by_id[question.question_id]
        profile = profile_map.get(hop_budget.route)
        if profile is None:
            raise ValueError("budget route lacks a cost profile.")
        expected = profile.estimate(hop_budget.max_results)
        actual = (
            hop_budget.estimated_cost_units,
            hop_budget.estimated_latency_ms,
            hop_budget.estimated_monetary_microunits,
        )
        if actual != expected:
            raise ValueError("hop budget does not match its route cost profile.")
    return by_id


def run_heterogeneous_multihop(
    plan: DecompositionPlan,
    *,
    adapters: Mapping[
        str,
        Callable[[HeterogeneousRouteRequest], Sequence[Any] | Iterable[Any]],
    ],
    budget: HeterogeneousMultiHopBudget | None = None,
    scope: str = "mixed",
    domain: str = "general",
    route_overrides: Mapping[str, str] | None = None,
    profiles: Mapping[str, RouteCostProfile] | None = None,
    top_k: int = 8,
    total_cost_limit: int = 2_000,
    total_latency_limit_ms: int = 60_000,
    total_monetary_limit_microunits: int = 100_000,
    max_workers: int = 4,
    hop_timeout_seconds: float = 30.0,
    global_timeout_seconds: float = 120.0,
) -> HeterogeneousMultiHopResult:
    """Execute one decomposition DAG across uploaded, web, and scholarly routes.

    Route adapters remain responsible for their own network/client deadlines. The executor's
    global deadline bounds the caller's wait but cannot forcibly terminate Python/provider
    work already running in a worker thread.
    """

    if not isinstance(plan, DecompositionPlan):
        raise ValueError("plan must be a DecompositionPlan.")
    route_adapters = _adapter_map(adapters)
    selected_budget = budget
    if selected_budget is None:
        selected_budget = allocate_heterogeneous_budget(
            plan,
            scope=scope,
            domain=domain,
            available_routes=tuple(route_adapters),
            route_overrides=route_overrides,
            profiles=profiles,
            top_k=top_k,
            total_cost_limit=total_cost_limit,
            total_latency_limit_ms=total_latency_limit_ms,
            total_monetary_limit_microunits=total_monetary_limit_microunits,
        )
    by_id = _validate_budget(
        plan, selected_budget, route_adapters, profiles
    )

    def search(
        question: Subquestion,
        dependencies: tuple[HopEvidence, ...],
    ) -> list[Any]:
        hop_budget = by_id[question.question_id]
        request = HeterogeneousRouteRequest(
            question=question,
            dependencies=dependencies,
            budget=hop_budget,
        )
        raw = route_adapters[hop_budget.route](request)
        if isinstance(raw, (str, bytes, bytearray)):
            raise RuntimeError(
                "heterogeneous adapter returned an invalid evidence collection."
            )
        try:
            rows = list(
                itertools.islice(iter(raw), hop_budget.max_results + 1)
            )
        except Exception as exc:
            raise RuntimeError(
                "heterogeneous adapter returned an invalid evidence collection."
            ) from exc
        return rows[: hop_budget.max_results]

    retrieval = run_multihop_retrieval(
        plan,
        search=search,
        max_workers=max_workers,
        per_hop_limit=max(
            item.max_results for item in selected_budget.allocations
        ),
        hop_timeout_seconds=hop_timeout_seconds,
        global_timeout_seconds=global_timeout_seconds,
        require_dependency_evidence=True,
    )
    return HeterogeneousMultiHopResult(retrieval, selected_budget)


__all__ = [
    "DEFAULT_ROUTE_PROFILES",
    "HeterogeneousHopBudget",
    "HeterogeneousMultiHopBudget",
    "HeterogeneousMultiHopResult",
    "HeterogeneousRouteRequest",
    "RouteCostProfile",
    "allocate_heterogeneous_budget",
    "run_heterogeneous_multihop",
    "select_subquestion_route",
]
