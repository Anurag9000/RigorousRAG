"""Deterministic per-hop routing and conservative cross-backend allocation."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping
from typing import Any

from tools.adaptive_route_experiments import (
    ROUTES,
    RouteExperimentCase,
    select_route,
)
from tools.heterogeneous_route_types import (
    DEFAULT_ROUTE_PROFILES,
    HeterogeneousHopBudget,
    HeterogeneousMultiHopBudget,
    MAX_COST,
    MAX_LATENCY_MS,
    MAX_MONEY_MICROUNITS,
    MAX_RESULTS,
    RouteCostProfile,
    domain_name,
    exact_integer,
    route_name,
    scope_name,
)
from tools.query_decomposition import DecompositionPlan, Subquestion


def _bounded_mapping_items(
    values: Mapping[Any, Any], label: str, maximum: int
) -> list[tuple[Any, Any]]:
    try:
        rows = list(itertools.islice(values.items(), maximum + 1))
    except Exception as exc:
        raise ValueError(f"{label} is not safely iterable.") from exc
    if len(rows) > maximum:
        raise ValueError(f"{label} contains too many entries.")
    return rows


def normalize_profiles(
    values: Mapping[str, RouteCostProfile] | None,
) -> dict[str, RouteCostProfile]:
    source = DEFAULT_ROUTE_PROFILES if values is None else values
    if not isinstance(source, Mapping):
        raise ValueError("profiles must be a mapping.")
    result: dict[str, RouteCostProfile] = {}
    for raw_route, profile in _bounded_mapping_items(
        source, "profiles", len(ROUTES)
    ):
        route = route_name(raw_route)
        if route in result:
            raise ValueError("profiles contains duplicate route entries.")
        if not isinstance(profile, RouteCostProfile) or profile.route != route:
            raise ValueError("profile keys must match RouteCostProfile.route.")
        result[route] = profile
    if not result:
        raise ValueError("at least one route profile is required.")
    return result


def _available_routes(
    values: Iterable[str], profile_map: Mapping[str, RouteCostProfile]
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("available_routes must be an iterable of routes.")
    try:
        rows = list(itertools.islice(iter(values), len(ROUTES) + 1))
    except Exception as exc:
        raise ValueError("available_routes is not safely iterable.") from exc
    if not rows or len(rows) > len(ROUTES):
        raise ValueError("available_routes is empty or exceeds the route limit.")
    result: list[str] = []
    seen: set[str] = set()
    for raw_route in rows:
        route = route_name(raw_route)
        if route not in profile_map:
            raise ValueError("available route lacks a cost profile.")
        if route not in seen:
            seen.add(route)
            result.append(route)
    return tuple(result)


def _route_overrides(
    values: Mapping[str, str] | None,
    plan: DecompositionPlan,
) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("route_overrides must be a mapping.")
    valid_ids = {item.question_id for item in plan.subquestions}
    result: dict[str, str] = {}
    for raw_question_id, raw_route in _bounded_mapping_items(
        values, "route_overrides", len(plan.subquestions)
    ):
        if not isinstance(raw_question_id, str) or raw_question_id not in valid_ids:
            raise ValueError("route override references an unknown question.")
        if raw_question_id in result:
            raise ValueError("route_overrides contains duplicate question IDs.")
        result[raw_question_id] = route_name(raw_route)
    return result


def _weight(node: Subquestion, terminal: bool) -> float:
    value = 1.0 + 0.20 * len(node.depends_on)
    if node.relation in {"compare", "synthesize"}:
        value += 0.50
    elif node.relation in {"explain", "temporal"}:
        value += 0.30
    if terminal:
        value += 0.50
    return value


def select_subquestion_route(
    question: Subquestion,
    *,
    scope: str = "mixed",
    domain: str = "general",
    available_routes: Iterable[str] = ROUTES,
) -> str:
    if not isinstance(question, Subquestion):
        raise ValueError("question must be a Subquestion.")
    return select_route(
        RouteExperimentCase(
            case_id=question.question_id,
            query=question.text,
            scope=scope_name(scope),
            domain=domain_name(domain),
        ),
        available_routes=available_routes,
    )


def allocate_heterogeneous_budget(
    plan: DecompositionPlan,
    *,
    scope: str = "mixed",
    domain: str = "general",
    available_routes: Iterable[str] = ROUTES,
    route_overrides: Mapping[str, str] | None = None,
    profiles: Mapping[str, RouteCostProfile] | None = None,
    top_k: int = 8,
    total_cost_limit: int = 2_000,
    total_latency_limit_ms: int = 60_000,
    total_monetary_limit_microunits: int = 100_000,
) -> HeterogeneousMultiHopBudget:
    """Allocate additive conservative ceilings across all selected hop routes.

    Latency is intentionally summed rather than discounted for parallel batches. This makes
    the estimate a conservative accounting ceiling, not a prediction of wall-clock runtime.
    """

    if not isinstance(plan, DecompositionPlan):
        raise ValueError("plan must be a DecompositionPlan.")
    selected_scope = scope_name(scope)
    selected_domain = domain_name(domain)
    desired_results = exact_integer(top_k, "top_k", 1, MAX_RESULTS)
    cost_limit = exact_integer(
        total_cost_limit, "total_cost_limit", 1, MAX_COST
    )
    latency_limit = exact_integer(
        total_latency_limit_ms,
        "total_latency_limit_ms",
        1,
        MAX_LATENCY_MS,
    )
    money_limit = exact_integer(
        total_monetary_limit_microunits,
        "total_monetary_limit_microunits",
        0,
        MAX_MONEY_MICROUNITS,
    )
    profile_map = normalize_profiles(profiles)
    available = _available_routes(available_routes, profile_map)
    overrides = _route_overrides(route_overrides, plan)
    terminal_ids = set(plan.terminal_questions)
    state: dict[str, dict[str, Any]] = {}
    for node in plan.subquestions:
        route = overrides.get(node.question_id)
        if route is None:
            route = select_subquestion_route(
                node,
                scope=selected_scope,
                domain=selected_domain,
                available_routes=available,
            )
        if route not in available:
            raise ValueError("route override is not available.")
        profile = profile_map[route]
        cost, latency, money = profile.estimate(1)
        state[node.question_id] = {
            "node": node,
            "route": route,
            "profile": profile,
            "results": 1,
            "cost": cost,
            "latency": latency,
            "money": money,
            "weight": _weight(node, node.question_id in terminal_ids),
        }
    totals = [
        sum(value["cost"] for value in state.values()),
        sum(value["latency"] for value in state.values()),
        sum(value["money"] for value in state.values()),
    ]
    if (
        totals[0] > cost_limit
        or totals[1] > latency_limit
        or totals[2] > money_limit
    ):
        raise ValueError(
            "global resource limits are below the minimum route allocation."
        )
    ranked = sorted(
        state,
        key=lambda question_id: (
            state[question_id]["weight"],
            question_id,
        ),
        reverse=True,
    )
    while True:
        progress = False
        for selected_id in ranked:
            value = state[selected_id]
            if value["results"] >= desired_results:
                continue
            profile = value["profile"]
            next_results = value["results"] + 1
            next_cost, next_latency, next_money = profile.estimate(next_results)
            delta = (
                next_cost - value["cost"],
                next_latency - value["latency"],
                next_money - value["money"],
            )
            if (
                totals[0] + delta[0] > cost_limit
                or totals[1] + delta[1] > latency_limit
                or totals[2] + delta[2] > money_limit
            ):
                continue
            value["results"] = next_results
            value["cost"] = next_cost
            value["latency"] = next_latency
            value["money"] = next_money
            totals[0] += delta[0]
            totals[1] += delta[1]
            totals[2] += delta[2]
            progress = True
        if not progress or all(
            value["results"] >= desired_results for value in state.values()
        ):
            break
    allocations = tuple(
        HeterogeneousHopBudget(
            question_id=node.question_id,
            route=state[node.question_id]["route"],
            max_results=state[node.question_id]["results"],
            estimated_cost_units=state[node.question_id]["cost"],
            estimated_latency_ms=state[node.question_id]["latency"],
            estimated_monetary_microunits=state[node.question_id]["money"],
            weight=round(state[node.question_id]["weight"], 6),
        )
        for node in plan.subquestions
    )
    return HeterogeneousMultiHopBudget(
        total_cost_limit=cost_limit,
        total_latency_limit_ms=latency_limit,
        total_monetary_limit_microunits=money_limit,
        allocated_cost_units=totals[0],
        allocated_latency_ms=totals[1],
        allocated_monetary_microunits=totals[2],
        allocations=allocations,
    )


__all__ = [
    "allocate_heterogeneous_budget",
    "normalize_profiles",
    "select_subquestion_route",
]
