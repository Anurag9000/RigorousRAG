"""Offline route experiments for dense, sparse, hybrid, web, and scholarly retrieval.

The harness accepts injected adapters and never performs network access itself. Reports contain
case IDs and aggregate metrics, not raw queries or evidence text.
"""

from __future__ import annotations

import itertools
import math
import operator
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tools.adaptive_retrieval import EvidenceSignals, analyze_query, evaluate_evidence, initial_attempt

ROUTES = ("dense", "corpus-sparse", "corpus-hybrid", "web", "scholarly")
_ROUTE_SET = frozenset(ROUTES)
_SCOPES = frozenset({"uploaded", "public", "mixed"})
_DOMAINS = frozenset({"general", "scholarly"})
_MAX_CASES = 10_000
_MAX_EVIDENCE = 100
_MAX_RELEVANT_IDS = 1_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        rendered = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= rendered <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return rendered


def _finite(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        rendered = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(rendered) or not minimum <= rendered <= maximum:
        raise ValueError(f"{label} must be finite and between {minimum} and {maximum}.")
    return rendered


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        try:
            return value.get(name, default)
        except Exception:
            return default
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _evidence_identifier(value: Any) -> str | None:
    for name in ("chunk_id", "source_id", "evidence_id", "doc_id", "url", "doi"):
        candidate = _safe_attr(value, name, None)
        if isinstance(candidate, str):
            rendered = candidate.strip()
            if rendered and len(rendered) <= 1_000 and not any(
                ord(character) < 32 or ord(character) == 127 for character in rendered
            ):
                return rendered
    return None


def _bounded_evidence(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError("route adapter returned an invalid evidence collection.")
    try:
        return list(itertools.islice(iter(values), _MAX_EVIDENCE + 1))[:_MAX_EVIDENCE]
    except Exception as exc:
        raise RuntimeError("route adapter returned an invalid evidence collection.") from exc


@dataclass(frozen=True)
class RouteExperimentCase:
    case_id: str
    query: str
    scope: str = "mixed"
    domain: str = "general"
    relevant_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id", 200))
        if not isinstance(self.query, str):
            raise ValueError("query must be a string.")
        analyze_query(self.query)
        if self.scope not in _SCOPES:
            raise ValueError("scope must be uploaded, public, or mixed.")
        if self.domain not in _DOMAINS:
            raise ValueError("domain must be general or scholarly.")
        if not isinstance(self.relevant_ids, frozenset):
            raise ValueError("relevant_ids must be a frozenset.")
        if len(self.relevant_ids) > _MAX_RELEVANT_IDS:
            raise ValueError("relevant_ids exceed the case limit.")
        object.__setattr__(
            self,
            "relevant_ids",
            frozenset(_identifier(value, "relevant_id", 1_000) for value in self.relevant_ids),
        )


@dataclass(frozen=True)
class RouteExecution:
    evidence: Sequence[Any] | Iterable[Any]
    cost_units: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class RouteObservation:
    case_id: str
    route: str
    signals: EvidenceSignals
    relevant_hits: int
    relevant_total: int
    quality: float
    utility: float
    cost_units: float
    latency_ms: float
    error_type: str | None = None

    @property
    def success(self) -> bool:
        if self.error_type is not None:
            return False
        if self.relevant_total:
            return self.relevant_hits > 0
        return self.signals.decision == "sufficient"


@dataclass(frozen=True)
class RouteCaseResult:
    case_id: str
    selected_route: str
    best_route: str
    selected_success: bool
    oracle_success: bool
    selection_correct: bool
    regret: float
    observations: tuple[RouteObservation, ...]


@dataclass(frozen=True)
class RouteAggregate:
    route: str
    observations: int
    successes: int
    mean_quality: float
    mean_utility: float
    mean_sufficiency: float
    mean_cost_units: float
    mean_latency_ms: float
    error_count: int


@dataclass(frozen=True)
class RouteBenchmarkReport:
    case_count: int
    selected_success_rate: float
    oracle_success_rate: float
    route_selection_accuracy: float
    mean_regret: float
    mean_selected_cost_units: float
    mean_selected_latency_ms: float
    per_route: tuple[RouteAggregate, ...]
    cases: tuple[RouteCaseResult, ...]


def select_route(case: RouteExperimentCase, *, available_routes: Iterable[str] = ROUTES) -> str:
    if not isinstance(case, RouteExperimentCase):
        raise ValueError("case must be a RouteExperimentCase.")
    if isinstance(available_routes, (str, bytes, bytearray)):
        raise ValueError("available_routes must be an iterable of route names.")
    try:
        bounded_available = list(itertools.islice(iter(available_routes), len(ROUTES) + 1))
    except Exception as exc:
        raise ValueError("available_routes is not safely iterable.") from exc
    if len(bounded_available) > len(ROUTES):
        raise ValueError("available_routes contains too many entries.")
    if any(not isinstance(route, str) for route in bounded_available):
        raise ValueError("available_routes must contain strings.")
    available = tuple(dict.fromkeys(bounded_available))
    if not available or any(route not in _ROUTE_SET for route in available):
        raise ValueError("available_routes contains an unsupported route.")
    analysis = analyze_query(case.query)
    if case.scope == "uploaded":
        preferred = initial_attempt(case.query).mode
    elif case.scope == "public":
        preferred = "scholarly" if case.domain == "scholarly" or analysis.citation_seeking else "web"
    elif analysis.exact_identifier:
        preferred = "corpus-sparse"
    elif analysis.temporal:
        preferred = "web"
    elif analysis.comparative:
        # Mixed-scope comparisons need local hybrid evidence even when the query
        # also contains generic methodological nouns such as "methods".
        preferred = "corpus-hybrid"
    elif case.domain == "scholarly" or analysis.citation_seeking:
        preferred = "scholarly"
    elif analysis.methodological:
        preferred = "corpus-hybrid"
    else:
        preferred = "corpus-hybrid"
    if preferred in available:
        return preferred
    fallbacks = (
        "corpus-hybrid",
        "corpus-sparse",
        "dense",
        "scholarly",
        "web",
    )
    return next(route for route in fallbacks if route in available)


def _observe(
    case: RouteExperimentCase,
    route: str,
    execution: RouteExecution,
    *,
    cost_weight: float,
    latency_weight: float,
    error_type: str | None = None,
) -> RouteObservation:
    evidence = _bounded_evidence(execution.evidence)
    signals = evaluate_evidence(evidence)
    found = {_evidence_identifier(value) for value in evidence}
    found.discard(None)
    hits = len(case.relevant_ids & found)
    relevance_quality = hits / len(case.relevant_ids) if case.relevant_ids else signals.sufficiency
    quality = max(0.0, min(1.0, 0.65 * signals.sufficiency + 0.35 * relevance_quality))
    cost = _finite(execution.cost_units, "cost_units", 0.0, 1_000_000_000.0)
    latency = _finite(execution.latency_ms, "latency_ms", 0.0, 86_400_000.0)
    utility = quality - cost_weight * min(cost / 1_000.0, 1.0) - latency_weight * min(latency / 10_000.0, 1.0)
    return RouteObservation(
        case_id=case.case_id,
        route=route,
        signals=signals,
        relevant_hits=hits,
        relevant_total=len(case.relevant_ids),
        quality=round(quality, 9),
        utility=round(utility, 9),
        cost_units=round(cost, 9),
        latency_ms=round(latency, 9),
        error_type=error_type,
    )


def run_route_benchmark(
    cases: Iterable[RouteExperimentCase],
    *,
    adapters: Mapping[str, Callable[[RouteExperimentCase], RouteExecution | Iterable[Any]]],
    cost_weight: float = 0.05,
    latency_weight: float = 0.05,
    clock: Callable[[], float] = time.perf_counter,
) -> RouteBenchmarkReport:
    if isinstance(cases, (str, bytes, bytearray)):
        raise ValueError("cases must be an iterable of RouteExperimentCase values.")
    try:
        bounded_cases = list(itertools.islice(iter(cases), _MAX_CASES + 1))
    except Exception as exc:
        raise ValueError("cases are not safely iterable.") from exc
    if len(bounded_cases) > _MAX_CASES:
        raise ValueError("route experiment case limit exceeded.")
    if any(not isinstance(case, RouteExperimentCase) for case in bounded_cases):
        raise ValueError("every case must be a RouteExperimentCase.")
    if len({case.case_id for case in bounded_cases}) != len(bounded_cases):
        raise ValueError("route experiment case IDs must be unique.")
    if not isinstance(adapters, Mapping):
        raise ValueError("adapters must be a mapping.")
    route_adapters: dict[str, Callable[[RouteExperimentCase], RouteExecution | Iterable[Any]]] = {}
    try:
        adapter_items = list(itertools.islice(adapters.items(), len(ROUTES) + 1))
    except Exception as exc:
        raise ValueError("adapters is not safely iterable.") from exc
    if len(adapter_items) > len(ROUTES):
        raise ValueError("adapters contains too many routes.")
    for route, adapter in adapter_items:
        if route not in _ROUTE_SET or not callable(adapter):
            raise ValueError("adapters contains an unsupported route or non-callable adapter.")
        route_adapters[route] = adapter
    if not route_adapters:
        raise ValueError("at least one route adapter is required.")
    cost_penalty = _finite(cost_weight, "cost_weight", 0.0, 1.0)
    latency_penalty = _finite(latency_weight, "latency_weight", 0.0, 1.0)
    if not callable(clock):
        raise ValueError("clock must be callable.")

    case_results: list[RouteCaseResult] = []
    route_rows: dict[str, list[RouteObservation]] = {route: [] for route in route_adapters}
    for case in bounded_cases:
        observations: list[RouteObservation] = []
        for route, adapter in route_adapters.items():
            try:
                started = _finite(clock(), "clock", -1.0e18, 1.0e18)
            except Exception as exc:
                raise ValueError("clock returned an invalid value.") from exc
            error_type: str | None = None
            try:
                raw = adapter(case)
                finished = _finite(clock(), "clock", -1.0e18, 1.0e18)
                elapsed = max(0.0, (finished - started) * 1_000.0)
                execution = raw if isinstance(raw, RouteExecution) else RouteExecution(raw, latency_ms=elapsed)
                if execution.latency_ms == 0.0:
                    execution = RouteExecution(
                        evidence=execution.evidence,
                        cost_units=execution.cost_units,
                        latency_ms=elapsed,
                    )
            except Exception as exc:
                try:
                    finished = _finite(clock(), "clock", -1.0e18, 1.0e18)
                    elapsed = max(0.0, (finished - started) * 1_000.0)
                except Exception:
                    elapsed = 0.0
                execution = RouteExecution((), latency_ms=elapsed)
                error_type = type(exc).__name__[:200]
            try:
                observation = _observe(
                    case,
                    route,
                    execution,
                    cost_weight=cost_penalty,
                    latency_weight=latency_penalty,
                    error_type=error_type,
                )
            except Exception as exc:
                observation = _observe(
                    case,
                    route,
                    RouteExecution((), latency_ms=elapsed),
                    cost_weight=cost_penalty,
                    latency_weight=latency_penalty,
                    error_type=type(exc).__name__[:200],
                )
            observations.append(observation)
            route_rows[route].append(observation)
        selected_route = select_route(case, available_routes=route_adapters)
        selected = next(row for row in observations if row.route == selected_route)
        best = max(
            observations,
            key=lambda row: (row.utility, row.quality, -row.cost_units, -row.latency_ms, row.route),
        )
        case_results.append(
            RouteCaseResult(
                case_id=case.case_id,
                selected_route=selected_route,
                best_route=best.route,
                selected_success=selected.success,
                oracle_success=best.success,
                selection_correct=selected_route == best.route,
                regret=round(max(0.0, best.utility - selected.utility), 9),
                observations=tuple(observations),
            )
        )

    per_route: list[RouteAggregate] = []
    for route in ROUTES:
        rows = route_rows.get(route, [])
        if not rows:
            continue
        per_route.append(
            RouteAggregate(
                route=route,
                observations=len(rows),
                successes=sum(row.success for row in rows),
                mean_quality=round(sum(row.quality for row in rows) / len(rows), 9),
                mean_utility=round(sum(row.utility for row in rows) / len(rows), 9),
                mean_sufficiency=round(
                    sum(row.signals.sufficiency for row in rows) / len(rows), 9
                ),
                mean_cost_units=round(sum(row.cost_units for row in rows) / len(rows), 9),
                mean_latency_ms=round(sum(row.latency_ms for row in rows) / len(rows), 9),
                error_count=sum(row.error_type is not None for row in rows),
            )
        )
    count = len(case_results)
    if not count:
        return RouteBenchmarkReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tuple(per_route), ())
    selected_rows = [
        next(row for row in case.observations if row.route == case.selected_route)
        for case in case_results
    ]
    return RouteBenchmarkReport(
        case_count=count,
        selected_success_rate=round(sum(case.selected_success for case in case_results) / count, 9),
        oracle_success_rate=round(sum(case.oracle_success for case in case_results) / count, 9),
        route_selection_accuracy=round(sum(case.selection_correct for case in case_results) / count, 9),
        mean_regret=round(sum(case.regret for case in case_results) / count, 9),
        mean_selected_cost_units=round(sum(row.cost_units for row in selected_rows) / count, 9),
        mean_selected_latency_ms=round(sum(row.latency_ms for row in selected_rows) / count, 9),
        per_route=tuple(per_route),
        cases=tuple(case_results),
    )


__all__ = [
    "ROUTES",
    "RouteAggregate",
    "RouteBenchmarkReport",
    "RouteCaseResult",
    "RouteExecution",
    "RouteExperimentCase",
    "RouteObservation",
    "run_route_benchmark",
    "select_route",
]
