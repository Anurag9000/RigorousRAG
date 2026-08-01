"""Validated route, budget, and result records for heterogeneous multi-hop RAG."""

from __future__ import annotations

import math
import operator
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tools.adaptive_route_experiments import ROUTES
from tools.multihop_retrieval import HopEvidence, MultiHopResult
from tools.query_decomposition import Subquestion

ROUTE_SET = frozenset(ROUTES)
SCOPES = frozenset({"uploaded", "public", "mixed"})
DOMAINS = frozenset({"general", "scholarly"})
MAX_RESULTS = 50
MAX_COST = 1_000_000
MAX_LATENCY_MS = 86_400_000
MAX_MONEY_MICROUNITS = 1_000_000_000
_QUESTION_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")


def exact_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def finite_number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def route_name(value: Any) -> str:
    if not isinstance(value, str) or value not in ROUTE_SET:
        raise ValueError("route is unsupported.")
    return value


def scope_name(value: Any) -> str:
    if not isinstance(value, str) or value not in SCOPES:
        raise ValueError("scope must be uploaded, public, or mixed.")
    return value


def domain_name(value: Any) -> str:
    if not isinstance(value, str) or value not in DOMAINS:
        raise ValueError("domain must be general or scholarly.")
    return value


def question_id(value: Any) -> str:
    if not isinstance(value, str) or _QUESTION_ID_RE.fullmatch(value) is None:
        raise ValueError("question_id is invalid.")
    return value


@dataclass(frozen=True)
class RouteCostProfile:
    """Heuristic accounting coefficients, not provider prices or SLA guarantees."""

    route: str
    fixed_cost_units: int
    per_result_cost_units: int
    fixed_latency_ms: int
    per_result_latency_ms: int
    fixed_monetary_microunits: int = 0
    per_result_monetary_microunits: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", route_name(self.route))
        for field_name, maximum in (
            ("fixed_cost_units", MAX_COST),
            ("per_result_cost_units", MAX_COST),
            ("fixed_latency_ms", MAX_LATENCY_MS),
            ("per_result_latency_ms", MAX_LATENCY_MS),
            ("fixed_monetary_microunits", MAX_MONEY_MICROUNITS),
            ("per_result_monetary_microunits", MAX_MONEY_MICROUNITS),
        ):
            object.__setattr__(
                self,
                field_name,
                exact_integer(getattr(self, field_name), field_name, 0, maximum),
            )
        if (
            self.fixed_cost_units + self.per_result_cost_units <= 0
            and self.fixed_latency_ms + self.per_result_latency_ms <= 0
            and self.fixed_monetary_microunits
            + self.per_result_monetary_microunits
            <= 0
        ):
            raise ValueError("route profile must consume at least one resource.")

    def estimate(self, results: int) -> tuple[int, int, int]:
        count = exact_integer(results, "results", 1, MAX_RESULTS)
        cost = self.fixed_cost_units + count * self.per_result_cost_units
        latency = self.fixed_latency_ms + count * self.per_result_latency_ms
        money = (
            self.fixed_monetary_microunits
            + count * self.per_result_monetary_microunits
        )
        if cost > MAX_COST or latency > MAX_LATENCY_MS or money > MAX_MONEY_MICROUNITS:
            raise ValueError("route profile estimate exceeds the supported resource range.")
        return cost, latency, money


# These values are conservative experiment-accounting defaults. They are deliberately not
# described as vendor billing prices, measured p95 latency, or deployment guarantees.
DEFAULT_ROUTE_PROFILES = MappingProxyType(
    {
        "dense": RouteCostProfile("dense", 2, 1, 200, 40),
        "corpus-sparse": RouteCostProfile("corpus-sparse", 1, 1, 80, 20),
        "corpus-hybrid": RouteCostProfile("corpus-hybrid", 3, 2, 300, 60),
        "web": RouteCostProfile("web", 10, 3, 1_000, 250, 100, 20),
        "scholarly": RouteCostProfile(
            "scholarly", 8, 3, 900, 220, 80, 15
        ),
    }
)


@dataclass(frozen=True)
class HeterogeneousHopBudget:
    question_id: str
    route: str
    max_results: int
    estimated_cost_units: int
    estimated_latency_ms: int
    estimated_monetary_microunits: int
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", question_id(self.question_id))
        object.__setattr__(self, "route", route_name(self.route))
        object.__setattr__(
            self,
            "max_results",
            exact_integer(self.max_results, "max_results", 1, MAX_RESULTS),
        )
        for field_name, maximum in (
            ("estimated_cost_units", MAX_COST),
            ("estimated_latency_ms", MAX_LATENCY_MS),
            ("estimated_monetary_microunits", MAX_MONEY_MICROUNITS),
        ):
            object.__setattr__(
                self,
                field_name,
                exact_integer(getattr(self, field_name), field_name, 0, maximum),
            )
        object.__setattr__(
            self,
            "weight",
            finite_number(self.weight, "weight", 0.000001, 1_000.0),
        )


@dataclass(frozen=True)
class HeterogeneousMultiHopBudget:
    total_cost_limit: int
    total_latency_limit_ms: int
    total_monetary_limit_microunits: int
    allocated_cost_units: int
    allocated_latency_ms: int
    allocated_monetary_microunits: int
    allocations: tuple[HeterogeneousHopBudget, ...]

    def __post_init__(self) -> None:
        limits = (
            ("total_cost_limit", 1, MAX_COST),
            ("total_latency_limit_ms", 1, MAX_LATENCY_MS),
            (
                "total_monetary_limit_microunits",
                0,
                MAX_MONEY_MICROUNITS,
            ),
            ("allocated_cost_units", 0, MAX_COST),
            ("allocated_latency_ms", 0, MAX_LATENCY_MS),
            (
                "allocated_monetary_microunits",
                0,
                MAX_MONEY_MICROUNITS,
            ),
        )
        for field_name, minimum, maximum in limits:
            object.__setattr__(
                self,
                field_name,
                exact_integer(
                    getattr(self, field_name), field_name, minimum, maximum
                ),
            )
        if not isinstance(self.allocations, tuple):
            raise ValueError("allocations must be a tuple.")
        if not self.allocations or len(self.allocations) > 12:
            raise ValueError("allocations are empty or exceed the hop limit.")
        if any(
            not isinstance(item, HeterogeneousHopBudget)
            for item in self.allocations
        ):
            raise ValueError("allocations contain an invalid hop budget.")
        if len({item.question_id for item in self.allocations}) != len(
            self.allocations
        ):
            raise ValueError("allocation question IDs must be unique.")
        expected = (
            sum(item.estimated_cost_units for item in self.allocations),
            sum(item.estimated_latency_ms for item in self.allocations),
            sum(
                item.estimated_monetary_microunits
                for item in self.allocations
            ),
        )
        actual = (
            self.allocated_cost_units,
            self.allocated_latency_ms,
            self.allocated_monetary_microunits,
        )
        if expected != actual:
            raise ValueError("allocated resource totals do not match hop budgets.")
        if (
            actual[0] > self.total_cost_limit
            or actual[1] > self.total_latency_limit_ms
            or actual[2] > self.total_monetary_limit_microunits
        ):
            raise ValueError("allocated resources exceed a global limit.")

    def by_id(self) -> dict[str, HeterogeneousHopBudget]:
        return {item.question_id: item for item in self.allocations}


@dataclass(frozen=True)
class HeterogeneousRouteRequest:
    question: Subquestion
    dependencies: tuple[HopEvidence, ...]
    budget: HeterogeneousHopBudget

    def __post_init__(self) -> None:
        if not isinstance(self.question, Subquestion):
            raise ValueError("question must be a Subquestion.")
        if not isinstance(self.dependencies, tuple) or any(
            not isinstance(item, HopEvidence) for item in self.dependencies
        ):
            raise ValueError("dependencies must be a tuple of HopEvidence values.")
        if not isinstance(self.budget, HeterogeneousHopBudget):
            raise ValueError("budget must be a HeterogeneousHopBudget.")
        if self.budget.question_id != self.question.question_id:
            raise ValueError("budget question_id does not match the question.")


@dataclass(frozen=True)
class HeterogeneousMultiHopResult:
    retrieval: MultiHopResult
    budget: HeterogeneousMultiHopBudget

    def __post_init__(self) -> None:
        if not isinstance(self.retrieval, MultiHopResult):
            raise ValueError("retrieval must be a MultiHopResult.")
        if not isinstance(self.budget, HeterogeneousMultiHopBudget):
            raise ValueError("budget must be a HeterogeneousMultiHopBudget.")

    @property
    def routes_by_hop(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (item.question_id, item.route) for item in self.budget.allocations
        )


__all__ = [
    "DEFAULT_ROUTE_PROFILES",
    "DOMAINS",
    "HeterogeneousHopBudget",
    "HeterogeneousMultiHopBudget",
    "HeterogeneousMultiHopResult",
    "HeterogeneousRouteRequest",
    "MAX_COST",
    "MAX_LATENCY_MS",
    "MAX_MONEY_MICROUNITS",
    "MAX_RESULTS",
    "ROUTE_SET",
    "RouteCostProfile",
    "SCOPES",
    "domain_name",
    "exact_integer",
    "finite_number",
    "question_id",
    "route_name",
    "scope_name",
]
