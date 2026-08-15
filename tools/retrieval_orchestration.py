"""Cross-profile and multi-route retrieval orchestration.

Retrieval backends return the shared ``RetrievalCandidate`` contract. This layer invokes
several model/profile routes, normalizes per-route ranks, reconciles candidate identity,
applies reciprocal-rank/score fusion and records route-level provenance. It performs no
model loading and never changes owner authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence

from tools.hybrid_retrieval import RetrievalCandidate

_MAX_ROUTES = 32
_MAX_RESULTS = 5000


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _weight(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or result < 0 or result > 1000:
        raise ValueError(f"{label} is invalid")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class RetrievalRoute:
    route_id: str
    backend_id: str
    profile_id: str
    mode: str
    weight: float = 1.0
    top_k: int = 100

    def __post_init__(self) -> None:
        for name, maximum in (("route_id", 128), ("backend_id", 256), ("profile_id", 256), ("mode", 64)):
            object.__setattr__(self, name, _text(getattr(self, name), name, maximum))
        object.__setattr__(self, "weight", _weight(self.weight, "weight"))
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or not 1 <= self.top_k <= 1000:
            raise ValueError("top_k is invalid")


class RetrievalRouteBackend(Protocol):
    def retrieve(self, query: str, *, route: RetrievalRoute) -> Sequence[RetrievalCandidate]: ...


@dataclass(frozen=True)
class RouteTrace:
    route_id: str
    success: bool
    candidate_count: int
    error_type: str = ""


@dataclass(frozen=True)
class OrchestratedCandidate:
    candidate: RetrievalCandidate
    fused_score: float
    route_scores: Mapping[str, float]
    route_ranks: Mapping[str, int]
    route_ids: tuple[str, ...]


@dataclass(frozen=True)
class OrchestrationResult:
    candidates: tuple[OrchestratedCandidate, ...]
    traces: tuple[RouteTrace, ...]
    fingerprint: str


def orchestrate_retrieval(
    query: str,
    routes: Sequence[RetrievalRoute],
    backends: Mapping[str, RetrievalRouteBackend],
    *,
    rrf_k: int = 60,
    score_weight: float = 0.35,
    rank_weight: float = 0.65,
    top_k: int = 100,
) -> OrchestrationResult:
    selected_query = _text(query, "query", 20_000)
    if not 1 <= len(routes) <= _MAX_ROUTES:
        raise ValueError("routes are empty or exceed the route limit")
    if len({route.route_id for route in routes}) != len(routes):
        raise ValueError("route IDs must be unique")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or not 1 <= rrf_k <= 10_000:
        raise ValueError("rrf_k is invalid")
    score_w, rank_w = _weight(score_weight, "score_weight"), _weight(rank_weight, "rank_weight")
    if score_w + rank_w <= 0:
        raise ValueError("fusion weights must have positive total")
    if not 1 <= top_k <= 1000:
        raise ValueError("top_k is invalid")

    traces: list[RouteTrace] = []
    route_results: dict[str, tuple[RetrievalCandidate, ...]] = {}
    route_by_id = {route.route_id: route for route in routes}
    for route in routes:
        backend = backends.get(route.backend_id)
        if backend is None:
            traces.append(RouteTrace(route.route_id, False, 0, "backend_unavailable"))
            continue
        try:
            values = tuple(item for item in backend.retrieve(selected_query, route=route) if isinstance(item, RetrievalCandidate))[: route.top_k]
        except Exception:
            traces.append(RouteTrace(route.route_id, False, 0, "backend_failure"))
            continue
        route_results[route.route_id] = values
        traces.append(RouteTrace(route.route_id, True, len(values)))

    aggregate: dict[str, dict[str, Any]] = {}
    for route_id, values in route_results.items():
        route = route_by_id[route_id]
        # Normalize raw dense_score only within this route; RRF supplies rank invariance.
        max_score = max((candidate.dense_score for candidate in values), default=0.0)
        for rank, candidate in enumerate(values, start=1):
            row = aggregate.setdefault(candidate.candidate_id, {"candidate": candidate, "route_scores": {}, "route_ranks": {}, "fused": 0.0})
            # Keep the first materialized candidate only if source/text identity remains stable.
            current = row["candidate"]
            if current.source_id != candidate.source_id or current.text != candidate.text:
                # Same candidate ID with inconsistent materialization is unsafe to fuse.
                continue
            normalized_score = candidate.dense_score / max_score if max_score > 0 else 0.0
            rank_component = 1.0 / (rrf_k + rank)
            route_total = route.weight * ((score_w * normalized_score) + (rank_w * rank_component)) / (score_w + rank_w)
            row["route_scores"][route_id] = normalized_score
            row["route_ranks"][route_id] = rank
            row["fused"] += route_total

    output: list[OrchestratedCandidate] = []
    for row in aggregate.values():
        output.append(
            OrchestratedCandidate(
                candidate=row["candidate"],
                fused_score=max(0.0, float(row["fused"])),
                route_scores=dict(sorted(row["route_scores"].items())),
                route_ranks=dict(sorted(row["route_ranks"].items())),
                route_ids=tuple(sorted(row["route_scores"])),
            )
        )
    output.sort(key=lambda item: (-item.fused_score, item.candidate.candidate_id))
    output = output[:top_k]
    payload = {
        "query_sha256": hashlib.sha256(selected_query.encode("utf-8")).hexdigest(),
        "routes": [asdict(route) for route in routes],
        "traces": [asdict(trace) for trace in traces],
        "results": [(item.candidate.candidate_id, item.fused_score, item.route_ids) for item in output],
    }
    return OrchestrationResult(tuple(output), tuple(traces), hashlib.sha256(_canonical(payload)).hexdigest())


__all__ = [
    "OrchestratedCandidate", "OrchestrationResult", "RetrievalRoute", "RetrievalRouteBackend",
    "RouteTrace", "orchestrate_retrieval",
]
