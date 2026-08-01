"""Public heterogeneous multi-hop research tool."""

from __future__ import annotations

import math
import operator
from dataclasses import asdict
from typing import Any, Iterable, Mapping

from tools.heterogeneous_adapters import build_production_route_adapters
from tools.heterogeneous_multihop import (
    HeterogeneousMultiHopResult,
    run_heterogeneous_multihop,
)
from tools.public_payload import public_model_payload
from tools.query_decomposition import build_decomposition_plan

HETEROGENEOUS_RAG_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_research_heterogeneous",
        "description": (
            "Route a bounded decomposition graph across uploaded dense/sparse/hybrid, "
            "public web, and scholarly retrieval under global cost, latency, money, "
            "evidence, and deadline ceilings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 10_000},
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "scope": {
                    "type": "string",
                    "enum": ["uploaded", "public", "mixed"],
                    "default": "mixed",
                },
                "domain": {
                    "type": "string",
                    "enum": ["general", "scholarly"],
                    "default": "general",
                },
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 253},
                    "maxItems": 50,
                },
                "year_from": {"type": "integer", "minimum": 0, "maximum": 9999},
                "year_to": {"type": "integer", "minimum": 0, "maximum": 9999},
                "max_subquestions": {
                    "type": "integer", "minimum": 1, "maximum": 12, "default": 8
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 8},
                "total_cost_limit": {
                    "type": "integer", "minimum": 1, "maximum": 1000000, "default": 2000
                },
                "total_latency_limit_ms": {
                    "type": "integer", "minimum": 1, "maximum": 86400000, "default": 60000
                },
                "total_monetary_limit_microunits": {
                    "type": "integer", "minimum": 0, "maximum": 1000000000, "default": 100000
                },
                "hop_timeout_seconds": {
                    "type": "number", "exclusiveMinimum": 0, "maximum": 600, "default": 30
                },
                "global_timeout_seconds": {
                    "type": "number", "exclusiveMinimum": 0, "maximum": 3600, "default": 120
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


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


def _positive(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed <= maximum:
        raise ValueError(f"{label} must be greater than zero and at most {maximum}.")
    return parsed


def search_research_heterogeneous(
    query: str,
    *,
    owner_id: str = "default_user",
    doc_id: str | None = None,
    scope: str = "mixed",
    domain: str = "general",
    allowed_domains: Iterable[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    max_subquestions: int = 8,
    top_k: int = 8,
    total_cost_limit: int = 2_000,
    total_latency_limit_ms: int = 60_000,
    total_monetary_limit_microunits: int = 100_000,
    max_workers: int = 4,
    hop_timeout_seconds: float = 30.0,
    global_timeout_seconds: float = 120.0,
    agent_client: Any = None,
    expansion_model: str = "gpt-4o-mini",
    diversity_lambda: float = 0.82,
    route_overrides: Mapping[str, str] | None = None,
    _adapters: Mapping[str, Any] | None = None,
) -> HeterogeneousMultiHopResult:
    question_limit = _integer(max_subquestions, "max_subquestions", 1, 12)
    result_limit = _integer(top_k, "top_k", 1, 10)
    workers = _integer(max_workers, "max_workers", 1, 16)
    hop_timeout = _positive(hop_timeout_seconds, "hop_timeout_seconds", 600.0)
    global_timeout = _positive(global_timeout_seconds, "global_timeout_seconds", 3600.0)
    plan = build_decomposition_plan(query, max_subquestions=question_limit)
    adapters = (
        _adapters
        if _adapters is not None
        else build_production_route_adapters(
            owner_id=owner_id,
            doc_id=doc_id,
            allowed_domains=allowed_domains,
            year_from=year_from,
            year_to=year_to,
            agent_client=agent_client,
            expansion_model=expansion_model,
            diversity_lambda=diversity_lambda,
        )
    )
    return run_heterogeneous_multihop(
        plan,
        adapters=adapters,
        scope=scope,
        domain=domain,
        route_overrides=route_overrides,
        top_k=result_limit,
        total_cost_limit=total_cost_limit,
        total_latency_limit_ms=total_latency_limit_ms,
        total_monetary_limit_microunits=total_monetary_limit_microunits,
        max_workers=workers,
        hop_timeout_seconds=hop_timeout,
        global_timeout_seconds=global_timeout,
    )


def heterogeneous_result_payload(result: HeterogeneousMultiHopResult) -> dict[str, Any]:
    if not isinstance(result, HeterogeneousMultiHopResult):
        raise ValueError("result must be a HeterogeneousMultiHopResult.")
    evidence: list[dict[str, Any]] = []
    for item in result.retrieval.evidence:
        citation = public_model_payload(item.raw)
        if citation is None:
            continue
        evidence.append(
            {
                "citation": citation,
                "lineage": {
                    "evidence_id": item.evidence_id,
                    "hop_id": item.hop_id,
                    "source_id": item.source_id,
                    "doc_id": item.doc_id,
                    "page_number": item.page_number,
                    "score": item.score,
                },
            }
        )
    return {
        "plan_fingerprint": result.retrieval.plan_fingerprint,
        "budget": asdict(result.budget),
        "routes_by_hop": [list(item) for item in result.routes_by_hop],
        "evidence": evidence,
        "traces": [asdict(trace) for trace in result.retrieval.traces],
        "joins": [asdict(join) for join in result.retrieval.joins],
        "terminal_questions": list(result.retrieval.terminal_questions),
        "terminal_evidence_count": result.retrieval.terminal_evidence_count,
        "exhausted": result.retrieval.exhausted,
        "abstain": result.retrieval.abstain,
    }


__all__ = [
    "HETEROGENEOUS_RAG_TOOL_DEF",
    "heterogeneous_result_payload",
    "search_research_heterogeneous",
]
