"""Install bounded adaptive, multi-hop, and heterogeneous retrieval strategies.

The legacy agent exposes one authoritative uploaded-document tool. This module
extends that existing schema and callable after the classic implementation is
fully defined. All strategies still return validated ``Citation`` objects, so
agent evidence registration, relabeling, API serialization, and browser
rendering continue through the same server-owned citation path.
"""

from __future__ import annotations

import copy
import itertools
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any, List, Optional

from tools.models import Citation

_STRATEGIES = {"single", "adaptive", "multihop", "heterogeneous"}
_RETRIEVAL_MODES = {
    "dense",
    "lexical",
    "hybrid",
    "corpus-sparse",
    "corpus-hybrid",
}
_RERANKERS = {"none", "heuristic", "cross-encoder"}
_MAX_CITATIONS = 50


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if not rendered or len(rendered) > maximum or _contains_control(rendered):
        raise ValueError(f"{label} must contain 1-{maximum} valid characters.")
    return rendered


def _optional_text(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum)


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


def _choice(value: Any, label: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}.")
    return value


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
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


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, min(parsed, 1.0))


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _bounded_count(value: Any, maximum: int) -> int:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return 0
    try:
        return len(list(itertools.islice(iter(value), maximum)))
    except Exception:
        return 0


def _copy_citation(
    value: Any,
    *,
    strategy: str,
    metadata: Mapping[str, Any],
) -> Citation | None:
    if not isinstance(value, Citation):
        return None
    try:
        citation = value.model_copy(deep=True)
        citation.metadata = {
            **citation.metadata,
            "retrieval_strategy": strategy,
            **dict(metadata),
        }
        return citation
    except Exception:
        return None


def _adaptive(
    query: str,
    *,
    owner_id: str,
    doc_id: str | None,
    n_results: int,
    max_attempts: int,
    max_estimated_cost: int,
    agent_client: Any,
    expansion_model: str,
    diversity_lambda: float,
) -> list[Citation]:
    from tools.adaptive_rag_tool import search_uploaded_docs_adaptive

    result = search_uploaded_docs_adaptive(
        query,
        owner_id=owner_id,
        doc_id=doc_id,
        top_k=n_results,
        max_attempts=max_attempts,
        max_estimated_cost=max_estimated_cost,
        agent_client=agent_client,
        expansion_model=expansion_model,
        diversity_lambda=diversity_lambda,
    )
    if bool(_safe_attr(result, "abstain", True)):
        return []
    metadata = {
        "adaptive_exhausted": bool(_safe_attr(result, "exhausted", False)),
        "adaptive_estimated_cost": _integer(
            _safe_attr(result, "estimated_cost", 0),
            "adaptive estimated cost",
            0,
            1_000_000,
        ),
        "adaptive_attempt_count": _bounded_count(
            _safe_attr(result, "traces", ()), 100
        ),
    }
    evidence = _safe_attr(result, "evidence", ())
    if isinstance(evidence, (str, bytes, bytearray)):
        return []
    citations: list[Citation] = []
    try:
        rows = itertools.islice(iter(evidence), n_results)
    except Exception:
        return []
    for value in rows:
        citation = _copy_citation(value, strategy="adaptive", metadata=metadata)
        if citation is not None:
            citations.append(citation)
    return citations


def _multihop(
    query: str,
    *,
    owner_id: str,
    doc_id: str | None,
    n_results: int,
    max_attempts: int,
    max_estimated_cost: int,
    max_subquestions: int,
    max_workers: int,
    hop_timeout_seconds: float,
    global_timeout_seconds: float,
    use_model_decomposition: bool,
    decomposition_model: str,
    max_total_estimated_cost: int,
    agent_client: Any,
    expansion_model: str,
    diversity_lambda: float,
) -> list[Citation]:
    from tools.multihop_rag_tool import search_uploaded_docs_multihop

    result = search_uploaded_docs_multihop(
        query,
        owner_id=owner_id,
        doc_id=doc_id,
        max_subquestions=max_subquestions,
        per_hop_limit=n_results,
        max_workers=max_workers,
        hop_timeout_seconds=hop_timeout_seconds,
        global_timeout_seconds=global_timeout_seconds,
        use_model_decomposition=use_model_decomposition,
        decomposition_model=decomposition_model,
        max_attempts=max_attempts,
        max_total_estimated_cost=max_total_estimated_cost,
        max_estimated_cost=max_estimated_cost,
        agent_client=agent_client,
        expansion_model=expansion_model,
        diversity_lambda=diversity_lambda,
    )
    if bool(_safe_attr(result, "abstain", True)):
        return []
    retrieval = _safe_attr(result, "retrieval")
    decomposition = _safe_attr(result, "decomposition")
    budget = _safe_attr(result, "budget")
    fingerprint = _safe_attr(
        retrieval,
        "plan_fingerprint",
        _safe_attr(_safe_attr(decomposition, "plan"), "fingerprint", ""),
    )
    common = {
        "multihop_plan_fingerprint": (
            fingerprint[:64] if isinstance(fingerprint, str) else ""
        ),
        "multihop_used_model": bool(_safe_attr(decomposition, "used_model", False)),
        "multihop_plan_quality": round(
            _score(_safe_attr(_safe_attr(decomposition, "quality"), "score", 0.0)),
            6,
        ),
        "multihop_terminal_evidence_count": _integer(
            _safe_attr(result, "terminal_evidence_count", 0),
            "terminal evidence count",
            0,
            10_000,
        ),
        "multihop_budget_limit": _integer(
            _safe_attr(budget, "total_limit", 0),
            "multi-hop budget limit",
            0,
            100_000,
        ),
        "multihop_allocated_budget": _integer(
            _safe_attr(budget, "allocated_cost", 0),
            "multi-hop allocated budget",
            0,
            100_000,
        ),
    }
    evidence = _safe_attr(result, "evidence", ())
    if isinstance(evidence, (str, bytes, bytearray)):
        return []
    citations: list[Citation] = []
    seen: set[tuple[str, str, str, str]] = set()
    try:
        rows = itertools.islice(iter(evidence), _MAX_CITATIONS)
    except Exception:
        return []
    for item in rows:
        hop_id = str(_safe_attr(item, "hop_id", ""))[:64]
        citation = _copy_citation(
            _safe_attr(item, "raw"),
            strategy="multihop",
            metadata={
                **common,
                "multihop_evidence_id": str(
                    _safe_attr(item, "evidence_id", "")
                )[:800],
                "multihop_hop_id": hop_id,
                "multihop_source_id": str(
                    _safe_attr(item, "source_id", "")
                )[:500],
                "multihop_score": round(
                    _score(_safe_attr(item, "score", 0.0)), 6
                ),
            },
        )
        if citation is None:
            continue
        identity = (
            citation.source_id or citation.url,
            citation.doc_id or "",
            citation.quote or citation.snippet or "",
            hop_id,
        )
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(citation)
    return citations


def _heterogeneous(
    query: str,
    *,
    owner_id: str,
    doc_id: str | None,
    n_results: int,
    max_subquestions: int,
    max_workers: int,
    hop_timeout_seconds: float,
    global_timeout_seconds: float,
    scope: str,
    domain: str,
    allowed_domains: list[str] | None,
    year_from: int | None,
    year_to: int | None,
    total_cost_limit: int,
    total_latency_limit_ms: int,
    total_monetary_limit_microunits: int,
    agent_client: Any,
    expansion_model: str,
    diversity_lambda: float,
) -> list[Citation]:
    from tools.heterogeneous_rag_tool import search_research_heterogeneous

    if n_results > 10:
        raise ValueError("n_results may be at most 10 for heterogeneous retrieval.")
    result = search_research_heterogeneous(
        query,
        owner_id=owner_id,
        doc_id=doc_id,
        scope=scope,
        domain=domain,
        allowed_domains=allowed_domains,
        year_from=year_from,
        year_to=year_to,
        max_subquestions=max_subquestions,
        top_k=n_results,
        total_cost_limit=total_cost_limit,
        total_latency_limit_ms=total_latency_limit_ms,
        total_monetary_limit_microunits=total_monetary_limit_microunits,
        max_workers=max_workers,
        hop_timeout_seconds=hop_timeout_seconds,
        global_timeout_seconds=global_timeout_seconds,
        agent_client=agent_client,
        expansion_model=expansion_model,
        diversity_lambda=diversity_lambda,
    )
    retrieval = _safe_attr(result, "retrieval")
    if bool(_safe_attr(retrieval, "abstain", True)):
        return []
    routes = _safe_attr(result, "routes_by_hop", ())
    route_by_hop: dict[str, str] = {}
    if not isinstance(routes, (str, bytes, bytearray)):
        try:
            for item in itertools.islice(iter(routes), 100):
                if (
                    isinstance(item, (tuple, list))
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                ):
                    route_by_hop[item[0][:64]] = item[1][:64]
        except Exception:
            route_by_hop = {}
    budget = _safe_attr(result, "budget")
    common = {
        "heterogeneous_allocated_cost_units": _integer(
            _safe_attr(budget, "allocated_cost_units", 0),
            "heterogeneous allocated cost",
            0,
            1_000_000,
        ),
        "heterogeneous_allocated_latency_ms": _integer(
            _safe_attr(budget, "allocated_latency_ms", 0),
            "heterogeneous allocated latency",
            0,
            86_400_000,
        ),
        "heterogeneous_allocated_monetary_microunits": _integer(
            _safe_attr(budget, "allocated_monetary_microunits", 0),
            "heterogeneous allocated money",
            0,
            1_000_000_000,
        ),
    }
    evidence = _safe_attr(retrieval, "evidence", ())
    if isinstance(evidence, (str, bytes, bytearray)):
        return []
    citations: list[Citation] = []
    seen: set[tuple[str, str, str, str]] = set()
    try:
        rows = itertools.islice(iter(evidence), _MAX_CITATIONS)
    except Exception:
        return []
    for item in rows:
        hop_id = str(_safe_attr(item, "hop_id", ""))[:64]
        citation = _copy_citation(
            _safe_attr(item, "raw"),
            strategy="heterogeneous",
            metadata={
                **common,
                "heterogeneous_hop_id": hop_id,
                "heterogeneous_route": route_by_hop.get(hop_id, "unknown"),
                "heterogeneous_evidence_id": str(
                    _safe_attr(item, "evidence_id", "")
                )[:800],
                "heterogeneous_score": round(
                    _score(_safe_attr(item, "score", 0.0)), 6
                ),
            },
        )
        if citation is None:
            continue
        identity = (
            citation.source_id or citation.url,
            citation.doc_id or "",
            citation.quote or citation.snippet or "",
            hop_id,
        )
        if identity in seen:
            continue
        seen.add(identity)
        citations.append(citation)
    return citations


def _schema(base_schema: Mapping[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(dict(base_schema))
    function = schema["function"]
    function["description"] = (
        "Search authenticated uploaded documents through classic, adaptive, or "
        "provenance-preserving multi-hop retrieval. The heterogeneous strategy may "
        "also route across bounded public-web and scholarly providers. Use single for "
        "one known route, adaptive when correction or abstention may be needed, "
        "multihop for uploaded-document dependency questions, and heterogeneous for "
        "mixed uploaded/public/scholarly evidence."
    )
    properties = function["parameters"]["properties"]
    properties.update(
        {
            "strategy": {
                "type": "string",
                "enum": sorted(_STRATEGIES),
                "default": "single",
            },
            "n_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
            },
            "max_attempts": {
                "type": "integer",
                "minimum": 1,
                "maximum": 6,
                "default": 4,
            },
            "max_estimated_cost": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10_000,
                "default": 500,
            },
            "max_subquestions": {
                "type": "integer",
                "minimum": 1,
                "maximum": 12,
                "default": 8,
            },
            "max_workers": {
                "type": "integer",
                "minimum": 1,
                "maximum": 16,
                "default": 4,
            },
            "hop_timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 600,
                "default": 30,
            },
            "global_timeout_seconds": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 3_600,
                "default": 120,
            },
            "use_model_decomposition": {
                "type": "boolean",
                "default": False,
            },
            "decomposition_model": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "default": "gpt-4o-mini",
            },
            "max_total_estimated_cost": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100_000,
                "default": 1_200,
            },
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
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 253,
                },
                "maxItems": 50,
            },
            "year_from": {"type": "integer", "minimum": 0, "maximum": 9_999},
            "year_to": {"type": "integer", "minimum": 0, "maximum": 9_999},
            "total_cost_limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000_000,
                "default": 2_000,
            },
            "total_latency_limit_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": 86_400_000,
                "default": 60_000,
            },
            "total_monetary_limit_microunits": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1_000_000_000,
                "default": 100_000,
            },
        }
    )
    return schema


def install_rag_strategies(
    base_schema: Mapping[str, Any],
    base_search: Callable[..., List[Citation]],
) -> tuple[dict[str, Any], Callable[..., List[Citation]]]:
    """Return an extended schema and backward-compatible authoritative wrapper."""

    if not callable(base_search):
        raise ValueError("base_search must be callable.")
    schema = _schema(base_schema)

    def search_uploaded_docs(
        query: str,
        *,
        owner_id: str = "default_user",
        doc_id: Optional[str] = None,
        strategy: str = "single",
        use_hyde: bool = False,
        use_multi_query: bool = False,
        agent_client: Optional[Any] = None,
        expansion_model: str = "gpt-4o-mini",
        n_results: int = 5,
        retrieval_mode: str = "dense",
        reranker: str = "none",
        candidate_pool: int = 20,
        diversity_lambda: float = 0.82,
        max_attempts: int = 4,
        max_estimated_cost: int = 500,
        max_subquestions: int = 8,
        max_workers: int = 4,
        hop_timeout_seconds: float = 30.0,
        global_timeout_seconds: float = 120.0,
        use_model_decomposition: bool = False,
        decomposition_model: str = "gpt-4o-mini",
        max_total_estimated_cost: int = 1_200,
        scope: str = "mixed",
        domain: str = "general",
        allowed_domains: Optional[List[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        total_cost_limit: int = 2_000,
        total_latency_limit_ms: int = 60_000,
        total_monetary_limit_microunits: int = 100_000,
    ) -> List[Citation]:
        selected = _choice(strategy, "strategy", _STRATEGIES)
        results = _integer(n_results, "n_results", 1, _MAX_CITATIONS)
        attempts = _integer(max_attempts, "max_attempts", 1, 6)
        per_hop_cost = _integer(
            max_estimated_cost, "max_estimated_cost", 1, 10_000
        )
        questions = _integer(max_subquestions, "max_subquestions", 1, 12)
        workers = _integer(max_workers, "max_workers", 1, 16)
        hop_timeout = _positive(hop_timeout_seconds, "hop_timeout_seconds", 600.0)
        global_timeout = _positive(
            global_timeout_seconds, "global_timeout_seconds", 3_600.0
        )
        total_estimated = _integer(
            max_total_estimated_cost,
            "max_total_estimated_cost",
            1,
            100_000,
        )
        diversity = _unit(diversity_lambda, "diversity_lambda")
        model = _text(expansion_model, "expansion_model", 200)
        decomposition = _text(decomposition_model, "decomposition_model", 200)
        document_id = _optional_text(doc_id, "doc_id", 200)
        selected_scope = _choice(scope, "scope", {"uploaded", "public", "mixed"})
        selected_domain = _choice(domain, "domain", {"general", "scholarly"})
        if allowed_domains is not None:
            if not isinstance(allowed_domains, list) or len(allowed_domains) > 50:
                raise ValueError(
                    "allowed_domains must be a list with at most 50 values."
                )
            domains = [_text(value, "allowed domain", 253) for value in allowed_domains]
        else:
            domains = None
        start_year = (
            _integer(year_from, "year_from", 0, 9_999)
            if year_from is not None
            else None
        )
        end_year = (
            _integer(year_to, "year_to", 0, 9_999)
            if year_to is not None
            else None
        )
        if start_year is not None and end_year is not None and start_year > end_year:
            raise ValueError("year_from may not exceed year_to.")
        heterogeneous_cost = _integer(
            total_cost_limit, "total_cost_limit", 1, 1_000_000
        )
        heterogeneous_latency = _integer(
            total_latency_limit_ms,
            "total_latency_limit_ms",
            1,
            86_400_000,
        )
        heterogeneous_money = _integer(
            total_monetary_limit_microunits,
            "total_monetary_limit_microunits",
            0,
            1_000_000_000,
        )
        if selected == "single":
            return base_search(
                query,
                owner_id=owner_id,
                doc_id=document_id,
                use_hyde=use_hyde,
                use_multi_query=use_multi_query,
                agent_client=agent_client,
                expansion_model=model,
                n_results=results,
                retrieval_mode=retrieval_mode,
                reranker=reranker,
                candidate_pool=candidate_pool,
                diversity_lambda=diversity,
            )
        if (
            use_hyde
            or use_multi_query
            or retrieval_mode != "dense"
            or reranker != "none"
            or candidate_pool != 20
        ):
            raise ValueError(
                "Classic retrieval controls may only be changed for single retrieval."
            )
        if selected == "adaptive":
            return _adaptive(
                query,
                owner_id=owner_id,
                doc_id=document_id,
                n_results=results,
                max_attempts=attempts,
                max_estimated_cost=per_hop_cost,
                agent_client=agent_client,
                expansion_model=model,
                diversity_lambda=diversity,
            )
        if selected == "multihop":
            return _multihop(
                query,
                owner_id=owner_id,
                doc_id=document_id,
                n_results=results,
                max_attempts=attempts,
                max_estimated_cost=per_hop_cost,
                max_subquestions=questions,
                max_workers=workers,
                hop_timeout_seconds=hop_timeout,
                global_timeout_seconds=global_timeout,
                use_model_decomposition=use_model_decomposition,
                decomposition_model=decomposition,
                max_total_estimated_cost=total_estimated,
                agent_client=agent_client,
                expansion_model=model,
                diversity_lambda=diversity,
            )
        if use_model_decomposition:
            raise ValueError(
                "use_model_decomposition is not supported for heterogeneous retrieval."
            )
        return _heterogeneous(
            query,
            owner_id=owner_id,
            doc_id=document_id,
            n_results=results,
            max_subquestions=questions,
            max_workers=workers,
            hop_timeout_seconds=hop_timeout,
            global_timeout_seconds=global_timeout,
            scope=selected_scope,
            domain=selected_domain,
            allowed_domains=domains,
            year_from=start_year,
            year_to=end_year,
            total_cost_limit=heterogeneous_cost,
            total_latency_limit_ms=heterogeneous_latency,
            total_monetary_limit_microunits=heterogeneous_money,
            agent_client=agent_client,
            expansion_model=model,
            diversity_lambda=diversity,
        )

    search_uploaded_docs.__name__ = "search_uploaded_docs"
    search_uploaded_docs.__qualname__ = "search_uploaded_docs"
    search_uploaded_docs.__doc__ = (
        "Search through single, adaptive, uploaded multi-hop, or heterogeneous "
        "strategies while returning only authoritative Citation objects."
    )
    return schema, search_uploaded_docs


__all__ = ["install_rag_strategies"]
