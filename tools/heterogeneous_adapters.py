"""Production adapters for uploaded, public-web, and scholarly multi-hop routes."""

from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Callable, Iterable
from types import MappingProxyType
from typing import Any

from tools.heterogeneous_route_types import HeterogeneousRouteRequest

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+\-][A-Za-z0-9]+)*")
_YEAR_RE = re.compile(r"^(?:18|19|20|21)\d{2}$")
_MAX_QUERY_CHARS = 2_000
_MAX_DEPENDENCY_ROWS = 100
_MAX_TERMS = 16
_MAX_TERM_CHARS = 80
_MAX_DOMAINS = 50
_STOPWORDS = {
    "about", "after", "also", "among", "and", "are", "because", "before",
    "between", "could", "evidence", "find", "from", "have", "into", "more",
    "most", "question", "relevant", "should", "system", "that", "their",
    "there", "these", "this", "those", "using", "what", "when", "where",
    "which", "with", "would",
}


def _text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = " ".join(value.split())
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _optional_identifier(value: Any, label: str, maximum: int) -> str | None:
    if value is None:
        return None
    return _text(value, label, maximum)


def _bounded_domains(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("allowed_domains must be an iterable of hostnames.")
    try:
        rows = list(itertools.islice(iter(values), _MAX_DOMAINS + 1))
    except Exception as exc:
        raise ValueError("allowed_domains is not safely iterable.") from exc
    if len(rows) > _MAX_DOMAINS:
        raise ValueError("allowed_domains exceeds the route limit.")
    result: list[str] = []
    for value in rows:
        domain = _text(value, "allowed domain", 253)
        if domain not in result:
            result.append(domain)
    return tuple(result)


def _constraint_query(request: HeterogeneousRouteRequest) -> str:
    question = request.question
    parts = [question.text]
    constraints = tuple(
        dict.fromkeys((*question.entities, *question.temporal_constraints))
    )
    if constraints:
        parts.append("Constraints: " + ", ".join(constraints))
    return "\n".join(parts)[:_MAX_QUERY_CHARS]


def _dependency_terms(request: HeterogeneousRouteRequest) -> tuple[str, ...]:
    original = {token.lower() for token in _TOKEN_RE.findall(request.question.text)}
    counts: Counter[str] = Counter()
    for item in request.dependencies[:_MAX_DEPENDENCY_ROWS]:
        for token in itertools.islice(_TOKEN_RE.findall(item.text), 500):
            lowered = token.lower()
            if (
                len(lowered) < 3
                or len(lowered) > _MAX_TERM_CHARS
                or lowered in original
                or lowered in _STOPWORDS
            ):
                continue
            counts[lowered] += 1
    return tuple(
        term
        for term, _count in sorted(
            counts.items(), key=lambda pair: (-pair[1], pair[0])
        )[:_MAX_TERMS]
    )


def _uploaded_query(request: HeterogeneousRouteRequest) -> str:
    parts = [_constraint_query(request)]
    propagated = _dependency_terms(request)
    if propagated:
        parts.append("Dependency-derived search terms: " + ", ".join(propagated))
    return "\n".join(parts)[:10_000]


def _year_bounds(
    request: HeterogeneousRouteRequest,
    explicit_from: int | None,
    explicit_to: int | None,
) -> tuple[int | None, int | None]:
    if explicit_from is not None or explicit_to is not None:
        return explicit_from, explicit_to
    years = sorted(
        {
            int(value)
            for value in request.question.temporal_constraints
            if _YEAR_RE.fullmatch(value)
        }
    )
    if not years:
        return None, None
    return years[0], years[-1]


def build_production_route_adapters(
    *,
    owner_id: str,
    doc_id: str | None = None,
    allowed_domains: Iterable[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    agent_client: Any = None,
    expansion_model: str = "gpt-4o-mini",
    diversity_lambda: float = 0.82,
    uploaded_search: Callable[..., Any] | None = None,
    web_search_fn: Callable[..., Any] | None = None,
    academic_search_fn: Callable[..., Any] | None = None,
    owner_normalizer: Callable[[str], str] | None = None,
) -> MappingProxyType:
    """Build route adapters while keeping private evidence out of public queries."""

    if owner_normalizer is None:
        from tools.security import normalize_owner_id as owner_normalizer
    owner = owner_normalizer(owner_id)
    document_id = _optional_identifier(doc_id, "doc_id", 200)
    domains = _bounded_domains(allowed_domains)
    model = _text(expansion_model, "expansion_model", 200)
    if isinstance(diversity_lambda, bool):
        raise ValueError("diversity_lambda must be numeric.")
    try:
        diversity = float(diversity_lambda)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("diversity_lambda must be numeric.") from exc
    if not 0.0 <= diversity <= 1.0:
        raise ValueError("diversity_lambda must be between 0 and 1.")

    def uploaded(request: HeterogeneousRouteRequest):
        search = uploaded_search
        if search is None:
            from tools.rag_tool import search_uploaded_docs as search
        reranker = "heuristic" if request.budget.route == "corpus-hybrid" else "none"
        return search(
            _uploaded_query(request),
            owner_id=owner,
            doc_id=document_id,
            use_hyde=False,
            use_multi_query=request.question.relation in {"compare", "synthesize"},
            agent_client=agent_client,
            expansion_model=model,
            n_results=request.budget.max_results,
            retrieval_mode=request.budget.route,
            reranker=reranker,
            candidate_pool=max(request.budget.max_results, 20),
            diversity_lambda=diversity,
        )

    def public_web(request: HeterogeneousRouteRequest):
        search = web_search_fn
        if search is None:
            from tools.web_search import web_search as search
        if request.budget.max_results > 10:
            raise ValueError("web route supports at most 10 results per hop.")
        return search(
            _constraint_query(request),
            None if domains is None else list(domains),
            limit=request.budget.max_results,
        )

    def scholarly(request: HeterogeneousRouteRequest):
        search = academic_search_fn
        if search is None:
            from tools.academic_search import academic_search as search
        if request.budget.max_results > 10:
            raise ValueError("scholarly route supports at most 10 results per hop.")
        selected_from, selected_to = _year_bounds(request, year_from, year_to)
        return search(
            _constraint_query(request),
            selected_from,
            selected_to,
            limit=request.budget.max_results,
        )

    return MappingProxyType(
        {
            "dense": uploaded,
            "corpus-sparse": uploaded,
            "corpus-hybrid": uploaded,
            "web": public_web,
            "scholarly": scholarly,
        }
    )


__all__ = ["build_production_route_adapters"]
