"""Public adaptive multi-hop uploaded-document retrieval tool."""

from __future__ import annotations

import itertools
import math
import operator
import re
from collections import Counter
from dataclasses import asdict
from typing import Any

from tools.adaptive_rag_tool import search_uploaded_docs_adaptive
from tools.multihop_retrieval import HopEvidence, MultiHopResult, run_multihop_retrieval
from tools.query_decomposition import Subquestion, build_decomposition_plan

_MAX_QUERY_CHARS = 10_000
_MAX_PROPAGATED_TERMS = 16
_MAX_PROPAGATED_TERM_CHARS = 80
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "because",
    "before",
    "between",
    "could",
    "evidence",
    "find",
    "from",
    "have",
    "into",
    "more",
    "most",
    "question",
    "relevant",
    "should",
    "system",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

MULTIHOP_RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs_multihop",
        "description": (
            "Decompose a complex uploaded-document question into a bounded dependency "
            "graph, execute independent hops in parallel, execute dependent hops in "
            "order, and preserve citation lineage for every hop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 10_000},
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "max_subquestions": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 12,
                    "default": 8,
                },
                "per_hop_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
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
                "max_attempts": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 3,
                },
                "max_estimated_cost": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5000,
                    "default": 240,
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


def _positive_float(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 < parsed <= maximum:
        raise ValueError(f"{label} must be greater than zero and at most {maximum}.")
    return parsed


def _propagated_terms(
    question: Subquestion,
    dependencies: tuple[HopEvidence, ...],
) -> tuple[str, ...]:
    """Derive bounded lexical hints without turning evidence prose into a citation."""

    original = {token.lower() for token in _TOKEN_RE.findall(question.text)}
    counts: Counter[str] = Counter()
    for item in dependencies[:100]:
        for token in itertools.islice(_TOKEN_RE.findall(item.text), 500):
            lowered = token.lower()
            if (
                len(lowered) < 3
                or len(lowered) > _MAX_PROPAGATED_TERM_CHARS
                or lowered in original
                or lowered in _STOPWORDS
            ):
                continue
            counts[lowered] += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return tuple(term for term, _count in ranked[:_MAX_PROPAGATED_TERMS])


def _hop_query(question: Subquestion, dependencies: tuple[HopEvidence, ...]) -> str:
    parts = [question.text]
    constraints = tuple(dict.fromkeys((*question.entities, *question.temporal_constraints)))
    if constraints:
        parts.append("Constraints: " + ", ".join(constraints))
    propagated = _propagated_terms(question, dependencies)
    if propagated:
        parts.append("Dependency-derived search terms: " + ", ".join(propagated))
    return "\n".join(parts)[:_MAX_QUERY_CHARS]


def search_uploaded_docs_multihop(
    query: str,
    *,
    owner_id: str = "default_user",
    doc_id: str | None = None,
    max_subquestions: int = 8,
    per_hop_limit: int = 8,
    max_workers: int = 4,
    hop_timeout_seconds: float = 30.0,
    max_attempts: int = 3,
    max_estimated_cost: int = 240,
    agent_client: Any = None,
    expansion_model: str = "gpt-4o-mini",
    diversity_lambda: float = 0.82,
) -> MultiHopResult:
    """Execute adaptive uploaded-document retrieval over a bounded decomposition DAG."""

    question_limit = _integer(max_subquestions, "max_subquestions", 1, 12)
    result_limit = _integer(per_hop_limit, "per_hop_limit", 1, 50)
    workers = _integer(max_workers, "max_workers", 1, 16)
    timeout = _positive_float(hop_timeout_seconds, "hop_timeout_seconds", 600.0)
    attempts = _integer(max_attempts, "max_attempts", 1, 6)
    estimated_cost = _integer(max_estimated_cost, "max_estimated_cost", 1, 5000)
    plan = build_decomposition_plan(query, max_subquestions=question_limit)

    def retrieve(
        question: Subquestion,
        dependencies: tuple[HopEvidence, ...],
    ) -> tuple[Any, ...]:
        adaptive = search_uploaded_docs_adaptive(
            _hop_query(question, dependencies),
            owner_id=owner_id,
            doc_id=doc_id,
            top_k=result_limit,
            max_attempts=attempts,
            max_estimated_cost=estimated_cost,
            agent_client=agent_client,
            expansion_model=expansion_model,
            diversity_lambda=diversity_lambda,
        )
        return adaptive.evidence

    return run_multihop_retrieval(
        plan,
        search=retrieve,
        max_workers=workers,
        per_hop_limit=result_limit,
        hop_timeout_seconds=timeout,
        require_dependency_evidence=True,
    )


def _citation_payload(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        try:
            payload = value.model_dump(mode="json", exclude_none=True)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    if isinstance(value, dict):
        return dict(value)
    return None


def multihop_result_payload(result: MultiHopResult) -> dict[str, Any]:
    """Serialize bounded traces while keeping citations and lineage separate."""

    if not isinstance(result, MultiHopResult):
        raise ValueError("result must be a MultiHopResult.")
    evidence: list[dict[str, Any]] = []
    for item in result.evidence:
        citation = _citation_payload(item.raw)
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
        "plan_fingerprint": result.plan_fingerprint,
        "evidence": evidence,
        "traces": [asdict(trace) for trace in result.traces],
        "joins": [asdict(join) for join in result.joins],
        "terminal_questions": list(result.terminal_questions),
        "terminal_evidence_count": result.terminal_evidence_count,
        "exhausted": result.exhausted,
        "abstain": result.abstain,
    }


__all__ = [
    "MULTIHOP_RAG_SEARCH_TOOL_DEF",
    "multihop_result_payload",
    "search_uploaded_docs_multihop",
]
