"""Public adaptive uploaded-document retrieval API with bounded trace payloads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from tools.adaptive_retrieval_runner import (
    AdaptiveRetrievalResult,
    run_adaptive_retrieval,
)
from tools.rag_tool import search_uploaded_docs

ADAPTIVE_RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs_adaptive",
        "description": (
            "Search uploaded documents with bounded adaptive routing, correction "
            "attempts, evidence-sufficiency checks, and abstention."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10_000,
                },
                "doc_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "top_k": {
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
                    "maximum": 5000,
                    "default": 300,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def search_uploaded_docs_adaptive(
    query: str,
    *,
    owner_id: str = "default_user",
    doc_id: Optional[str] = None,
    top_k: int = 5,
    max_attempts: int = 4,
    max_estimated_cost: int = 300,
    agent_client: Any = None,
    expansion_model: str = "gpt-4o-mini",
    diversity_lambda: float = 0.82,
) -> AdaptiveRetrievalResult:
    return run_adaptive_retrieval(
        query,
        search=search_uploaded_docs,
        owner_id=owner_id,
        doc_id=doc_id,
        top_k=top_k,
        max_attempts=max_attempts,
        max_estimated_cost=max_estimated_cost,
        agent_client=agent_client,
        expansion_model=expansion_model,
        diversity_lambda=diversity_lambda,
    )


def adaptive_result_payload(result: AdaptiveRetrievalResult) -> dict[str, Any]:
    if not isinstance(result, AdaptiveRetrievalResult):
        raise ValueError("result must be an AdaptiveRetrievalResult.")
    citations: list[dict[str, Any]] = []
    for value in result.evidence:
        if hasattr(value, "model_dump"):
            try:
                citations.append(value.model_dump(mode="json", exclude_none=True))
                continue
            except Exception:
                pass
        if isinstance(value, dict):
            citations.append(dict(value))
    traces = [
        {
            "attempt": asdict(trace.attempt),
            "returned_evidence": trace.returned_evidence,
            "accumulated_evidence": trace.accumulated_evidence,
            "signals": asdict(trace.signals),
            "error_type": trace.error_type,
        }
        for trace in result.traces
    ]
    return {
        "citations": citations,
        "traces": traces,
        "final_signals": asdict(result.final_signals),
        "exhausted": result.exhausted,
        "abstain": result.abstain,
        "estimated_cost": result.estimated_cost,
    }


__all__ = [
    "ADAPTIVE_RAG_SEARCH_TOOL_DEF",
    "adaptive_result_payload",
    "search_uploaded_docs_adaptive",
]
