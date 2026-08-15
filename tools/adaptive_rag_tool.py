"""Public adaptive uploaded-document retrieval API with bounded trace payloads."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any, Optional

from tools.adaptive_policy_runtime import RetrievalPolicyProvider
from tools.adaptive_retrieval_runner import (
    AdaptiveRetrievalResult,
    run_adaptive_retrieval,
)
from tools.adaptive_trace_runtime import get_adaptive_trace_store
from tools.adaptive_trace_store import AdaptiveTraceStore
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
                "query": {"type": "string", "minLength": 1, "maxLength": 10_000},
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 200},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "max_attempts": {"type": "integer", "minimum": 1, "maximum": 6, "default": 4},
                "max_estimated_cost": {
                    "type": "integer", "minimum": 1, "maximum": 5000, "default": 300
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

_MAX_CITATIONS = 100
_MAX_COLLECTION_ITEMS = 1_000
_MAX_PAYLOAD_STRING_CHARS = 100_000
_PRIVATE_KEYS = {
    "api_key", "authorization", "cookie", "file_path", "local_path", "password",
    "secret", "source_path", "storage_path", "token",
}


def _private_key(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered in _PRIVATE_KEYS
        or lowered.endswith("_path")
        or any(marker in lowered for marker in ("authorization", "cookie", "password", "secret", "token"))
    )


def _payload_key(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("payload keys must be strings.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > 200
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("payload key is invalid.")
    return rendered


def _payload_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        raise ValueError("payload nesting exceeds the limit.")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("payload contains a non-finite number.")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_PAYLOAD_STRING_CHARS or any(
            (ord(character) < 32 and character not in "\t\r\n")
            or ord(character) == 127
            for character in value
        ):
            raise ValueError("payload contains invalid or oversized text.")
        return value
    if type(value) is dict:
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("payload mapping exceeds the item limit.")
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _payload_key(raw_key)
            if _private_key(key):
                continue
            result[key] = _payload_value(raw_value, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("payload collection exceeds the item limit.")
        return [_payload_value(item, depth=depth + 1) for item in value]
    raise ValueError("payload contains an unsupported value.")


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
    trace_store: AdaptiveTraceStore | None = None,
    trace_run_id: str | None = None,
    policy_provider: RetrievalPolicyProvider | None = None,
    domain_registry: Any = None,
) -> AdaptiveRetrievalResult:
    if trace_store is not None and not isinstance(trace_store, AdaptiveTraceStore):
        raise ValueError("trace_store must be an AdaptiveTraceStore or null.")
    selected_trace_store = trace_store if trace_store is not None else get_adaptive_trace_store()
    started_at = time.time()
    result = run_adaptive_retrieval(
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
        policy_provider=policy_provider,
        domain_registry=domain_registry,
    )
    if selected_trace_store is not None:
        selected_trace_store.record_result(
            query=query,
            owner_id=owner_id,
            result=result,
            run_id=trace_run_id,
            started_at=started_at,
            completed_at=time.time(),
        )
    return result


def adaptive_result_payload(result: AdaptiveRetrievalResult) -> dict[str, Any]:
    if not isinstance(result, AdaptiveRetrievalResult):
        raise ValueError("result must be an AdaptiveRetrievalResult.")
    citations: list[dict[str, Any]] = []
    for value in result.evidence[:_MAX_CITATIONS]:
        candidate: Any = None
        try:
            model_dump = getattr(value, "model_dump", None)
        except Exception:
            model_dump = None
        if callable(model_dump):
            try:
                candidate = model_dump(mode="json", exclude_none=True)
            except Exception:
                candidate = None
        elif type(value) is dict:
            candidate = value
        if type(candidate) is not dict:
            continue
        try:
            rendered = _payload_value(candidate)
        except ValueError:
            continue
        if isinstance(rendered, dict):
            citations.append(rendered)
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
        "policy": {
            "policy_id": result.policy_id,
            "version": result.policy_version,
            "fallback_used": result.policy_fallback_used,
            "feature_sha256": result.policy_feature_sha256,
            "decision_sha256": result.policy_decision_sha256,
        },
    }


__all__ = [
    "ADAPTIVE_RAG_SEARCH_TOOL_DEF",
    "adaptive_result_payload",
    "search_uploaded_docs_adaptive",
]
