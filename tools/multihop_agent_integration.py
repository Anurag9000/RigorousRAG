"""Install bounded multi-hop uploaded-document retrieval on the research agent."""

from __future__ import annotations

import copy
import json
from types import ModuleType
from typing import Any

from tools.models import Citation
from tools.multihop_rag_tool import (
    MULTIHOP_RAG_SEARCH_TOOL_DEF,
    MultiHopRAGResult,
    multihop_result_payload,
    search_uploaded_docs_multihop,
)

_TOOL_NAME = "search_uploaded_docs_multihop"
_PROMPT_LINE = (
    "- Multi-hop uploaded-document retrieval (`search_uploaded_docs_multihop`) — "
    "use for questions that require dependency-aware evidence across multiple "
    "retrieval hops; if the tool abstains, treat the chain as unsupported."
)


def _schema_name(value: Any) -> str | None:
    try:
        name = value["function"]["name"]
    except (KeyError, TypeError):
        return None
    return name if isinstance(name, str) else None


def _citation_limit(module: ModuleType) -> int:
    value = getattr(module, "_MAX_EVIDENCE_SOURCES", 100)
    if isinstance(value, bool) or not isinstance(value, int):
        return 100
    return max(1, min(value, 500))


def _citation_key(citation: Citation) -> tuple[str, str, str]:
    return (
        citation.source_id or citation.url,
        citation.doc_id or "",
        citation.quote or citation.snippet or "",
    )


def _authoritative_citations(result: MultiHopRAGResult, *, maximum: int) -> list[Citation]:
    """Return source citations only when the dependency chain reaches evidence."""
    if result.abstain:
        return []
    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in result.evidence:
        raw = item.raw
        if not isinstance(raw, Citation):
            continue
        key = _citation_key(raw)
        if key in seen:
            continue
        seen.add(key)
        citations.append(raw)
        if len(citations) >= maximum:
            break
    return citations


def _agent_payload(result: MultiHopRAGResult, *, citation_count: int) -> dict[str, Any]:
    """Strip local citation labels; the outer agent supplies authoritative labels."""
    payload = multihop_result_payload(result)
    lineage: list[dict[str, Any]] = []
    raw_evidence = payload.get("evidence")
    if isinstance(raw_evidence, list):
        for row in raw_evidence:
            if not isinstance(row, dict):
                continue
            value = row.get("lineage")
            if isinstance(value, dict):
                lineage.append(value)
    payload["evidence"] = lineage
    payload["citation_gate"] = {
        "status": "abstain" if result.abstain else "terminal_evidence_available",
        "authoritative_citation_count": citation_count,
        "instruction": "Use only the server-supplied citation objects outside this result payload.",
    }
    return payload


def install_multihop_agent_tool(module: ModuleType) -> ModuleType:
    """Extend one loaded ``search_agent_legacy`` module idempotently."""
    if not isinstance(module, ModuleType):
        raise ValueError("module must be a loaded module.")
    if getattr(module, "_multihop_agent_tool_installed", False):
        return module
    schemas = getattr(module, "TOOLS_SCHEMA", None)
    parameter_schemas = getattr(module, "_TOOL_PARAMETER_SCHEMAS", None)
    agent_class = getattr(module, "SearchAgent", None)
    original_dispatch = getattr(agent_class, "_dispatch", None)
    if not isinstance(schemas, list):
        raise RuntimeError("search agent TOOLS_SCHEMA is unavailable.")
    if not isinstance(parameter_schemas, dict):
        raise RuntimeError("search agent parameter schema registry is unavailable.")
    if not isinstance(agent_class, type) or not callable(original_dispatch):
        raise RuntimeError("search agent dispatch boundary is unavailable.")

    definition = copy.deepcopy(MULTIHOP_RAG_SEARCH_TOOL_DEF)
    schemas[:] = [value for value in schemas if _schema_name(value) != _TOOL_NAME]
    schemas.append(definition)
    parameter_schemas[_TOOL_NAME] = copy.deepcopy(definition["function"]["parameters"])
    maximum = _citation_limit(module)

    def dispatch(self: Any, tool_name: str, arguments: dict[str, Any]):
        if tool_name != _TOOL_NAME:
            return original_dispatch(self, tool_name, arguments)
        expansion_model = getattr(self, "_expansion_model", None)
        kwargs: dict[str, Any] = {
            "owner_id": getattr(self, "owner_id"),
            "agent_client": getattr(self, "client", None),
            **arguments,
        }
        policy_provider = getattr(self, "adaptive_policy_provider", None)
        if policy_provider is not None:
            kwargs["policy_provider"] = policy_provider
        domain_registry = getattr(self, "domain_registry", None)
        if domain_registry is not None:
            kwargs["domain_registry"] = domain_registry
        if callable(expansion_model):
            kwargs["expansion_model"] = expansion_model()
        result = search_uploaded_docs_multihop(**kwargs)
        citations = _authoritative_citations(result, maximum=maximum)
        content = json.dumps(
            _agent_payload(result, citation_count=len(citations)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return content, citations

    agent_class._dispatch = dispatch
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if isinstance(prompt, str) and _PROMPT_LINE not in prompt:
        module.SYSTEM_PROMPT = prompt.rstrip() + "\n" + _PROMPT_LINE + "\n"
    module._multihop_original_dispatch = original_dispatch
    module._multihop_agent_tool_installed = True
    return module


__all__ = ["install_multihop_agent_tool"]
