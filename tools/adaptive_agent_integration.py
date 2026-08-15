"""Install bounded adaptive uploaded-document retrieval on the research agent.

The integration mirrors the multi-hop citation boundary: tool payloads expose traces and
policy metadata, while only server-owned ``Citation`` objects are returned to the outer
agent evidence registry. An adaptive abstention publishes zero authoritative citations.
"""

from __future__ import annotations

import copy
import json
from types import ModuleType
from typing import Any

from tools.adaptive_rag_tool import (
    ADAPTIVE_RAG_SEARCH_TOOL_DEF,
    AdaptiveRetrievalResult,
    adaptive_result_payload,
    search_uploaded_docs_adaptive,
)
from tools.models import Citation

_TOOL_NAME = "search_uploaded_docs_adaptive"
_PROMPT_LINE = (
    "- Adaptive uploaded-document retrieval (`search_uploaded_docs_adaptive`) — use for "
    "owner-scoped questions where retrieval strategy may need bounded correction; if "
    "the tool abstains, do not treat retrieved passages as supported evidence."
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


def _authoritative_citations(
    result: AdaptiveRetrievalResult,
    *,
    maximum: int,
) -> list[Citation]:
    if result.abstain:
        return []
    output: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()
    for item in result.evidence:
        if not isinstance(item, Citation):
            continue
        key = _citation_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= maximum:
            break
    return output


def _agent_payload(
    result: AdaptiveRetrievalResult,
    *,
    citation_count: int,
) -> dict[str, Any]:
    payload = adaptive_result_payload(result)
    # The outer SearchAgent registry is authoritative for labels and citation objects.
    payload.pop("citations", None)
    payload["citation_gate"] = {
        "status": "abstain" if result.abstain else "evidence_available",
        "authoritative_citation_count": citation_count,
        "instruction": "Use only the server-supplied citation objects outside this payload.",
    }
    return payload


def install_adaptive_agent_tool(module: ModuleType) -> ModuleType:
    if not isinstance(module, ModuleType):
        raise ValueError("module must be a loaded module")
    if getattr(module, "_adaptive_agent_tool_installed", False):
        return module
    schemas = getattr(module, "TOOLS_SCHEMA", None)
    parameter_schemas = getattr(module, "_TOOL_PARAMETER_SCHEMAS", None)
    agent_class = getattr(module, "SearchAgent", None)
    original_dispatch = getattr(agent_class, "_dispatch", None)
    if not isinstance(schemas, list) or not isinstance(parameter_schemas, dict):
        raise RuntimeError("search agent tool schema boundary is unavailable")
    if not isinstance(agent_class, type) or not callable(original_dispatch):
        raise RuntimeError("search agent dispatch boundary is unavailable")

    definition = copy.deepcopy(ADAPTIVE_RAG_SEARCH_TOOL_DEF)
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
            "policy_provider": getattr(self, "adaptive_policy_provider", None),
            "domain_registry": getattr(self, "domain_registry", None),
            **arguments,
        }
        if callable(expansion_model):
            kwargs["expansion_model"] = expansion_model()
        result = search_uploaded_docs_adaptive(**kwargs)
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
    module._adaptive_original_dispatch = original_dispatch
    module._adaptive_agent_tool_installed = True
    return module


__all__ = ["install_adaptive_agent_tool"]
