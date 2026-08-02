"""Install authoritative GraphRAG retrieval into the existing agent surface."""

from __future__ import annotations

import copy
from types import ModuleType
from typing import Any

from tools.evidence_graph_rag_tool import (
    GRAPH_RAG_SEARCH_TOOL_DEF,
    search_evidence_graph,
)

_TOOL_NAME = "search_evidence_graph"
_PROMPT_LINE = (
    "- Reviewed Evidence Graph (`search_evidence_graph`) — use for bounded "
    "cross-document retrieval over explicit, generation-validated relations."
)


def _schema_name(value: Any) -> str | None:
    try:
        name = value["function"]["name"]
    except (KeyError, TypeError):
        return None
    return name if isinstance(name, str) else None


def install_evidence_graph_agent_tool(module: ModuleType) -> ModuleType:
    """Extend one loaded ``search_agent_legacy`` module idempotently."""

    if not isinstance(module, ModuleType):
        raise ValueError("module must be a loaded module.")
    if getattr(module, "_evidence_graph_agent_tool_installed", False):
        return module
    schemas = getattr(module, "TOOLS_SCHEMA", None)
    parameter_schemas = getattr(module, "_TOOL_PARAMETER_SCHEMAS", None)
    agent_class = getattr(module, "SearchAgent", None)
    original_dispatch = getattr(agent_class, "_dispatch", None)
    if not isinstance(schemas, list):
        raise RuntimeError("search agent TOOLS_SCHEMA is unavailable.")
    if not isinstance(parameter_schemas, dict):
        raise RuntimeError(
            "search agent parameter schema registry is unavailable."
        )
    if not isinstance(agent_class, type) or not callable(original_dispatch):
        raise RuntimeError("search agent dispatch boundary is unavailable.")

    schema = copy.deepcopy(GRAPH_RAG_SEARCH_TOOL_DEF)
    schemas[:] = [
        value for value in schemas if _schema_name(value) != _TOOL_NAME
    ]
    schemas.append(schema)
    parameter_schemas[_TOOL_NAME] = copy.deepcopy(
        schema["function"]["parameters"]
    )

    def dispatch(self: Any, tool_name: str, arguments: dict[str, Any]):
        if tool_name == _TOOL_NAME:
            citations = search_evidence_graph(
                owner_id=getattr(self, "owner_id"),
                **arguments,
            )
            return "Reviewed evidence-graph citations retrieved.", citations
        return original_dispatch(self, tool_name, arguments)

    agent_class._dispatch = dispatch
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if isinstance(prompt, str) and _PROMPT_LINE not in prompt:
        module.SYSTEM_PROMPT = prompt.rstrip() + "\n" + _PROMPT_LINE + "\n"
    module._evidence_graph_original_dispatch = original_dispatch
    module._evidence_graph_agent_tool_installed = True
    return module


__all__ = ["install_evidence_graph_agent_tool"]
