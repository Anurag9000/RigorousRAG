"""Install authoritative GraphRAG discovery/retrieval into the existing agent surface."""

from __future__ import annotations

import copy
import json
import sys
from types import ModuleType
from typing import Any

from tools.evidence_graph_rag_tool import (
    GRAPH_RAG_SEARCH_TOOL_DEF,
    search_evidence_graph,
)
from tools.evidence_graph_set_discovery import (
    LIST_EVIDENCE_GRAPH_SETS_TOOL_DEF,
    list_evidence_graph_sets,
)

_LIST_TOOL_NAME = "list_evidence_graph_sets"
_SEARCH_TOOL_NAME = "search_evidence_graph"
_TOOL_NAMES = frozenset({_LIST_TOOL_NAME, _SEARCH_TOOL_NAME})
_PROMPT_LINES = (
    "- Reviewed Evidence Graph Sets (`list_evidence_graph_sets`) — discover "
    "the authenticated user's current reviewed graph-set keys before searching.",
    "- Reviewed Evidence Graph (`search_evidence_graph`) — use a discovered "
    "graph-set key for bounded cross-document retrieval over explicit, "
    "generation-validated relations.",
)


def _schema_name(value: Any) -> str | None:
    try:
        name = value["function"]["name"]
    except (KeyError, TypeError):
        return None
    return name if isinstance(name, str) else None


def _ensure_governed_pipeline_for_live_agent(module: ModuleType) -> None:
    """Complete the ordered pipeline when legacy is imported directly.

    ``search_agent_legacy`` historically finalised only the evidence-graph tool.
    When this installer is invoked by that live module before the shared import
    hook has run, bootstrap the shared hook after GraphRAG registration.  Calls
    made for isolated/fake modules remain local, and calls made from the shared
    pipeline are already marked by the registry bridge and therefore do not
    recurse.
    """

    if sys.modules.get("search_agent_legacy") is not module:
        return
    if getattr(module, "_agent_tool_registry_bridge_installed", False):
        return
    from tools.evidence_graph_agent_import_hook import (
        install_evidence_graph_agent_import_hook,
    )

    install_evidence_graph_agent_import_hook()


def install_evidence_graph_agent_tool(module: ModuleType) -> ModuleType:
    """Extend one loaded ``search_agent_legacy`` module idempotently."""

    if not isinstance(module, ModuleType):
        raise ValueError("module must be a loaded module.")
    if getattr(module, "_evidence_graph_agent_tool_installed", False):
        _ensure_governed_pipeline_for_live_agent(module)
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

    definitions = (
        copy.deepcopy(LIST_EVIDENCE_GRAPH_SETS_TOOL_DEF),
        copy.deepcopy(GRAPH_RAG_SEARCH_TOOL_DEF),
    )
    schemas[:] = [
        value for value in schemas if _schema_name(value) not in _TOOL_NAMES
    ]
    schemas.extend(definitions)
    for schema in definitions:
        name = schema["function"]["name"]
        parameter_schemas[name] = copy.deepcopy(
            schema["function"]["parameters"]
        )

    def dispatch(self: Any, tool_name: str, arguments: dict[str, Any]):
        owner_id = getattr(self, "owner_id")
        if tool_name == _LIST_TOOL_NAME:
            values = list_evidence_graph_sets(
                owner_id=owner_id,
                **arguments,
            )
            return (
                json.dumps(
                    {
                        "graph_sets": values,
                        "count": len(values),
                        "source_text_returned": False,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
                [],
            )
        if tool_name == _SEARCH_TOOL_NAME:
            citations = search_evidence_graph(
                owner_id=owner_id,
                **arguments,
            )
            return "Reviewed evidence-graph citations retrieved.", citations
        return original_dispatch(self, tool_name, arguments)

    agent_class._dispatch = dispatch
    prompt = getattr(module, "SYSTEM_PROMPT", None)
    if isinstance(prompt, str):
        updated = prompt.rstrip()
        for line in _PROMPT_LINES:
            if line not in updated:
                updated += "\n" + line
        module.SYSTEM_PROMPT = updated + "\n"
    module._evidence_graph_original_dispatch = original_dispatch
    module._evidence_graph_agent_tool_installed = True
    _ensure_governed_pipeline_for_live_agent(module)
    return module


__all__ = ["install_evidence_graph_agent_tool"]
