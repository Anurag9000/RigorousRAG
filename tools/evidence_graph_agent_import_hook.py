"""Lazy import hook that registers GraphRAG on the existing research agent."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

_TARGET = "search_agent_legacy"
_MARKER = "_rigorousrag_evidence_graph_agent_import_hook"


class _EvidenceGraphAgentLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module: ModuleType) -> None:
        execute = getattr(self._wrapped, "exec_module", None)
        if not callable(execute):
            raise ImportError(
                "search_agent_legacy loader cannot execute modules."
            )
        execute(module)
        from tools.evidence_graph_agent_integration import (
            install_evidence_graph_agent_tool,
        )

        install_evidence_graph_agent_tool(module)


class _EvidenceGraphAgentFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> Any:
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _EvidenceGraphAgentLoader):
            return spec
        spec.loader = _EvidenceGraphAgentLoader(spec.loader)
        return spec


def install_evidence_graph_agent_import_hook() -> None:
    existing = sys.modules.get(_TARGET)
    if (
        isinstance(existing, ModuleType)
        and hasattr(existing, "TOOLS_SCHEMA")
        and hasattr(existing, "_TOOL_PARAMETER_SCHEMAS")
        and hasattr(existing, "SearchAgent")
    ):
        from tools.evidence_graph_agent_integration import (
            install_evidence_graph_agent_tool,
        )

        install_evidence_graph_agent_tool(existing)
    if any(getattr(finder, _MARKER, False) for finder in sys.meta_path):
        return
    finder = _EvidenceGraphAgentFinder()
    setattr(finder, _MARKER, True)
    sys.meta_path.insert(0, finder)


install_evidence_graph_agent_import_hook()

__all__ = ["install_evidence_graph_agent_import_hook"]
