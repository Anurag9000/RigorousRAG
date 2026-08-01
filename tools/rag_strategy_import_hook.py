"""Lazy import boundary that installs authoritative RAG strategies.

The hook wraps only ``tools.rag_tool`` and delegates source loading to Python's
standard ``PathFinder``. It avoids importing the RAG stack merely because an
unrelated ``tools`` submodule is used, while ensuring all normal imports and
reloads receive the extended schema and server-owned Citation return path.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

_TARGET = "tools.rag_tool"
_MARKER = "_rigorousrag_strategy_import_hook"


class _StrategyLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module: ModuleType) -> None:
        execute = getattr(self._wrapped, "exec_module", None)
        if not callable(execute):
            raise ImportError("tools.rag_tool loader cannot execute modules.")
        execute(module)
        from tools.rag_strategy_integration import install_rag_strategies

        base_schema = getattr(module, "RAG_SEARCH_TOOL_DEF", None)
        base_search = getattr(module, "search_uploaded_docs", None)
        schema, search = install_rag_strategies(base_schema, base_search)
        module._strategy_original_RAG_SEARCH_TOOL_DEF = base_schema
        module._strategy_original_search_uploaded_docs = base_search
        module.RAG_SEARCH_TOOL_DEF = schema
        module.search_uploaded_docs = search
        module._rag_strategies_installed = True


class _StrategyFinder(importlib.abc.MetaPathFinder):
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
        if isinstance(spec.loader, _StrategyLoader):
            return spec
        spec.loader = _StrategyLoader(spec.loader)
        return spec


def install_rag_strategy_import_hook() -> None:
    if any(getattr(finder, _MARKER, False) for finder in sys.meta_path):
        return
    finder = _StrategyFinder()
    setattr(finder, _MARKER, True)
    sys.meta_path.insert(0, finder)


install_rag_strategy_import_hook()

__all__ = ["install_rag_strategy_import_hook"]
