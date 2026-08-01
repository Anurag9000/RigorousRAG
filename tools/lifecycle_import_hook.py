"""Lazy import hooks for fourth-store lifecycle coordination."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any, Callable

_MARKER = "_rigorousrag_lifecycle_import_hook"
_TARGETS: dict[str, tuple[str, str]] = {
    "tools.authoritative_document_index": (
        "tools.lifecycle_boundary",
        "install_authoritative_lifecycle_boundary",
    ),
    "tools.rag": (
        "tools.lifecycle_boundary",
        "install_rag_lifecycle_boundary",
    ),
    "tools.document_store": (
        "tools.lifecycle_source_context",
        "install_document_store_source_boundary",
    ),
    "tools.document_service": (
        "tools.lifecycle_source_context",
        "install_document_service_source_boundary",
    ),
}


class _LifecycleLoader(importlib.abc.Loader):
    def __init__(
        self,
        wrapped: importlib.abc.Loader,
        installer_name: str,
        installer_module: str = "tools.lifecycle_boundary",
    ) -> None:
        self._wrapped = wrapped
        self._installer_name = installer_name
        self._installer_module = installer_module

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module: ModuleType) -> None:
        execute = getattr(self._wrapped, "exec_module", None)
        if not callable(execute):
            raise ImportError("lifecycle target loader cannot execute modules.")
        execute(module)
        boundary = importlib.import_module(self._installer_module)
        installer: Callable[[ModuleType], None] | None = getattr(
            boundary,
            self._installer_name,
            None,
        )
        if not callable(installer):
            raise ImportError("lifecycle boundary installer is unavailable.")
        installed_module = sys.modules.get(module.__name__, module)
        if not isinstance(installed_module, ModuleType):
            raise ImportError("lifecycle target did not publish a module.")
        installer(installed_module)


class _LifecycleFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> Any:
        target_definition = _TARGETS.get(fullname)
        if target_definition is None:
            return None
        installer_module, installer_name = target_definition
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _LifecycleLoader):
            return spec
        spec.loader = _LifecycleLoader(
            spec.loader,
            installer_name,
            installer_module,
        )
        return spec


def install_lifecycle_import_hook() -> None:
    if any(getattr(finder, _MARKER, False) for finder in sys.meta_path):
        return
    finder = _LifecycleFinder()
    setattr(finder, _MARKER, True)
    sys.meta_path.insert(0, finder)


install_lifecycle_import_hook()

__all__ = ["install_lifecycle_import_hook"]
