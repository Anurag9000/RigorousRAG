"""Lazy import hooks for fourth-store lifecycle coordination."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any, Callable

_MARKER = "_rigorousrag_lifecycle_import_hook"
_TARGETS: dict[str, str] = {
    "tools.authoritative_document_index": "install_authoritative_lifecycle_boundary",
    "tools.rag": "install_rag_lifecycle_boundary",
}


class _LifecycleLoader(importlib.abc.Loader):
    def __init__(
        self,
        wrapped: importlib.abc.Loader,
        installer_name: str,
    ) -> None:
        self._wrapped = wrapped
        self._installer_name = installer_name

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module: ModuleType) -> None:
        execute = getattr(self._wrapped, "exec_module", None)
        if not callable(execute):
            raise ImportError("lifecycle target loader cannot execute modules.")
        execute(module)
        from tools import lifecycle_boundary

        installer: Callable[[ModuleType], None] | None = getattr(
            lifecycle_boundary,
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
        installer_name = _TARGETS.get(fullname)
        if installer_name is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        if isinstance(spec.loader, _LifecycleLoader):
            return spec
        spec.loader = _LifecycleLoader(spec.loader, installer_name)
        return spec


def install_lifecycle_import_hook() -> None:
    if any(getattr(finder, _MARKER, False) for finder in sys.meta_path):
        return
    finder = _LifecycleFinder()
    setattr(finder, _MARKER, True)
    sys.meta_path.insert(0, finder)


install_lifecycle_import_hook()

__all__ = ["install_lifecycle_import_hook"]
