"""Lazy import hook that registers governed research-agent retrieval and answer gates.

The hook covers three import orders without importing the retrieval stacks eagerly:

* a future ``search_agent_legacy`` import is wrapped through ``PathFinder``;
* an already-complete module is installed immediately; and
* a module currently executing is temporarily watched until its schema registry
  and ``SearchAgent`` class have all been assigned.

The governed tool registry, adaptive retrieval, evidence graph, bounded multi-hop,
source-status citation gate, optional semantic entailment gate and final evidence-
admissibility gate share this hook so one import/reload path defines the production
ordering. Each integration remains independently idempotent and fail closed.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any

_TARGET = "search_agent_legacy"
_MARKER = "_rigorousrag_evidence_graph_agent_import_hook"
_REQUIRED_ATTRIBUTES = frozenset({"TOOLS_SCHEMA", "_TOOL_PARAMETER_SCHEMAS", "SearchAgent"})
_ORIGINAL_MODULE_CLASS = "_evidence_graph_original_module_class"
_INTEGRATION_MARKERS = (
    "_agent_tool_registry_bridge_installed",
    "_agent_tool_registry_original_dispatch",
    "_agent_tool_registry_dispatcher_name",
    "_adaptive_agent_tool_installed",
    "_adaptive_original_dispatch",
    "_evidence_graph_agent_tool_installed",
    "_evidence_graph_original_dispatch",
    "_multihop_agent_tool_installed",
    "_multihop_original_dispatch",
    "_source_status_agent_gate_installed",
    "_source_status_original_register_citations",
    "_source_status_original_run",
    "_claim_entailment_agent_gate_installed",
    "_claim_entailment_original_run",
    "_evidence_admissibility_agent_gate_installed",
    "_evidence_admissibility_original_run",
)


def _ready(module: ModuleType) -> bool:
    return all(hasattr(module, name) for name in _REQUIRED_ATTRIBUTES)


def _install(module: ModuleType) -> None:
    from tools.adaptive_agent_integration import install_adaptive_agent_tool
    from tools.agent_tool_registry_integration import install_agent_tool_registry_bridge
    from tools.claim_entailment_agent_integration import install_claim_entailment_agent_gate
    from tools.evidence_admissibility_agent_integration import (
        install_evidence_admissibility_agent_gate,
    )
    from tools.evidence_graph_agent_integration import install_evidence_graph_agent_tool
    from tools.multihop_agent_integration import install_multihop_agent_tool
    from tools.source_status_agent_integration import install_source_status_agent_gate

    install_agent_tool_registry_bridge(module)
    install_adaptive_agent_tool(module)
    install_evidence_graph_agent_tool(module)
    install_multihop_agent_tool(module)
    # Publication order is deliberate:
    # 1. source status removes administratively unusable evidence;
    # 2. entailment removes semantically unsupported claims/citations when configured;
    # 3. admissibility applies reviewed trust/method policy to what remains.
    install_source_status_agent_gate(module)
    install_claim_entailment_agent_gate(module)
    install_evidence_admissibility_agent_gate(module)


def _disarm(module: ModuleType) -> None:
    original = module.__dict__.pop(_ORIGINAL_MODULE_CLASS, ModuleType)
    if isinstance(original, type) and issubclass(original, ModuleType):
        module.__class__ = original


def _install_if_ready(module: ModuleType) -> bool:
    if not _ready(module):
        return False
    _install(module)
    _disarm(module)
    return True


class _DeferredEvidenceGraphModule(ModuleType):
    """Watch only the final required assignments of an in-progress import."""

    def __setattr__(self, name: str, value: Any) -> None:
        ModuleType.__setattr__(self, name, value)
        if name in _REQUIRED_ATTRIBUTES:
            _install_if_ready(self)


def _arm_deferred_install(module: ModuleType) -> None:
    if isinstance(module, _DeferredEvidenceGraphModule):
        return
    if module.__class__ is not ModuleType:
        raise RuntimeError("search_agent_legacy uses an unsupported custom module class.")
    ModuleType.__setattr__(module, _ORIGINAL_MODULE_CLASS, module.__class__)
    module.__class__ = _DeferredEvidenceGraphModule
    _install_if_ready(module)


class _EvidenceGraphAgentLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader) -> None:
        self._wrapped = wrapped

    def create_module(self, spec: Any) -> ModuleType | None:
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if callable(create) else None

    def exec_module(self, module: ModuleType) -> None:
        execute = getattr(self._wrapped, "exec_module", None)
        if not callable(execute):
            raise ImportError("search_agent_legacy loader cannot execute modules.")
        for name in _INTEGRATION_MARKERS:
            module.__dict__.pop(name, None)
        execute(module)
        _install(module)


class _EvidenceGraphAgentFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: ModuleType | None = None) -> Any:
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
    if isinstance(existing, ModuleType):
        if _ready(existing):
            _install(existing)
        else:
            _arm_deferred_install(existing)
    if any(getattr(finder, _MARKER, False) for finder in sys.meta_path):
        return
    finder = _EvidenceGraphAgentFinder()
    setattr(finder, _MARKER, True)
    sys.meta_path.insert(0, finder)


install_evidence_graph_agent_import_hook()

__all__ = ["install_evidence_graph_agent_import_hook"]
