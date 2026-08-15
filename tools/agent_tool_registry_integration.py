"""Bridge ``AgentToolRegistry`` into the legacy research-agent dispatcher.

This is the migration seam away from per-tool dispatch monkey patches. Existing graph
and multi-hop integrations remain compatible while new tools can register once through
the governed registry and inherit closed-schema validation, owner injection, budgets and
citation policy.
"""

from __future__ import annotations

import json
from types import ModuleType
from typing import Any

from tools.agent_tool_registry import AgentToolRegistry, AgentToolSpec
from tools.retrieval_runtime import BudgetLimits, RuntimeBudget

_MARKER = "_agent_tool_registry_bridge_installed"
_ORIGINAL = "_agent_tool_registry_original_dispatch"
_DISPATCHER_NAME = "_agent_tool_registry_dispatcher_name"
_REGISTRY = "AGENT_TOOL_REGISTRY"
_MAX_TOOL_CALL_WALL_MS = 30_000.0


def _registry(module: ModuleType) -> AgentToolRegistry:
    existing = getattr(module, _REGISTRY, None)
    if existing is None:
        existing = AgentToolRegistry()
        setattr(module, _REGISTRY, existing)
    if not isinstance(existing, AgentToolRegistry):
        raise RuntimeError("search_agent_legacy AGENT_TOOL_REGISTRY has an incompatible type")
    return existing


def _schema_names(module: ModuleType) -> set[str]:
    names: set[str] = set()
    for entry in getattr(module, "TOOLS_SCHEMA", ()):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _sync_schemas(module: ModuleType) -> None:
    registry = _registry(module)
    schemas = getattr(module, "TOOLS_SCHEMA", None)
    parameter_schemas = getattr(module, "_TOOL_PARAMETER_SCHEMAS", None)
    if not isinstance(schemas, list) or not isinstance(parameter_schemas, dict):
        raise RuntimeError("search_agent_legacy tool schema registries are incompatible")
    known = _schema_names(module)
    for entry in registry.schemas(permissions=()):
        function = entry.get("function", {})
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        if name not in known:
            schemas.append(entry)
            known.add(name)
        parameter_schemas[name] = parameters


def register_governed_agent_tool(module: ModuleType, spec: AgentToolSpec) -> None:
    registry = _registry(module)
    registry.register(spec)
    _sync_schemas(module)


def _dispatcher_name(search_agent: type[Any]) -> str:
    """Return the live dispatcher boundary, preferring the production ``_dispatch`` API."""

    for name in ("_dispatch", "_dispatch_tool"):
        if callable(getattr(search_agent, name, None)):
            return name
    raise RuntimeError("search_agent_legacy does not expose the expected SearchAgent dispatcher")


def install_agent_tool_registry_bridge(module: ModuleType) -> None:
    if getattr(module, _MARKER, False):
        _sync_schemas(module)
        return
    search_agent = getattr(module, "SearchAgent", None)
    if not isinstance(search_agent, type):
        raise RuntimeError("search_agent_legacy does not expose the expected SearchAgent dispatcher")
    dispatcher_name = _dispatcher_name(search_agent)
    _registry(module)
    original = getattr(search_agent, _ORIGINAL, None)
    if original is None:
        original = getattr(search_agent, dispatcher_name)
        setattr(search_agent, _ORIGINAL, original)
    if not callable(original):
        raise RuntimeError("search_agent_legacy original dispatcher is incompatible")

    def dispatch_with_registry(self: Any, name: str, args: dict[str, Any]):
        current = _registry(module)
        registered_names = {spec.name for spec in current._tools.values()}  # registry-owned read only
        if name not in registered_names:
            return original(self, name, args)
        permissions = tuple(getattr(self, "permissions", ()) or ())
        budget = RuntimeBudget(
            BudgetLimits(
                max_wall_ms=_MAX_TOOL_CALL_WALL_MS,
                max_calls=4,
                max_input_tokens=100_000,
                max_output_tokens=50_000,
                max_cost=10.0,
            )
        )
        result = current.dispatch(
            name=name,
            owner_id=self.owner_id,
            arguments=args,
            permissions=permissions,
            budget=budget,
        )
        snapshot = budget.snapshot()
        content = json.dumps(
            {
                "result": dict(result.content),
                "warnings": list(result.warnings),
                "runtime_budget": {
                    "elapsed_ms": snapshot.elapsed_ms,
                    "calls": snapshot.calls,
                    "input_tokens": snapshot.input_tokens,
                    "output_tokens": snapshot.output_tokens,
                    "cost": snapshot.cost,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return content, list(result.citations)

    setattr(search_agent, dispatcher_name, dispatch_with_registry)
    setattr(module, _DISPATCHER_NAME, dispatcher_name)
    _sync_schemas(module)
    setattr(module, _MARKER, True)


__all__ = ["install_agent_tool_registry_bridge", "register_governed_agent_tool"]
