"""Explicit governed agent-tool registry.

This replaces per-feature import-hook assumptions with a reusable registry contract for
schema validation, permission checks, owner injection, citation publication and runtime
budgets. Existing agent code can migrate tools into this registry incrementally.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from tools.capability_registry import CapabilityRegistry
from tools.models import Citation
from tools.retrieval_runtime import RuntimeBudget
from tools.security import normalize_owner_id

_TOOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_MAX_TOOLS = 1024


def _text(value: Any, label: str, maximum: int = 500, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _validate_schema(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(schema, Mapping):
        raise ValueError("tool schema must be a mapping")
    payload = dict(schema)
    if payload.get("type") != "object":
        raise ValueError("tool schema root must be an object")
    properties = payload.get("properties", {})
    if not isinstance(properties, Mapping) or len(properties) > 128:
        raise ValueError("tool schema properties are invalid")
    if payload.get("additionalProperties", False) is not False:
        raise ValueError("agent tool schemas must be closed")
    if len(_canonical(payload)) > 256_000:
        raise ValueError("tool schema exceeds its size limit")
    return payload


def _validate_value(value: Any, schema: Mapping[str, Any], path: str = "arguments") -> Any:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = set(schema.get("required", ()))
        unknown = set(value) - set(properties)
        if unknown and schema.get("additionalProperties", False) is False:
            raise ValueError(f"{path} contains unknown fields")
        missing = required - set(value)
        if missing:
            raise ValueError(f"{path} is missing required fields")
        return {key: _validate_value(item, properties[key], f"{path}.{key}") for key, item in value.items()}
    if expected == "array":
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise ValueError(f"{path} must be an array")
        maximum = int(schema.get("maxItems", 100))
        minimum = int(schema.get("minItems", 0))
        if not minimum <= len(value) <= maximum:
            raise ValueError(f"{path} has an invalid number of items")
        item_schema = schema.get("items", {})
        return [_validate_value(item, item_schema, f"{path}[]") for item in value]
    if expected == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path} must be a string")
        minimum, maximum = int(schema.get("minLength", 0)), int(schema.get("maxLength", 20_000))
        if not minimum <= len(value) <= maximum or any(ord(ch) == 0 for ch in value):
            raise ValueError(f"{path} string length/content is invalid")
        if "enum" in schema and value not in schema["enum"]:
            raise ValueError(f"{path} is not an allowed value")
        return value
    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        if "minimum" in schema and value < schema["minimum"] or "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} integer is outside its range")
        return value
    if expected == "number":
        if isinstance(value, bool):
            raise ValueError(f"{path} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite")
        if "minimum" in schema and number < schema["minimum"] or "maximum" in schema and number > schema["maximum"]:
            raise ValueError(f"{path} number is outside its range")
        return number
    if expected == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be boolean")
        return value
    if expected is None:
        return value
    raise ValueError(f"unsupported schema type at {path}")


@dataclass(frozen=True)
class ToolResult:
    content: Mapping[str, Any]
    citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.content, Mapping):
            raise ValueError("tool content must be a mapping")
        try:
            _canonical(dict(self.content))
        except (TypeError, ValueError) as exc:
            raise ValueError("tool content must be strict JSON") from exc
        if len(self.citations) > 100 or any(not isinstance(item, Citation) for item in self.citations):
            raise ValueError("tool citations are invalid")
        if len(self.warnings) > 100:
            raise ValueError("too many tool warnings")
        object.__setattr__(self, "warnings", tuple(_text(item, "warning", 2000) for item in self.warnings))


ToolHandler = Callable[[str, Mapping[str, Any], RuntimeBudget | None], ToolResult]


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    description: str
    schema: Mapping[str, Any]
    handler: ToolHandler
    required_permissions: tuple[str, ...] = ()
    capability_id: str = ""
    citation_policy: str = "server_only"
    owner_injected: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TOOL_RE.fullmatch(self.name):
            raise ValueError("tool name is invalid")
        object.__setattr__(self, "description", _text(self.description, "description", 4000))
        object.__setattr__(self, "schema", _validate_schema(self.schema))
        if not callable(self.handler):
            raise ValueError("handler must be callable")
        if len(self.required_permissions) > 64:
            raise ValueError("required_permissions exceed the item limit")
        object.__setattr__(self, "required_permissions", tuple(dict.fromkeys(_text(item, "permission", 100) for item in self.required_permissions)))
        object.__setattr__(self, "capability_id", _text(self.capability_id, "capability_id", 200, allow_empty=True))
        policy = _text(self.citation_policy, "citation_policy", 32).lower()
        if policy not in {"server_only", "none"}:
            raise ValueError("unsupported citation policy")
        object.__setattr__(self, "citation_policy", policy)
        if not isinstance(self.owner_injected, bool):
            raise ValueError("owner_injected must be boolean")

    @property
    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "description": self.description,
            "schema": dict(self.schema),
            "required_permissions": self.required_permissions,
            "capability_id": self.capability_id,
            "citation_policy": self.citation_policy,
            "owner_injected": self.owner_injected,
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


class AgentToolRegistry:
    def __init__(self, *, capabilities: CapabilityRegistry | None = None) -> None:
        self._tools: dict[str, AgentToolSpec] = {}
        self.capabilities = capabilities

    def register(self, spec: AgentToolSpec) -> None:
        if not isinstance(spec, AgentToolSpec):
            raise TypeError("spec must be AgentToolSpec")
        existing = self._tools.get(spec.name)
        if existing is not None and existing.fingerprint != spec.fingerprint:
            raise ValueError("a different tool is already registered under this name")
        if existing is None and len(self._tools) >= _MAX_TOOLS:
            raise RuntimeError("agent tool registry capacity reached")
        self._tools[spec.name] = spec

    def schemas(self, *, permissions: Sequence[str] = ()) -> tuple[Mapping[str, Any], ...]:
        allowed = frozenset(permissions)
        output = []
        for spec in sorted(self._tools.values(), key=lambda item: item.name):
            if not set(spec.required_permissions).issubset(allowed):
                continue
            output.append({"type": "function", "function": {"name": spec.name, "description": spec.description, "parameters": dict(spec.schema)}})
        return tuple(output)

    def dispatch(
        self,
        *,
        name: str,
        owner_id: str,
        arguments: Mapping[str, Any],
        permissions: Sequence[str] = (),
        budget: RuntimeBudget | None = None,
    ) -> ToolResult:
        spec = self._tools[_text(name, "tool name", 128)]
        owner = normalize_owner_id(owner_id)
        granted = frozenset(permissions)
        if not set(spec.required_permissions).issubset(granted):
            raise PermissionError("tool permission requirement is not satisfied")
        if spec.capability_id and self.capabilities is not None:
            self.capabilities.resolve(spec.capability_id, required_permissions=spec.required_permissions)
        validated = _validate_value(arguments, spec.schema)
        if budget is not None:
            budget.reserve(calls=1)
        result = spec.handler(owner, validated, budget)
        if not isinstance(result, ToolResult):
            raise RuntimeError("tool handler returned an invalid result")
        if spec.citation_policy == "none" and result.citations:
            raise RuntimeError("tool returned citations while its citation policy forbids them")
        return result

    @property
    def fingerprint(self) -> str:
        payload = [(name, spec.fingerprint) for name, spec in sorted(self._tools.items())]
        return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = ["AgentToolRegistry", "AgentToolSpec", "ToolHandler", "ToolResult"]
