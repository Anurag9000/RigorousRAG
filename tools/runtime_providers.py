"""Process-local registry for explicitly injected trusted runtime provider objects.

This is not plugin discovery. Application bootstrap code must register concrete objects
under known identifiers before composing the production app. Objects are never exposed by
snapshot APIs; only non-secret binding metadata and health are observable.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")
_MAX_BINDINGS = 256


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class RuntimeProviderBinding:
    provider_id: str
    capabilities: tuple[str, ...] = ()
    version: str = "1.0.0"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _identifier(self.provider_id, "provider_id"))
        if len(self.capabilities) > 64:
            raise ValueError("capabilities exceed the item limit")
        object.__setattr__(
            self,
            "capabilities",
            tuple(dict.fromkeys(_identifier(item, "capability") for item in self.capabilities)),
        )
        object.__setattr__(self, "version", _identifier(self.version, "version"))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("provider metadata must be a bounded mapping")
        safe: dict[str, str] = {}
        for key, value in self.metadata.items():
            name = _identifier(str(key), "metadata key")
            text = str(value).replace("\x00", " ").strip()
            if len(text) > 500:
                raise ValueError("provider metadata value is too long")
            safe[name] = text
        object.__setattr__(self, "metadata", safe)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical(
                {
                    "provider_id": self.provider_id,
                    "capabilities": self.capabilities,
                    "version": self.version,
                    "metadata": dict(sorted(self.metadata.items())),
                }
            )
        ).hexdigest()


class RuntimeProviderRegistry:
    def __init__(self) -> None:
        self._objects: dict[str, Any] = {}
        self._bindings: dict[str, RuntimeProviderBinding] = {}
        self._health: dict[str, Callable[[], bool] | bool] = {}
        self._lock = threading.RLock()

    def register(
        self,
        provider_id: str,
        provider: Any,
        *,
        capabilities: Sequence[str] = (),
        version: str = "1.0.0",
        metadata: Mapping[str, str] | None = None,
        health_check: Callable[[], bool] | bool = True,
        replace: bool = False,
    ) -> RuntimeProviderBinding:
        identifier = _identifier(provider_id, "provider_id")
        if provider is None:
            raise ValueError("provider object must be supplied")
        binding = RuntimeProviderBinding(
            identifier,
            tuple(capabilities),
            version,
            metadata or {},
        )
        with self._lock:
            if identifier in self._objects and not replace:
                if self._objects[identifier] is not provider:
                    raise ValueError("a different provider object is already registered")
                return self._bindings[identifier]
            if identifier not in self._objects and len(self._objects) >= _MAX_BINDINGS:
                raise RuntimeError("runtime provider registry capacity reached")
            self._objects[identifier] = provider
            self._bindings[identifier] = binding
            self._health[identifier] = health_check
        return binding

    def unregister(self, provider_id: str) -> bool:
        identifier = _identifier(provider_id, "provider_id")
        with self._lock:
            existed = identifier in self._objects
            self._objects.pop(identifier, None)
            self._bindings.pop(identifier, None)
            self._health.pop(identifier, None)
            return existed

    def get(self, provider_id: str) -> Any | None:
        identifier = _identifier(provider_id, "provider_id")
        with self._lock:
            return self._objects.get(identifier)

    def require(self, provider_id: str) -> Any:
        value = self.get(provider_id)
        if value is None:
            raise RuntimeError(f"required runtime provider is not registered: {provider_id}")
        if not self.healthy(provider_id):
            raise RuntimeError(f"required runtime provider is unhealthy: {provider_id}")
        return value

    def healthy(self, provider_id: str) -> bool:
        identifier = _identifier(provider_id, "provider_id")
        with self._lock:
            if identifier not in self._objects:
                return False
            check = self._health.get(identifier, True)
        try:
            return bool(check() if callable(check) else check)
        except Exception:
            return False

    def capability_health(self) -> Mapping[str, bool]:
        with self._lock:
            bindings = tuple(self._bindings.values())
        output: dict[str, bool] = {}
        for binding in bindings:
            state = self.healthy(binding.provider_id)
            for capability in binding.capabilities:
                output[capability] = output.get(capability, False) or state
        return output

    def bindings(self) -> tuple[RuntimeProviderBinding, ...]:
        with self._lock:
            return tuple(sorted(self._bindings.values(), key=lambda item: item.provider_id))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            _canonical([binding.fingerprint for binding in self.bindings()])
        ).hexdigest()


runtime_providers = RuntimeProviderRegistry()


__all__ = ["RuntimeProviderBinding", "RuntimeProviderRegistry", "runtime_providers"]
