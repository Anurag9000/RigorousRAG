"""Unified runtime capability registry for governed RigorousRAG backends.

This registry is deliberately model/provider agnostic.  It describes *what* a runtime
capability can do, the trust/resource envelope in which it is allowed to run, its
dependencies and explicit fallbacks.  It does not download models, train artifacts or
silently promote implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

_VERSION_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][A-Za-z0-9.-]+)?$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")
_ALLOWED_KINDS = frozenset(
    {
        "embedding",
        "sparse_retriever",
        "late_interaction",
        "multimodal_retriever",
        "reranker",
        "router",
        "planner",
        "generator",
        "extractor",
        "graph_backend",
        "storage",
        "queue",
        "coordination",
        "parser",
        "ocr",
        "scanner",
        "secret_provider",
        "domain_adapter",
        "tool",
        "other",
    }
)
_ALLOWED_MODALITIES = frozenset(
    {"text", "image", "page_image", "table", "figure", "formula", "graph", "raster", "timeseries", "geospatial"}
)
_ALLOWED_TRUST_LEVELS = frozenset({"local", "sandboxed", "private_remote", "public_remote"})
_MAX_CAPABILITIES = 10_000
_MAX_DEPENDENCIES = 64
_MAX_FALLBACKS = 32
_MAX_SCHEMA_BYTES = 256_000


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not _IDENTIFIER_RE.fullmatch(cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _version(value: Any) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value.strip()):
        raise ValueError("version must follow semantic version form MAJOR.MINOR.PATCH")
    return value.strip()


def _bounded_strings(values: Sequence[str], label: str, maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence")
    if len(values) > maximum:
        raise ValueError(f"{label} exceeds the item limit")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _identifier(value, label)
        if item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _finite_nonnegative(value: Any, label: str, maximum: float = 1e15) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative") from exc
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > maximum:
        raise ValueError(f"{label} must be finite, non-negative and bounded")
    return parsed


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ResourceEnvelope:
    """Maximum resource envelope advertised by a capability."""

    max_calls: int = 1
    max_latency_ms: float = 0.0
    max_input_bytes: int = 0
    max_output_bytes: int = 0
    max_tokens: int = 0
    max_cost: float = 0.0
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        for name in ("max_calls", "max_input_bytes", "max_output_bytes", "max_tokens", "max_concurrency"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 10**12:
                raise ValueError(f"{name} must be a bounded non-negative integer")
        object.__setattr__(self, "max_latency_ms", _finite_nonnegative(self.max_latency_ms, "max_latency_ms"))
        object.__setattr__(self, "max_cost", _finite_nonnegative(self.max_cost, "max_cost"))


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Immutable declaration for one concrete runtime capability version."""

    capability_id: str
    version: str
    kind: str
    provider: str
    modalities: tuple[str, ...] = ("text",)
    dependencies: tuple[str, ...] = ()
    fallbacks: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    trust_level: str = "local"
    config_schema: Mapping[str, Any] = field(default_factory=dict)
    resources: ResourceEnvelope = field(default_factory=ResourceEnvelope)
    artifact_sha256: str = ""
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "capability_id", _identifier(self.capability_id, "capability_id"))
        object.__setattr__(self, "version", _version(self.version))
        selected_kind = _identifier(self.kind, "kind").lower()
        if selected_kind not in _ALLOWED_KINDS:
            raise ValueError("unsupported capability kind")
        object.__setattr__(self, "kind", selected_kind)
        object.__setattr__(self, "provider", _identifier(self.provider, "provider"))
        if not isinstance(self.modalities, tuple):
            object.__setattr__(self, "modalities", tuple(self.modalities))
        modalities = tuple(dict.fromkeys(str(item).strip().lower() for item in self.modalities))
        if not modalities or any(item not in _ALLOWED_MODALITIES for item in modalities):
            raise ValueError("modalities contain unsupported values")
        object.__setattr__(self, "modalities", modalities)
        object.__setattr__(self, "dependencies", _bounded_strings(self.dependencies, "dependencies", _MAX_DEPENDENCIES))
        object.__setattr__(self, "fallbacks", _bounded_strings(self.fallbacks, "fallbacks", _MAX_FALLBACKS))
        object.__setattr__(self, "permissions", _bounded_strings(self.permissions, "permissions", 64))
        if self.capability_id in self.dependencies or self.capability_id in self.fallbacks:
            raise ValueError("a capability cannot depend on or fall back to itself")
        if self.trust_level not in _ALLOWED_TRUST_LEVELS:
            raise ValueError("unsupported trust_level")
        if not isinstance(self.config_schema, Mapping):
            raise ValueError("config_schema must be a mapping")
        if len(_canonical_json(dict(self.config_schema))) > _MAX_SCHEMA_BYTES:
            raise ValueError("config_schema exceeds the bounded size")
        if not isinstance(self.resources, ResourceEnvelope):
            raise ValueError("resources must be a ResourceEnvelope")
        digest = self.artifact_sha256.lower().strip() if isinstance(self.artifact_sha256, str) else ""
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("artifact_sha256 must be empty or a SHA-256 hex digest")
        object.__setattr__(self, "artifact_sha256", digest)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")

    @property
    def key(self) -> str:
        return f"{self.capability_id}@{self.version}"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(asdict(self))).hexdigest()


@dataclass(frozen=True)
class CapabilityHealth:
    available: bool
    reason: str = "available"

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise ValueError("available must be boolean")
        if not isinstance(self.reason, str) or not self.reason.strip() or len(self.reason) > 500:
            raise ValueError("reason is invalid")
        object.__setattr__(self, "reason", " ".join(self.reason.split()))


@dataclass(frozen=True)
class CapabilityResolution:
    requested_id: str
    selected: CapabilityDescriptor
    dependency_order: tuple[CapabilityDescriptor, ...]
    fallback_used: bool
    resolution_fingerprint: str


HealthCheck = Callable[[CapabilityDescriptor], CapabilityHealth]


class CapabilityRegistry:
    """Thread-safe registry with dependency validation and explicit fallbacks."""

    def __init__(self) -> None:
        self._records: dict[str, CapabilityDescriptor] = {}
        self._active: dict[str, str] = {}
        self._health_checks: dict[str, HealthCheck] = {}
        self._lock = threading.RLock()

    def register(
        self,
        descriptor: CapabilityDescriptor,
        *,
        activate: bool = False,
        health_check: HealthCheck | None = None,
    ) -> None:
        if not isinstance(descriptor, CapabilityDescriptor):
            raise TypeError("descriptor must be CapabilityDescriptor")
        with self._lock:
            if descriptor.key in self._records and self._records[descriptor.key] != descriptor:
                raise ValueError("a different descriptor is already registered for this capability version")
            if descriptor.key not in self._records and len(self._records) >= _MAX_CAPABILITIES:
                raise ValueError("capability registry limit reached")
            self._records[descriptor.key] = descriptor
            if health_check is not None:
                self._health_checks[descriptor.key] = health_check
            if activate:
                self._active[descriptor.capability_id] = descriptor.key

    def promote(self, capability_id: str, version: str) -> CapabilityDescriptor:
        identifier = _identifier(capability_id, "capability_id")
        key = f"{identifier}@{_version(version)}"
        with self._lock:
            descriptor = self._records.get(key)
            if descriptor is None:
                raise KeyError(key)
            if not descriptor.enabled:
                raise ValueError("disabled capabilities cannot be promoted")
            self._active[identifier] = key
            return descriptor

    def active(self, capability_id: str) -> CapabilityDescriptor | None:
        identifier = _identifier(capability_id, "capability_id")
        with self._lock:
            key = self._active.get(identifier)
            return self._records.get(key) if key else None

    def versions(self, capability_id: str) -> tuple[CapabilityDescriptor, ...]:
        identifier = _identifier(capability_id, "capability_id")
        with self._lock:
            values = [record for record in self._records.values() if record.capability_id == identifier]
        values.sort(key=lambda item: tuple(int(part) if part.isdigit() else part for part in re.split(r"[.+-]", item.version)), reverse=True)
        return tuple(values)

    def health(self, descriptor: CapabilityDescriptor) -> CapabilityHealth:
        if not descriptor.enabled:
            return CapabilityHealth(False, "disabled")
        check = self._health_checks.get(descriptor.key)
        if check is None:
            return CapabilityHealth(True)
        try:
            result = check(descriptor)
        except Exception:
            return CapabilityHealth(False, "health_check_failed")
        return result if isinstance(result, CapabilityHealth) else CapabilityHealth(False, "invalid_health_result")

    def _active_record(self, capability_id: str) -> CapabilityDescriptor:
        record = self.active(capability_id)
        if record is None:
            raise KeyError(f"no active capability for {capability_id}")
        return record

    def _dependency_order(self, root: CapabilityDescriptor) -> tuple[CapabilityDescriptor, ...]:
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[CapabilityDescriptor] = []

        def visit(record: CapabilityDescriptor) -> None:
            if record.capability_id in visiting:
                raise ValueError("capability dependency graph contains a cycle")
            if record.capability_id in visited:
                return
            visiting.add(record.capability_id)
            for dependency_id in record.dependencies:
                dependency = self._active_record(dependency_id)
                if not self.health(dependency).available:
                    raise RuntimeError(f"required capability {dependency_id} is unavailable")
                visit(dependency)
            visiting.remove(record.capability_id)
            visited.add(record.capability_id)
            ordered.append(record)

        visit(root)
        return tuple(ordered)

    def resolve(
        self,
        capability_id: str,
        *,
        required_modalities: Sequence[str] = (),
        required_permissions: Sequence[str] = (),
        allow_fallback: bool = True,
    ) -> CapabilityResolution:
        requested = _identifier(capability_id, "capability_id")
        modalities = frozenset(str(value).strip().lower() for value in required_modalities)
        permissions = frozenset(_identifier(value, "permission") for value in required_permissions)
        if not modalities.issubset(_ALLOWED_MODALITIES):
            raise ValueError("required_modalities contain unsupported values")

        candidates: list[tuple[CapabilityDescriptor, bool]] = []
        primary = self.active(requested)
        if primary is not None:
            candidates.append((primary, False))
            if allow_fallback:
                for fallback_id in primary.fallbacks:
                    fallback = self.active(fallback_id)
                    if fallback is not None:
                        candidates.append((fallback, True))
        if not candidates:
            raise KeyError(f"no active capability for {requested}")

        failures: list[str] = []
        for candidate, fallback_used in candidates:
            if not modalities.issubset(candidate.modalities):
                failures.append("modality_mismatch")
                continue
            if not permissions.issubset(candidate.permissions):
                failures.append("permission_mismatch")
                continue
            state = self.health(candidate)
            if not state.available:
                failures.append(state.reason)
                continue
            try:
                dependency_order = self._dependency_order(candidate)
            except (KeyError, RuntimeError, ValueError):
                failures.append("dependency_unavailable")
                continue
            payload = {
                "requested": requested,
                "selected": candidate.fingerprint,
                "dependencies": [item.fingerprint for item in dependency_order],
                "fallback_used": fallback_used,
                "required_modalities": sorted(modalities),
                "required_permissions": sorted(permissions),
            }
            return CapabilityResolution(
                requested_id=requested,
                selected=candidate,
                dependency_order=dependency_order,
                fallback_used=fallback_used,
                resolution_fingerprint=hashlib.sha256(_canonical_json(payload)).hexdigest(),
            )
        raise RuntimeError(f"no compatible healthy capability for {requested}: {','.join(sorted(set(failures)))}")

    def snapshot(self) -> tuple[CapabilityDescriptor, ...]:
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda item: (item.capability_id, item.version)))

    @property
    def fingerprint(self) -> str:
        with self._lock:
            payload = {
                "active": dict(sorted(self._active.items())),
                "records": [asdict(item) for item in self.snapshot()],
            }
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


__all__ = [
    "CapabilityDescriptor",
    "CapabilityHealth",
    "CapabilityRegistry",
    "CapabilityResolution",
    "ResourceEnvelope",
]
