"""Governed domain-extension contracts for RigorousRAG.

A domain adapter contributes metadata normalization, query features, unit aliases,
graph enrichment and report fields without modifying the central research agent.  The
registry is intentionally explicit: adapters are registered by trusted application code,
never discovered by importing arbitrary user-controlled modules.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.document_ir import ScientificDocumentIR
from tools.graph_reasoning import GraphEdge, GraphNode

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}$")
_MAX_ADAPTERS = 256


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
class DomainDescriptor:
    domain_id: str
    version: str
    label: str
    supported_mime_types: tuple[str, ...] = ()
    supported_languages: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    metadata_schema: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain_id", _identifier(self.domain_id, "domain_id"))
        object.__setattr__(self, "version", _identifier(self.version, "version"))
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label.strip()) > 500:
            raise ValueError("label is invalid")
        object.__setattr__(self, "label", " ".join(self.label.split()))
        for name, maximum in (("supported_mime_types", 64), ("supported_languages", 64), ("capabilities", 64)):
            values = getattr(self, name)
            if isinstance(values, (str, bytes, bytearray)) or len(values) > maximum:
                raise ValueError(f"{name} is invalid")
            object.__setattr__(self, name, tuple(dict.fromkeys(_identifier(str(item), name) for item in values)))
        if not isinstance(self.metadata_schema, Mapping):
            raise ValueError("metadata_schema must be a mapping")
        try:
            encoded = _canonical(dict(self.metadata_schema))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata_schema must contain strict JSON values") from exc
        if len(encoded) > 256_000:
            raise ValueError("metadata_schema exceeds the size limit")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class DomainQueryFeatures:
    domain_id: str
    scores: Mapping[str, float]
    filters: Mapping[str, Any] = field(default_factory=dict)
    normalized_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain_id", _identifier(self.domain_id, "domain_id"))
        if not isinstance(self.scores, Mapping) or len(self.scores) > 128:
            raise ValueError("scores must be a bounded mapping")
        normalized_scores: dict[str, float] = {}
        for key, value in self.scores.items():
            name = _identifier(str(key), "score key")
            if isinstance(value, bool):
                raise ValueError("domain feature scores must be numeric")
            parsed = float(value)
            if not 0.0 <= parsed <= 1.0:
                raise ValueError("domain feature scores must lie in [0,1]")
            normalized_scores[name] = parsed
        object.__setattr__(self, "scores", normalized_scores)
        if not isinstance(self.filters, Mapping) or len(self.filters) > 128:
            raise ValueError("filters must be a bounded mapping")
        _canonical(dict(self.filters))
        if len(self.normalized_terms) > 256:
            raise ValueError("normalized_terms exceed the item limit")
        terms: list[str] = []
        for item in self.normalized_terms:
            if not isinstance(item, str) or not item.strip() or len(item.strip()) > 500:
                raise ValueError("normalized term is invalid")
            terms.append(" ".join(item.split()))
        object.__setattr__(self, "normalized_terms", tuple(dict.fromkeys(terms)))


class ScientificDomainAdapter(Protocol):
    @property
    def descriptor(self) -> DomainDescriptor: ...

    def normalize_metadata(self, metadata: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def query_features(self, query: str) -> DomainQueryFeatures: ...

    def unit_aliases(self) -> Mapping[str, str]: ...

    def enrich_document_graph(
        self,
        document: ScientificDocumentIR,
    ) -> tuple[Sequence[GraphNode], Sequence[GraphEdge]]: ...

    def report_fields(self) -> Sequence[str]: ...


class DomainAdapterRegistry:
    """Explicit versioned registry; no dynamic filesystem/plugin imports."""

    def __init__(self) -> None:
        self._adapters: dict[str, ScientificDomainAdapter] = {}
        self._active: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(self, adapter: ScientificDomainAdapter, *, activate: bool = False) -> None:
        descriptor = adapter.descriptor
        if not isinstance(descriptor, DomainDescriptor):
            raise ValueError("domain adapter must expose a DomainDescriptor")
        key = f"{descriptor.domain_id}@{descriptor.version}"
        with self._lock:
            existing = self._adapters.get(key)
            if existing is not None and existing is not adapter:
                if existing.descriptor.fingerprint != descriptor.fingerprint:
                    raise ValueError("a different domain adapter is already registered")
                return
            if existing is None and len(self._adapters) >= _MAX_ADAPTERS:
                raise RuntimeError("domain adapter registry capacity reached")
            self._adapters[key] = adapter
            if activate:
                self._active[descriptor.domain_id] = key

    def activate(self, domain_id: str, version: str) -> DomainDescriptor:
        identifier = _identifier(domain_id, "domain_id")
        key = f"{identifier}@{_identifier(version, 'version')}"
        with self._lock:
            adapter = self._adapters.get(key)
            if adapter is None:
                raise KeyError(key)
            self._active[identifier] = key
            return adapter.descriptor

    def active(self, domain_id: str) -> ScientificDomainAdapter | None:
        identifier = _identifier(domain_id, "domain_id")
        with self._lock:
            key = self._active.get(identifier)
            return self._adapters.get(key) if key else None

    def route(self, query: str, *, minimum_score: float = 0.5) -> tuple[tuple[ScientificDomainAdapter, DomainQueryFeatures], ...]:
        if not isinstance(query, str) or not query.strip() or len(query) > 20_000:
            raise ValueError("query is invalid")
        if isinstance(minimum_score, bool):
            raise ValueError("minimum_score must be numeric")
        threshold = float(minimum_score)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("minimum_score must lie in [0,1]")
        with self._lock:
            adapters = [self._adapters[key] for key in sorted(set(self._active.values()))]
        output: list[tuple[ScientificDomainAdapter, DomainQueryFeatures]] = []
        for adapter in adapters:
            try:
                features = adapter.query_features(query)
            except Exception:
                continue
            if not isinstance(features, DomainQueryFeatures) or features.domain_id != adapter.descriptor.domain_id:
                continue
            score = max(features.scores.values(), default=0.0)
            if score >= threshold:
                output.append((adapter, features))
        output.sort(key=lambda item: (-max(item[1].scores.values(), default=0.0), item[0].descriptor.domain_id))
        return tuple(output)

    def descriptors(self) -> tuple[DomainDescriptor, ...]:
        with self._lock:
            return tuple(sorted((adapter.descriptor for adapter in self._adapters.values()), key=lambda item: (item.domain_id, item.version)))

    @property
    def fingerprint(self) -> str:
        with self._lock:
            payload = {
                "active": dict(sorted(self._active.items())),
                "descriptors": [asdict(item) for item in self.descriptors()],
            }
        return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = [
    "DomainAdapterRegistry",
    "DomainDescriptor",
    "DomainQueryFeatures",
    "ScientificDomainAdapter",
]
