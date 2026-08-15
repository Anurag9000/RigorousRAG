"""Versioned hierarchical runtime configuration with secret references and overlays.

Configuration values are explicit typed data, while secret values remain references
resolved only by a ``SecretProvider``.  Environment overlays are allowlisted by schema;
unknown/deprecated fields fail closed instead of silently changing runtime behavior.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

from tools.production_runtime import SecretProvider


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class SecretRef:
    reference: str

    def __post_init__(self) -> None:
        ref = _text(self.reference, "secret reference", 500)
        if "://" not in ref:
            raise ValueError("secret reference must include a provider scheme")
        object.__setattr__(self, "reference", ref)

    def resolve(self, provider: SecretProvider) -> str:
        return provider.get(self.reference)


@dataclass(frozen=True)
class StorageConfig:
    vector_backend: str = "local"
    sparse_backend: str = "sqlite"
    graph_backend: str = "local"
    object_backend: str = "local"
    metadata_backend: str = "sqlite"
    connection_secret: SecretRef | None = None

    def __post_init__(self) -> None:
        for name in ("vector_backend", "sparse_backend", "graph_backend", "object_backend", "metadata_backend"):
            object.__setattr__(self, name, _text(getattr(self, name), name, 100).lower())
        if self.connection_secret is not None and not isinstance(self.connection_secret, SecretRef):
            raise ValueError("connection_secret must be SecretRef")


@dataclass(frozen=True)
class RetrievalConfig:
    default_strategy: str = "adaptive"
    max_candidates: int = 200
    max_evidence: int = 40
    max_hops: int = 8
    max_wall_ms: int = 30_000
    enable_multimodal: bool = True
    enable_graph: bool = True
    enable_web: bool = False
    policy_capability_id: str = ""

    def __post_init__(self) -> None:
        strategy = _text(self.default_strategy, "default_strategy", 64).lower()
        if strategy not in {"single", "adaptive", "multihop", "heterogeneous"}:
            raise ValueError("unsupported default retrieval strategy")
        object.__setattr__(self, "default_strategy", strategy)
        for name, minimum, maximum in (("max_candidates", 1, 5000), ("max_evidence", 1, 500), ("max_hops", 1, 32), ("max_wall_ms", 100, 3_600_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is invalid")
        for name in ("enable_multimodal", "enable_graph", "enable_web"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        object.__setattr__(self, "policy_capability_id", _text(self.policy_capability_id, "policy_capability_id", 256, allow_empty=True))


@dataclass(frozen=True)
class SecurityConfig:
    parser_sandbox_required: bool = True
    malware_scan_required: bool = False
    remote_egress_enabled: bool = False
    max_upload_bytes: int = 100_000_000
    secret_cache_ttl_seconds: int = 300

    def __post_init__(self) -> None:
        for name in ("parser_sandbox_required", "malware_scan_required", "remote_egress_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if isinstance(self.max_upload_bytes, bool) or not isinstance(self.max_upload_bytes, int) or not 1 <= self.max_upload_bytes <= 10**12:
            raise ValueError("max_upload_bytes is invalid")
        if isinstance(self.secret_cache_ttl_seconds, bool) or not isinstance(self.secret_cache_ttl_seconds, int) or not 0 <= self.secret_cache_ttl_seconds <= 86_400:
            raise ValueError("secret_cache_ttl_seconds is invalid")


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: str
    environment: str
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    instance_id: str = "rigorousrag"

    def __post_init__(self) -> None:
        version = _text(self.schema_version, "schema_version", 32)
        if version != "1.0.0":
            raise ValueError("unsupported runtime configuration schema_version")
        object.__setattr__(self, "schema_version", version)
        env = _text(self.environment, "environment", 32).lower()
        if env not in {"development", "test", "staging", "production"}:
            raise ValueError("unsupported environment")
        object.__setattr__(self, "environment", env)
        if not isinstance(self.storage, StorageConfig) or not isinstance(self.retrieval, RetrievalConfig) or not isinstance(self.security, SecurityConfig):
            raise ValueError("runtime configuration sections are invalid")
        object.__setattr__(self, "instance_id", _text(self.instance_id, "instance_id", 128))

    @property
    def fingerprint(self) -> str:
        payload = asdict(self)
        connection = payload.get("storage", {}).get("connection_secret")
        if connection:
            payload["storage"]["connection_secret"] = {"reference": connection["reference"]}
        return hashlib.sha256(_canonical(payload)).hexdigest()


_ALLOWED_ROOT = {"schema_version", "environment", "storage", "retrieval", "security", "instance_id"}
_SECTION_FIELDS = {
    "storage": set(StorageConfig.__dataclass_fields__),
    "retrieval": set(RetrievalConfig.__dataclass_fields__),
    "security": set(SecurityConfig.__dataclass_fields__),
}


def runtime_config_from_mapping(value: Mapping[str, Any]) -> RuntimeConfig:
    if not isinstance(value, Mapping):
        raise ValueError("configuration must be a mapping")
    unknown = set(value) - _ALLOWED_ROOT
    if unknown:
        raise ValueError(f"unknown runtime configuration fields: {sorted(unknown)!r}")
    sections: dict[str, Any] = {}
    for section, cls in (("storage", StorageConfig), ("retrieval", RetrievalConfig), ("security", SecurityConfig)):
        raw = value.get(section, {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"{section} must be a mapping")
        unknown_section = set(raw) - _SECTION_FIELDS[section]
        if unknown_section:
            raise ValueError(f"unknown {section} configuration fields: {sorted(unknown_section)!r}")
        prepared = dict(raw)
        if section == "storage" and isinstance(prepared.get("connection_secret"), str):
            prepared["connection_secret"] = SecretRef(prepared["connection_secret"])
        sections[section] = cls(**prepared)
    return RuntimeConfig(
        schema_version=value.get("schema_version", "1.0.0"),
        environment=value.get("environment", "development"),
        storage=sections["storage"],
        retrieval=sections["retrieval"],
        security=sections["security"],
        instance_id=value.get("instance_id", "rigorousrag"),
    )


_ENV_OVERLAYS = {
    "RIGOROUSRAG_ENVIRONMENT": ("environment", str),
    "RIGOROUSRAG_VECTOR_BACKEND": ("storage.vector_backend", str),
    "RIGOROUSRAG_SPARSE_BACKEND": ("storage.sparse_backend", str),
    "RIGOROUSRAG_GRAPH_BACKEND": ("storage.graph_backend", str),
    "RIGOROUSRAG_OBJECT_BACKEND": ("storage.object_backend", str),
    "RIGOROUSRAG_METADATA_BACKEND": ("storage.metadata_backend", str),
    "RIGOROUSRAG_CONNECTION_SECRET": ("storage.connection_secret", SecretRef),
    "RIGOROUSRAG_DEFAULT_STRATEGY": ("retrieval.default_strategy", str),
    "RIGOROUSRAG_POLICY_CAPABILITY": ("retrieval.policy_capability_id", str),
}


def apply_environment_overlays(config: RuntimeConfig, *, environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    env = os.environ if environ is None else environ
    root: dict[str, Any] = asdict(config)
    connection = root["storage"].get("connection_secret")
    if connection:
        root["storage"]["connection_secret"] = connection["reference"]
    for variable, (path, converter) in _ENV_OVERLAYS.items():
        if variable not in env:
            continue
        raw = env[variable]
        section_and_field = path.split(".")
        converted = converter(raw)
        if len(section_and_field) == 1:
            root[section_and_field[0]] = converted.reference if isinstance(converted, SecretRef) else converted
        else:
            section, name = section_and_field
            root[section][name] = converted.reference if isinstance(converted, SecretRef) else converted
    return runtime_config_from_mapping(root)


__all__ = [
    "RetrievalConfig", "RuntimeConfig", "SecretRef", "SecurityConfig", "StorageConfig",
    "apply_environment_overlays", "runtime_config_from_mapping",
]
