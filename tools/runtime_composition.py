"""Trusted production composition root for RigorousRAG runtime services.

This module converts typed ``RuntimeConfig`` into explicit capability/domain selections.
It does not dynamically import untrusted plugins, contact providers, download models or
silently enable optional capabilities. Provider objects may be injected by trusted
application bootstrap code and are represented through runtime-health signals.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from tools.capability_registry import CapabilityRegistry
from tools.default_capability_registry import build_default_capability_registry
from tools.default_domain_registry import build_default_domain_registry
from tools.domain_adapter import DomainAdapterRegistry
from tools.runtime_config import RuntimeConfig, apply_environment_overlays, runtime_config_from_mapping
from tools.runtime_providers import runtime_providers

_MAX_CONFIG_JSON_BYTES = 256_000


def _bool_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config(*, environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Load one strict configuration from an optional bounded JSON document plus overlays."""

    env = os.environ if environ is None else environ
    raw = env.get("RIGOROUSRAG_RUNTIME_CONFIG_JSON", "")
    if raw:
        if len(raw.encode("utf-8")) > _MAX_CONFIG_JSON_BYTES:
            raise ValueError("RIGOROUSRAG_RUNTIME_CONFIG_JSON exceeds the size limit")
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("RIGOROUSRAG_RUNTIME_CONFIG_JSON must contain one JSON object")
        base = runtime_config_from_mapping(parsed)
    else:
        base = runtime_config_from_mapping(
            {
                "schema_version": "1.0.0",
                "environment": env.get("RIGOROUSRAG_ENVIRONMENT", "development"),
            }
        )
    return apply_environment_overlays(base, environ=env)


@dataclass(frozen=True)
class RuntimeComposition:
    config: RuntimeConfig
    capabilities: CapabilityRegistry
    domains: DomainAdapterRegistry
    selected_capabilities: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.config, RuntimeConfig):
            raise TypeError("config must be RuntimeConfig")
        if not isinstance(self.capabilities, CapabilityRegistry):
            raise TypeError("capabilities must be CapabilityRegistry")
        if not isinstance(self.domains, DomainAdapterRegistry):
            raise TypeError("domains must be DomainAdapterRegistry")
        if not isinstance(self.selected_capabilities, Mapping):
            raise TypeError("selected_capabilities must be a mapping")


def _storage_capability(kind: str, value: str) -> str:
    normalized = value.strip().lower()
    options = {
        ("metadata", "sqlite"): "storage.metadata.sqlite",
        ("metadata", "postgres"): "storage.metadata.postgres",
        ("metadata", "postgresql"): "storage.metadata.postgres",
        ("object", "local"): "storage.object.local",
        ("object", "s3"): "storage.object.s3",
        ("object", "s3-compatible"): "storage.object.s3",
    }
    capability = options.get((kind, normalized))
    if capability is None:
        raise ValueError(f"unsupported {kind} storage backend: {normalized}")
    return capability


def _workspace_capability(metadata_backend: str) -> str:
    normalized = metadata_backend.strip().lower()
    if normalized == "sqlite":
        return "research.workspace"
    if normalized in {"postgres", "postgresql"}:
        return "research.workspace.postgres"
    raise ValueError(f"unsupported workspace metadata backend: {normalized}")


def _selected_healthy(registry: CapabilityRegistry, capability_id: str) -> str:
    resolution = registry.resolve(capability_id, allow_fallback=True)
    return resolution.selected.capability_id


def build_runtime_composition(
    config: RuntimeConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    runtime_health: Mapping[str, Any] | None = None,
) -> RuntimeComposition:
    env = os.environ if environ is None else environ
    selected_config = config or load_runtime_config(environ=env)

    health: dict[str, Any] = dict(runtime_providers.capability_health())
    health.setdefault("nli.claim_entailment", _bool_env(env.get("RIGOROUSRAG_NLI_CONFIGURED")))
    health.setdefault(
        "retrieval.page_late_interaction",
        selected_config.retrieval.enable_multimodal
        and _bool_env(env.get("RIGOROUSRAG_PAGE_MODEL_CONFIGURED")),
    )
    health.setdefault(
        "retrieval.multimodal",
        selected_config.retrieval.enable_multimodal
        and _bool_env(env.get("RIGOROUSRAG_MULTIMODAL_MODEL_CONFIGURED")),
    )
    health.setdefault(
        "policy.learned_adaptive",
        bool(selected_config.retrieval.policy_capability_id)
        and _bool_env(env.get("RIGOROUSRAG_LEARNED_POLICY_CONFIGURED")),
    )
    health.setdefault(
        "storage.metadata.postgres",
        selected_config.storage.metadata_backend in {"postgres", "postgresql"}
        and _bool_env(env.get("RIGOROUSRAG_POSTGRES_CONFIGURED")),
    )
    health.setdefault(
        "storage.object.s3",
        selected_config.storage.object_backend in {"s3", "s3-compatible"}
        and _bool_env(env.get("RIGOROUSRAG_S3_CONFIGURED")),
    )
    health.setdefault("queue.redis", _bool_env(env.get("RIGOROUSRAG_REDIS_CONFIGURED")))
    health.setdefault("secret.external", _bool_env(env.get("RIGOROUSRAG_SECRET_PROVIDER_CONFIGURED")))
    if runtime_health:
        health.update(dict(runtime_health))

    capabilities = build_default_capability_registry(runtime_health=health)
    domains = build_default_domain_registry()

    strategy_map = {
        "single": "retrieval.hybrid",
        "adaptive": "policy.deterministic_adaptive",
        "multihop": "planner.multihop",
        "heterogeneous": "planner.multihop",
    }
    retrieval_id = strategy_map[selected_config.retrieval.default_strategy]
    policy_id = selected_config.retrieval.policy_capability_id.strip() or retrieval_id

    metadata_id = _storage_capability("metadata", selected_config.storage.metadata_backend)
    object_id = _storage_capability("object", selected_config.storage.object_backend)
    workspace_id = _workspace_capability(selected_config.storage.metadata_backend)
    selected: dict[str, str] = {
        "retrieval": _selected_healthy(capabilities, retrieval_id),
        "policy": _selected_healthy(capabilities, policy_id),
        "metadata_storage": _selected_healthy(capabilities, metadata_id),
        "object_storage": _selected_healthy(capabilities, object_id),
        "workspace": _selected_healthy(capabilities, workspace_id),
        "domain": _selected_healthy(capabilities, "domain.hydrology"),
    }
    if selected_config.retrieval.enable_graph:
        selected["graph"] = _selected_healthy(capabilities, "retrieval.graph")
    if selected_config.retrieval.enable_multimodal and health.get("retrieval.multimodal"):
        selected["multimodal"] = _selected_healthy(capabilities, "retrieval.multimodal")
    if selected_config.retrieval.enable_multimodal and health.get("retrieval.page_late_interaction"):
        selected["page_late_interaction"] = _selected_healthy(capabilities, "retrieval.page_late_interaction")
    if health.get("nli.claim_entailment"):
        selected["claim_entailment"] = _selected_healthy(capabilities, "nli.claim_entailment")

    return RuntimeComposition(
        config=selected_config,
        capabilities=capabilities,
        domains=domains,
        selected_capabilities=selected,
    )


__all__ = ["RuntimeComposition", "build_runtime_composition", "load_runtime_config"]
