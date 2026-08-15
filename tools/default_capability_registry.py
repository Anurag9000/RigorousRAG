"""Trusted built-in capability catalog for RigorousRAG.

The catalog separates implementation availability from deployment readiness. Optional
provider-backed capabilities (for example an injected NLI or page model) can be declared
without being reported healthy until trusted application composition supplies a positive
runtime-health signal. No model is downloaded and no provider is contacted here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tools.capability_registry import (
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityRegistry,
    ResourceEnvelope,
)


def _health_check(capability_id: str, runtime_health: Mapping[str, Any]):
    def check(_descriptor: CapabilityDescriptor) -> CapabilityHealth:
        value = runtime_health.get(capability_id, True)
        if isinstance(value, CapabilityHealth):
            return value
        if isinstance(value, bool):
            return CapabilityHealth(value, "available" if value else "provider_not_configured")
        if callable(value):
            result = value()
            if isinstance(result, CapabilityHealth):
                return result
            if isinstance(result, bool):
                return CapabilityHealth(result, "available" if result else "provider_not_configured")
        return CapabilityHealth(False, "invalid_runtime_health_signal")

    return check


def _descriptor(
    capability_id: str,
    *,
    kind: str,
    provider: str,
    modalities: tuple[str, ...] = ("text",),
    dependencies: tuple[str, ...] = (),
    fallbacks: tuple[str, ...] = (),
    trust_level: str = "local",
    max_calls: int = 1,
    max_latency_ms: float = 0.0,
    max_input_bytes: int = 0,
    max_output_bytes: int = 0,
    max_tokens: int = 0,
    max_cost: float = 0.0,
    max_concurrency: int = 1,
) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        capability_id=capability_id,
        version="1.0.0",
        kind=kind,
        provider=provider,
        modalities=modalities,
        dependencies=dependencies,
        fallbacks=fallbacks,
        trust_level=trust_level,
        resources=ResourceEnvelope(
            max_calls=max_calls,
            max_latency_ms=max_latency_ms,
            max_input_bytes=max_input_bytes,
            max_output_bytes=max_output_bytes,
            max_tokens=max_tokens,
            max_cost=max_cost,
            max_concurrency=max_concurrency,
        ),
    )


def build_default_capability_registry(
    *,
    runtime_health: Mapping[str, Any] | None = None,
) -> CapabilityRegistry:
    """Build the trusted built-in capability catalog.

    ``runtime_health`` is supplied only by trusted application composition. Missing
    values default to healthy for dependency-free local implementations. Optional model
    capabilities are explicitly seeded unhealthy below unless the caller overrides them.
    """

    health: dict[str, Any] = {
        "nli.claim_entailment": False,
        "retrieval.page_late_interaction": False,
        "retrieval.multimodal": False,
        "policy.learned_adaptive": False,
    }
    if runtime_health:
        health.update(dict(runtime_health))

    descriptors = (
        _descriptor(
            "storage.metadata.sqlite",
            kind="storage",
            provider="tools.research_workspace_sqlite",
            modalities=("text",),
            max_concurrency=64,
        ),
        _descriptor(
            "storage.object.local",
            kind="storage",
            provider="tools.production_runtime",
            modalities=("text", "image", "page_image", "table", "figure", "formula", "graph", "raster", "timeseries", "geospatial"),
            max_concurrency=32,
        ),
        _descriptor(
            "queue.local",
            kind="queue",
            provider="tools.production_runtime",
            modalities=("text",),
            max_concurrency=64,
        ),
        _descriptor(
            "retrieval.dense",
            kind="embedding",
            provider="tools.rag",
            modalities=("text",),
            max_calls=64,
            max_input_bytes=5_000_000,
            max_output_bytes=5_000_000,
            max_concurrency=32,
        ),
        _descriptor(
            "retrieval.sparse",
            kind="sparse_retriever",
            provider="tools.sparse_index",
            modalities=("text",),
            max_calls=64,
            max_input_bytes=5_000_000,
            max_output_bytes=5_000_000,
            max_concurrency=32,
        ),
        _descriptor(
            "retrieval.hybrid",
            kind="other",
            provider="tools.hybrid_retrieval",
            dependencies=("retrieval.dense", "retrieval.sparse"),
            max_calls=64,
            max_concurrency=32,
        ),
        _descriptor(
            "retrieval.contextual",
            kind="other",
            provider="tools.contextual_retrieval",
            dependencies=("retrieval.hybrid",),
            max_calls=64,
            max_concurrency=32,
        ),
        _descriptor(
            "retrieval.graph",
            kind="graph_backend",
            provider="tools.graph_reasoning",
            modalities=("text", "graph"),
            max_calls=32,
            max_concurrency=16,
        ),
        _descriptor(
            "planner.multihop",
            kind="planner",
            provider="tools.multihop_retrieval",
            modalities=("text", "graph"),
            dependencies=("retrieval.hybrid",),
            max_calls=32,
            max_concurrency=16,
        ),
        _descriptor(
            "policy.deterministic_adaptive",
            kind="router",
            provider="tools.adaptive_rag_tool",
            dependencies=("retrieval.hybrid",),
            max_calls=64,
            max_concurrency=32,
        ),
        _descriptor(
            "policy.learned_adaptive",
            kind="router",
            provider="tools.learned_retrieval_policy",
            dependencies=("policy.deterministic_adaptive",),
            fallbacks=("policy.deterministic_adaptive",),
            max_calls=64,
            max_concurrency=32,
        ),
        _descriptor(
            "retrieval.multimodal",
            kind="multimodal_retriever",
            provider="tools.hf_multimodal_backend",
            modalities=("text", "image"),
            max_calls=32,
            max_concurrency=8,
        ),
        _descriptor(
            "retrieval.page_late_interaction",
            kind="late_interaction",
            provider="tools.transformer_page_late_interaction",
            modalities=("text", "page_image"),
            max_calls=32,
            max_concurrency=8,
        ),
        _descriptor(
            "nli.claim_entailment",
            kind="reranker",
            provider="tools.transformer_entailment",
            modalities=("text",),
            max_calls=128,
            max_concurrency=16,
        ),
        _descriptor(
            "domain.hydrology",
            kind="domain_adapter",
            provider="tools.hydrology_domain_adapter",
            modalities=("text", "table", "graph", "raster", "timeseries", "geospatial"),
            max_calls=64,
            max_concurrency=16,
        ),
        _descriptor(
            "research.workspace",
            kind="storage",
            provider="tools.research_workspace_sqlite",
            modalities=("text",),
            dependencies=("storage.metadata.sqlite",),
            max_concurrency=64,
        ),
    )

    registry = CapabilityRegistry()
    for descriptor in descriptors:
        registry.register(
            descriptor,
            activate=True,
            health_check=_health_check(descriptor.capability_id, health),
        )
    return registry


__all__ = ["build_default_capability_registry"]
