"""Trusted runtime composition helpers for encrypted research replay recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.capability_registry import (
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityRegistry,
    ResourceEnvelope,
)
from tools.replay_recipe_store import EncryptedReplayRecipeStore
from tools.runtime_providers import RuntimeProviderRegistry, runtime_providers

_REPLAY_CAPABILITY_ID = "replay.encrypted_recipe"
_REPLAY_PROVIDER_ID = "replay.cipher"


def ensure_replay_capability(
    registry: CapabilityRegistry,
    *,
    providers: RuntimeProviderRegistry | None = None,
) -> CapabilityDescriptor:
    """Ensure encrypted replay appears in the catalog even when not configured."""

    if not isinstance(registry, CapabilityRegistry):
        raise TypeError("registry must be CapabilityRegistry")
    provider_registry = providers or runtime_providers
    existing = registry.active(_REPLAY_CAPABILITY_ID)
    if existing is not None:
        return existing
    descriptor = CapabilityDescriptor(
        capability_id=_REPLAY_CAPABILITY_ID,
        version="1.0.0",
        kind="storage",
        provider="tools.replay_recipe_store.EncryptedReplayRecipeStore",
        modalities=("text",),
        trust_level="local",
        resources=ResourceEnvelope(
            max_calls=64,
            max_input_bytes=100_000,
            max_output_bytes=100_000,
            max_concurrency=32,
        ),
    )

    def health(_descriptor: CapabilityDescriptor) -> CapabilityHealth:
        configured = provider_registry.get(_REPLAY_PROVIDER_ID) is not None
        healthy = configured and provider_registry.healthy(_REPLAY_PROVIDER_ID)
        return CapabilityHealth(
            healthy,
            "available" if healthy else "replay_cipher_not_configured",
        )

    registry.register(descriptor, activate=True, health_check=health)
    return descriptor


def build_replay_recipe_store(
    path: str | Path,
    *,
    providers: RuntimeProviderRegistry | None = None,
) -> EncryptedReplayRecipeStore | None:
    """Build the encrypted replay store only when a healthy cipher was injected."""

    provider_registry = providers or runtime_providers
    cipher: Any = provider_registry.get(_REPLAY_PROVIDER_ID)
    if cipher is None:
        return None
    if not provider_registry.healthy(_REPLAY_PROVIDER_ID):
        raise RuntimeError("configured replay cipher is unhealthy")
    return EncryptedReplayRecipeStore(path, cipher=cipher)


__all__ = [
    "build_replay_recipe_store",
    "ensure_replay_capability",
]
