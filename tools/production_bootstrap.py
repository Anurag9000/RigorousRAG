"""Explicit bootstrap helpers for trusted production provider injection.

These helpers register already-created provider objects. They do not read credentials,
construct network clients, import arbitrary modules, download models or contact remote
services. Deployments retain full control over client creation and secret management.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from tools.runtime_providers import RuntimeProviderBinding, RuntimeProviderRegistry, runtime_providers


def _register(
    provider_id: str,
    provider: Any,
    *,
    capabilities: tuple[str, ...],
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    registry: RuntimeProviderRegistry | None = None,
) -> RuntimeProviderBinding:
    target = registry or runtime_providers
    return target.register(
        provider_id,
        provider,
        capabilities=capabilities,
        version=version,
        metadata=metadata,
        health_check=health_check,
    )


def register_postgres_connection_factory(connection_factory: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(connection_factory):
        raise TypeError("connection_factory must be callable")
    return _register("postgres.connection_factory", connection_factory, capabilities=("storage.metadata.postgres", "research.workspace.postgres"), version=version, health_check=health_check, registry=registry)


def register_nli_provider(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(getattr(provider, "score", None)):
        raise TypeError("NLI provider must implement score(claim, evidence)")
    return _register("nli.provider", provider, capabilities=("nli.claim_entailment",), version=version, health_check=health_check, registry=registry)


def register_multimodal_backend(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(getattr(provider, "encode_multimodal", None)):
        raise TypeError("multimodal backend must implement encode_multimodal")
    return _register("multimodal.backend", provider, capabilities=("retrieval.multimodal",), version=version, health_check=health_check, registry=registry)


def register_page_late_interaction_backend(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(getattr(provider, "embed_query", None)) or not callable(getattr(provider, "embed_page", None)):
        raise TypeError("page backend must implement embed_query and embed_page")
    return _register("page_late_interaction.backend", provider, capabilities=("retrieval.page_late_interaction",), version=version, health_check=health_check, registry=registry)


def register_adaptive_policy_provider(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(getattr(provider, "decide", None)):
        raise TypeError("adaptive policy provider must implement decide")
    if not isinstance(getattr(provider, "policy_id", None), str) or not isinstance(getattr(provider, "version", None), str):
        raise TypeError("adaptive policy provider must expose policy_id and version")
    return _register("adaptive_policy.provider", provider, capabilities=("policy.learned_adaptive",), version=version, health_check=health_check, registry=registry)


def register_replay_cipher(cipher: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(getattr(cipher, "seal", None)) or not callable(getattr(cipher, "open", None)):
        raise TypeError("replay cipher must implement seal and open")
    if not isinstance(getattr(cipher, "key_id", None), str):
        raise TypeError("replay cipher must expose key_id")
    return _register("replay.cipher", cipher, capabilities=("replay.encrypted_recipe",), version=version, health_check=health_check, registry=registry)


def register_s3_object_store(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    for name in ("put", "get", "head", "delete"):
        if not callable(getattr(provider, name, None)):
            raise TypeError("object store must implement put/get/head/delete")
    return _register("object_store.s3", provider, capabilities=("storage.object.s3",), version=version, health_check=health_check, registry=registry)


def register_redis_queue(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    for name in ("enqueue", "claim", "ack", "retry"):
        if not callable(getattr(provider, name, None)):
            raise TypeError("queue provider must implement enqueue/claim/ack/retry")
    return _register("queue.redis", provider, capabilities=("queue.redis",), version=version, health_check=health_check, registry=registry)


def register_secret_provider(provider: Any, *, version: str = "1.0.0", health_check: Callable[[], bool] | bool = True, registry: RuntimeProviderRegistry | None = None) -> RuntimeProviderBinding:
    if not callable(getattr(provider, "get", None)):
        raise TypeError("secret provider must implement get(reference)")
    return _register("secret.provider", provider, capabilities=("secret.external",), version=version, health_check=health_check, registry=registry)


__all__ = [
    "register_adaptive_policy_provider",
    "register_multimodal_backend",
    "register_nli_provider",
    "register_page_late_interaction_backend",
    "register_postgres_connection_factory",
    "register_redis_queue",
    "register_replay_cipher",
    "register_s3_object_store",
    "register_secret_provider",
]
