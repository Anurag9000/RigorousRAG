"""Explicit bootstrap helpers for trusted multimodal runtime providers.

These helpers register already-constructed provider objects only. They never import model
SDKs, load checkpoints, discover credentials, or construct network clients.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from tools.multimodal_runtime import OBJECT_STORE_PROVIDER_ID, PAGE_BACKEND_PROVIDER_ID
from tools.runtime_providers import RuntimeProviderBinding, RuntimeProviderRegistry, runtime_providers


def _health(value: Callable[[], bool] | bool) -> Callable[[], bool] | bool:
    if not isinstance(value, bool) and not callable(value):
        raise TypeError("health_check must be boolean or callable")
    return value


def _metadata(value: Mapping[str, str] | None) -> Mapping[str, str]:
    return {} if value is None else value


def register_multimodal_object_store(
    provider: Any,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    if provider is None:
        raise ValueError("multimodal object-store provider is required")
    required = ("put", "get", "head", "delete")
    if any(not callable(getattr(provider, name, None)) for name in required):
        raise TypeError("multimodal object-store provider does not satisfy the ObjectStore contract")
    return registry.register(
        OBJECT_STORE_PROVIDER_ID,
        provider,
        capabilities=("object_store", "multimodal_retrieval", "page_native_retrieval"),
        version=version,
        metadata=_metadata(metadata),
        health_check=_health(health_check),
        replace=replace,
    )


def register_multimodal_page_backend(
    provider: Any,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    if provider is None:
        raise ValueError("multimodal page backend is required")
    model_id = getattr(provider, "model_id", None)
    if not isinstance(model_id, str) or not model_id.strip():
        raise TypeError("multimodal page backend must expose a non-empty model_id")
    if not callable(getattr(provider, "embed_query", None)) or not callable(getattr(provider, "embed_page", None)):
        raise TypeError("multimodal page backend does not satisfy the PageEmbeddingBackend contract")
    details = dict(_metadata(metadata))
    details.setdefault("model_id", model_id.strip())
    return registry.register(
        PAGE_BACKEND_PROVIDER_ID,
        provider,
        capabilities=("multimodal_retrieval", "page_native_retrieval", "page_embedding"),
        version=version,
        metadata=details,
        health_check=_health(health_check),
        replace=replace,
    )


__all__ = ["register_multimodal_object_store", "register_multimodal_page_backend"]
