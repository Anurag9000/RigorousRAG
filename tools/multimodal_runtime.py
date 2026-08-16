"""Trusted provider composition for page-native multimodal retrieval.

No provider discovery, model loading, credential lookup or implicit ephemeral fallback occurs
here. Production bootstrap explicitly registers trusted objects under the fixed provider
IDs below. Runtime status exposes binding fingerprints/health only, never provider objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tools.page_embedding_object_store import PageEmbeddingObjectRepository
from tools.runtime_providers import RuntimeProviderBinding, RuntimeProviderRegistry

OBJECT_STORE_PROVIDER_ID = "multimodal.object_store"
PAGE_BACKEND_PROVIDER_ID = "multimodal.page_backend"


def _object_store(value: Any) -> bool:
    return all(callable(getattr(value, name, None)) for name in ("put", "get", "head", "delete"))


def _page_backend(value: Any) -> bool:
    return (
        isinstance(getattr(value, "model_id", None), str)
        and bool(getattr(value, "model_id", "").strip())
        and callable(getattr(value, "embed_query", None))
        and callable(getattr(value, "embed_page", None))
    )


def _binding(bindings: tuple[RuntimeProviderBinding, ...], provider_id: str) -> RuntimeProviderBinding | None:
    return next((item for item in bindings if item.provider_id == provider_id), None)


@dataclass(frozen=True)
class MultimodalRuntime:
    object_repository: PageEmbeddingObjectRepository | None
    page_backend: Any | None
    object_store_binding_sha256: str
    page_backend_binding_sha256: str
    object_store_healthy: bool
    page_backend_healthy: bool
    page_model_id: str

    @property
    def retrieval_ready(self) -> bool:
        return bool(self.object_repository is not None and self.page_backend is not None and self.object_store_healthy and self.page_backend_healthy)

    def status(self) -> Mapping[str, Any]:
        return {
            "retrieval_ready": self.retrieval_ready,
            "object_store_configured": self.object_repository is not None,
            "object_store_healthy": self.object_store_healthy,
            "object_store_binding_sha256": self.object_store_binding_sha256 or None,
            "page_backend_configured": self.page_backend is not None,
            "page_backend_healthy": self.page_backend_healthy,
            "page_backend_binding_sha256": self.page_backend_binding_sha256 or None,
            "page_model_id": self.page_model_id or None,
            "ephemeral_fallback": False,
        }


def build_multimodal_runtime(providers: RuntimeProviderRegistry) -> MultimodalRuntime:
    if not isinstance(providers, RuntimeProviderRegistry):
        raise TypeError("providers must be RuntimeProviderRegistry")
    bindings = providers.bindings()

    object_binding = _binding(bindings, OBJECT_STORE_PROVIDER_ID)
    object_provider = providers.get(OBJECT_STORE_PROVIDER_ID) if object_binding is not None else None
    object_healthy = bool(object_binding is not None and providers.healthy(OBJECT_STORE_PROVIDER_ID))
    if object_provider is not None and not _object_store(object_provider):
        raise RuntimeError("multimodal.object_store does not satisfy the ObjectStore contract")
    object_repository = PageEmbeddingObjectRepository(object_provider) if object_provider is not None and object_healthy else None

    page_binding = _binding(bindings, PAGE_BACKEND_PROVIDER_ID)
    page_provider = providers.get(PAGE_BACKEND_PROVIDER_ID) if page_binding is not None else None
    page_healthy = bool(page_binding is not None and providers.healthy(PAGE_BACKEND_PROVIDER_ID))
    if page_provider is not None and not _page_backend(page_provider):
        raise RuntimeError("multimodal.page_backend does not satisfy the PageEmbeddingBackend contract")
    page_backend = page_provider if page_provider is not None and page_healthy else None

    return MultimodalRuntime(
        object_repository=object_repository,
        page_backend=page_backend,
        object_store_binding_sha256=object_binding.fingerprint if object_binding is not None else "",
        page_backend_binding_sha256=page_binding.fingerprint if page_binding is not None else "",
        object_store_healthy=object_healthy,
        page_backend_healthy=page_healthy,
        page_model_id=str(getattr(page_provider, "model_id", "")).strip() if page_provider is not None else "",
    )


__all__ = [
    "MultimodalRuntime",
    "OBJECT_STORE_PROVIDER_ID",
    "PAGE_BACKEND_PROVIDER_ID",
    "build_multimodal_runtime",
]
