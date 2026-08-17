"""Explicit bootstrap helpers for trusted signing/privacy/governance providers.

Only already-constructed provider objects are registered. This module performs contract
checks and records non-secret metadata; it never imports cloud SDKs, discovers credentials,
constructs network clients, or auto-discovers federated institutions.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from tools.attestation_keyring import RotatingManifestKeyring
from tools.federated_retrieval import FederatedCollection
from tools.hardened_parser_sandbox import SandboxDeploymentCapabilities
from tools.object_retention import ProviderRetentionCapabilities
from tools.runtime_providers import RuntimeProviderBinding, RuntimeProviderRegistry, runtime_providers

ATTESTATION_KEYRING_PROVIDER_ID = "attestation.keyring"
OBJECT_GOVERNANCE_PROVIDER_ID = "object.governance"
SECURE_AGGREGATION_PROVIDER_ID = "privacy.secure_aggregation"
HARDENED_PARSER_EXECUTOR_PROVIDER_ID = "parser.hardened_executor"
FEDERATED_PROVIDER_PREFIX = "federated.collection."


def _health(value: Callable[[], bool] | bool) -> Callable[[], bool] | bool:
    if not isinstance(value, bool) and not callable(value):
        raise TypeError("health_check must be boolean or callable")
    return value


def _metadata(value: Mapping[str, str] | None) -> dict[str, str]:
    return dict(value or {})


def register_attestation_keyring(
    provider: RotatingManifestKeyring,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    if not isinstance(provider, RotatingManifestKeyring):
        raise TypeError("provider must be RotatingManifestKeyring")
    details = _metadata(metadata)
    details.setdefault("active_key_id", provider.active_key_id or "verification-only")
    return registry.register(
        ATTESTATION_KEYRING_PROVIDER_ID,
        provider,
        capabilities=("manifest_attestation", "review_attestation", "public_key_verification", "key_rotation"),
        version=version,
        metadata=details,
        health_check=_health(health_check),
        replace=replace,
    )


def register_object_governance_provider(
    provider: Any,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, ProviderRetentionCapabilities):
        raise TypeError("object governance provider must expose ProviderRetentionCapabilities")
    for name in ("apply_retention", "request_deletion", "status"):
        if not callable(getattr(provider, name, None)):
            raise TypeError("object governance provider does not satisfy the required contract")
    details = _metadata(metadata)
    details.setdefault("provider", capabilities.provider_id)
    details.setdefault("capabilities_fingerprint", capabilities.fingerprint)
    return registry.register(
        OBJECT_GOVERNANCE_PROVIDER_ID,
        provider,
        capabilities=("object_retention", "legal_hold", "version_lock", "secure_deletion_status"),
        version=version,
        metadata=details,
        health_check=_health(health_check),
        replace=replace,
    )


def register_secure_aggregation_provider(
    provider: Any,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    provider_id = getattr(provider, "provider_id", None)
    if not isinstance(provider_id, str) or not provider_id.strip() or not callable(getattr(provider, "aggregate", None)):
        raise TypeError("secure aggregation provider must expose provider_id and aggregate")
    details = _metadata(metadata)
    details.setdefault("provider", provider_id.strip())
    return registry.register(
        SECURE_AGGREGATION_PROVIDER_ID,
        provider,
        capabilities=("secure_aggregation", "private_aggregate_learning"),
        version=version,
        metadata=details,
        health_check=_health(health_check),
        replace=replace,
    )


def register_hardened_parser_executor(
    provider: Any,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    executor_id = getattr(provider, "executor_id", None)
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(executor_id, str) or not executor_id.strip() or not isinstance(capabilities, SandboxDeploymentCapabilities):
        raise TypeError("hardened parser executor must expose executor_id and SandboxDeploymentCapabilities")
    if not callable(getattr(provider, "execute", None)):
        raise TypeError("hardened parser executor must expose execute")
    details = _metadata(metadata)
    details.setdefault("executor_id", executor_id.strip())
    details.setdefault("sandbox_provider", capabilities.provider_id)
    return registry.register(
        HARDENED_PARSER_EXECUTOR_PROVIDER_ID,
        provider,
        capabilities=("parser_sandbox", "kernel_isolation", "untrusted_document_parsing"),
        version=version,
        metadata=details,
        health_check=_health(health_check),
        replace=replace,
    )


def register_federated_search_provider(
    provider: Any,
    *,
    registry: RuntimeProviderRegistry = runtime_providers,
    version: str = "1.0.0",
    metadata: Mapping[str, str] | None = None,
    health_check: Callable[[], bool] | bool = True,
    replace: bool = False,
) -> RuntimeProviderBinding:
    collection = getattr(provider, "collection", None)
    if not isinstance(collection, FederatedCollection) or not callable(getattr(provider, "search", None)):
        raise TypeError("federated search provider must expose collection and search")
    safe_suffix = collection.provider_id.replace(":", "_").replace("/", "_")
    provider_key = FEDERATED_PROVIDER_PREFIX + safe_suffix
    details = _metadata(metadata)
    details.setdefault("provider", collection.provider_id)
    details.setdefault("collection_id", collection.collection_id)
    details.setdefault("collection_fingerprint", collection.collection_fingerprint)
    details.setdefault("disclosure_policy_fingerprint", collection.disclosure_policy_fingerprint)
    return registry.register(
        provider_key,
        provider,
        capabilities=("federated_retrieval", "private_collection_search"),
        version=version,
        metadata=details,
        health_check=_health(health_check),
        replace=replace,
    )


__all__ = [
    "ATTESTATION_KEYRING_PROVIDER_ID",
    "FEDERATED_PROVIDER_PREFIX",
    "HARDENED_PARSER_EXECUTOR_PROVIDER_ID",
    "OBJECT_GOVERNANCE_PROVIDER_ID",
    "SECURE_AGGREGATION_PROVIDER_ID",
    "register_attestation_keyring",
    "register_federated_search_provider",
    "register_hardened_parser_executor",
    "register_object_governance_provider",
    "register_secure_aggregation_provider",
]
