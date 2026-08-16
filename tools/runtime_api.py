"""Owner-authenticated read-only runtime composition metadata."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends

from tools.runtime_composition import RuntimeComposition


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("principal owner identity is unavailable")
    return value


def _safe_persistence_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 32:
        raise ValueError("persistence_metadata must be a bounded mapping")
    output: dict[str, Any] = {}
    for key, raw in value.items():
        name = str(key).strip()
        if not name or len(name) > 100:
            raise ValueError("persistence metadata key is invalid")
        if isinstance(raw, bool) or raw is None:
            output[name] = raw
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            output[name] = raw
        elif isinstance(raw, str):
            if len(raw) > 500 or "\x00" in raw:
                raise ValueError("persistence metadata value is invalid")
            output[name] = raw
        else:
            raise ValueError("persistence metadata values must be scalar and non-secret")
    return dict(sorted(output.items()))


def build_runtime_router(
    *,
    principal_dependency: Callable[..., Any],
    composition: RuntimeComposition,
    persistence_metadata: Mapping[str, Any] | None = None,
) -> APIRouter:
    if not isinstance(composition, RuntimeComposition):
        raise TypeError("composition must be RuntimeComposition")
    persistence = _safe_persistence_metadata(persistence_metadata)
    router = APIRouter(prefix="/research", tags=["research-runtime"])

    @router.get("/runtime")
    async def runtime_metadata(
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        _owner(principal)
        capabilities: list[Mapping[str, Any]] = []
        selected_ids = set(composition.selected_capabilities.values())
        for descriptor in composition.capabilities.snapshot():
            health = composition.capabilities.health(descriptor)
            capabilities.append(
                {
                    "capability_id": descriptor.capability_id,
                    "version": descriptor.version,
                    "kind": descriptor.kind,
                    "provider": descriptor.provider,
                    "modalities": list(descriptor.modalities),
                    "trust_level": descriptor.trust_level,
                    "enabled": descriptor.enabled,
                    "healthy": health.available,
                    "health_reason": health.reason,
                    "selected": descriptor.capability_id in selected_ids,
                    "fingerprint": descriptor.fingerprint,
                }
            )
        return {
            "runtime_config_fingerprint": composition.config.fingerprint,
            "capability_registry_fingerprint": composition.capabilities.fingerprint,
            "domain_registry_fingerprint": composition.domains.fingerprint,
            "selected_capabilities": dict(sorted(composition.selected_capabilities.items())),
            "environment": composition.config.environment,
            "instance_id": composition.config.instance_id,
            "persistence": persistence,
            "capabilities": capabilities,
            "domains": [
                {
                    "domain_id": descriptor.domain_id,
                    "version": descriptor.version,
                    "label": descriptor.label,
                    "capabilities": list(descriptor.capabilities),
                    "fingerprint": descriptor.fingerprint,
                }
                for descriptor in composition.domains.descriptors()
            ],
        }

    return router


__all__ = ["build_runtime_router"]
