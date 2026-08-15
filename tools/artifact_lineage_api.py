"""Owner-scoped inspection API for immutable artifact replacement lineage."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query

from tools.artifact_replacements import ArtifactReplacementStore
from tools.dependency_invalidation import DependencyRef


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def build_artifact_lineage_router(
    *,
    principal_dependency: Callable[..., Any],
    replacements: ArtifactReplacementStore,
) -> APIRouter:
    if not isinstance(replacements, ArtifactReplacementStore):
        raise TypeError("replacements must be ArtifactReplacementStore")
    router = APIRouter(prefix="/research", tags=["research-lineage"])

    @router.get("/artifacts/{kind}/{resource_id}/lineage")
    async def artifact_lineage(
        kind: str,
        resource_id: str,
        max_depth: int = Query(default=64, ge=1, le=256),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            start = DependencyRef(kind, resource_id)
            chain = replacements.chain(owner, start, max_depth=max_depth)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Artifact identity is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Artifact replacement lineage is unavailable.") from exc
        current = chain[-1].new if chain else start
        return {
            "requested": {"kind": start.kind, "resource_id": start.resource_id},
            "current": {"kind": current.kind, "resource_id": current.resource_id},
            "superseded": current != start,
            "replacements": [
                {
                    "old": {"kind": item.old.kind, "resource_id": item.old.resource_id},
                    "new": {"kind": item.new.kind, "resource_id": item.new.resource_id},
                    "reason": item.reason,
                    "event_sha256": item.triggering_event_sha256,
                    "replacement_sha256": item.replacement_sha256,
                    "created_at": item.created_at,
                }
                for item in chain
            ],
        }

    return router


__all__ = ["build_artifact_lineage_router"]
