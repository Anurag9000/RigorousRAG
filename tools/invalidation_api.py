"""Owner-scoped APIs for source-status events and derived-artifact invalidation."""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.dependency_invalidation import (
    DependencyInvalidationStore,
    DependencyRef,
    InvalidationImpact,
)
from tools.retraction_propagation import SourceStatusEvent


class SourceStatusRequest(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=1000)
    status: str = Field(..., pattern=r"^(active|retracted|superseded|withdrawn|corrected)$")
    effective_at: float | None = Field(default=None, ge=0)
    event_source_id: str = Field(..., min_length=1, max_length=1000)
    replacement_source_id: str = Field(default="", max_length=1000)
    reason: str = Field(default="", max_length=5000)


class InvalidateRequest(BaseModel):
    kind: str = Field(..., min_length=1, max_length=64)
    resource_id: str = Field(..., min_length=1, max_length=1000)
    reason: str = Field(..., min_length=1, max_length=5000)
    event_type: str = Field(..., min_length=1, max_length=128)
    replacement_id: str = Field(default="", max_length=1000)
    event_sha256: str = Field(default="", max_length=64, pattern=r"^(?:|[0-9a-fA-F]{64})$")
    max_depth: int = Field(default=32, ge=1, le=64)
    max_impact: int = Field(default=10000, ge=1, le=100000)


class AcknowledgeInvalidationRequest(BaseModel):
    kind: str = Field(..., min_length=1, max_length=64)
    resource_id: str = Field(..., min_length=1, max_length=1000)
    event_sha256: str | None = Field(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _impact_payload(impact: InvalidationImpact | None) -> Mapping[str, Any] | None:
    if impact is None:
        return None
    return {
        "event_sha256": impact.triggering_event_sha256,
        "root": {"kind": impact.root.kind, "resource_id": impact.root.resource_id},
        "affected": [
            {"kind": item.kind, "resource_id": item.resource_id}
            for item in impact.affected
        ],
        "recompute_tasks": [
            {
                "task_id": item.task_id,
                "artifact": {"kind": item.artifact.kind, "resource_id": item.artifact.resource_id},
                "event_sha256": item.triggering_event_sha256,
                "reason": item.reason,
                "status": item.status,
                "attempts": item.attempts,
                "created_at": item.created_at,
            }
            for item in impact.recompute_tasks
        ],
    }


def build_invalidation_router(
    *,
    principal_dependency: Callable[..., Any],
    store: DependencyInvalidationStore,
) -> APIRouter:
    if not isinstance(store, DependencyInvalidationStore):
        raise TypeError("store must be DependencyInvalidationStore")
    router = APIRouter(prefix="/research", tags=["research-invalidation"])

    @router.post("/source-status", status_code=201)
    async def record_source_status(
        request: SourceStatusRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            event = SourceStatusEvent(
                source_id=request.source_id,
                status=request.status,
                effective_at=time.time() if request.effective_at is None else request.effective_at,
                event_source_id=request.event_source_id,
                replacement_source_id=request.replacement_source_id,
                reason=request.reason,
            )
            impact = store.record_source_status(owner, event)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source status event is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source status persistence is unavailable.") from exc
        return {
            "event": {
                "source_id": event.source_id,
                "status": event.status,
                "effective_at": event.effective_at,
                "event_source_id": event.event_source_id,
                "replacement_source_id": event.replacement_source_id,
                "reason": event.reason,
                "event_sha256": event.event_sha256,
            },
            "impact": _impact_payload(impact),
        }

    @router.get("/source-status/{source_id}")
    async def get_source_status_events(
        source_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            events = store.source_status_events(owner, source_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source status persistence is unavailable.") from exc
        return {
            "source_id": source_id,
            "events": [
                {
                    "status": item.status,
                    "effective_at": item.effective_at,
                    "event_source_id": item.event_source_id,
                    "replacement_source_id": item.replacement_source_id,
                    "reason": item.reason,
                    "event_sha256": item.event_sha256,
                }
                for item in events
            ],
        }

    @router.post("/invalidate", status_code=201)
    async def invalidate_resource(
        request: InvalidateRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            impact = store.invalidate(
                owner,
                root=DependencyRef(request.kind, request.resource_id),
                reason=request.reason,
                event_type=request.event_type,
                replacement_id=request.replacement_id,
                event_sha256=request.event_sha256,
                max_depth=request.max_depth,
                max_impact=request.max_impact,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalidation request is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Invalidation persistence is unavailable.") from exc
        return _impact_payload(impact) or {}

    @router.get("/stale")
    async def list_stale(
        kind: str | None = Query(default=None, min_length=1, max_length=64),
        include_acknowledged: bool = Query(default=False),
        limit: int = Query(default=500, ge=1, le=10000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            rows = store.list_stale(
                owner,
                kind=kind,
                include_acknowledged=include_acknowledged,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Stale-artifact query is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Invalidation persistence is unavailable.") from exc
        return {
            "artifacts": [
                {
                    "kind": item.artifact.kind,
                    "resource_id": item.artifact.resource_id,
                    "event_sha256": item.triggering_event_sha256,
                    "reason": item.reason,
                    "replacement_id": item.replacement_id,
                    "stale_at": item.stale_at,
                    "acknowledged_at": item.acknowledged_at,
                }
                for item in rows
            ]
        }

    @router.post("/stale/acknowledge")
    async def acknowledge_stale(
        request: AcknowledgeInvalidationRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            count = store.acknowledge(
                owner,
                DependencyRef(request.kind, request.resource_id),
                event_sha256=request.event_sha256,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Acknowledgement request is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Invalidation persistence is unavailable.") from exc
        return {"acknowledged": count}

    @router.get("/recompute")
    async def list_recompute(
        status: str | None = Query(default=None, max_length=32),
        limit: int = Query(default=500, ge=1, le=10000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            rows = store.list_recompute(owner, status=status, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Recompute query is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Invalidation persistence is unavailable.") from exc
        return {
            "tasks": [
                {
                    "task_id": item.task_id,
                    "artifact": {"kind": item.artifact.kind, "resource_id": item.artifact.resource_id},
                    "event_sha256": item.triggering_event_sha256,
                    "reason": item.reason,
                    "status": item.status,
                    "attempts": item.attempts,
                    "created_at": item.created_at,
                    "claimed_at": item.claimed_at,
                    "completed_at": item.completed_at,
                    "error_type": item.error_type,
                }
                for item in rows
            ]
        }

    return router


__all__ = [
    "AcknowledgeInvalidationRequest",
    "InvalidateRequest",
    "SourceStatusRequest",
    "build_invalidation_router",
]
