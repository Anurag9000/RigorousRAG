"""Owner-scoped FastAPI routes for durable review and feedback state."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.feedback_store import FeedbackStore
from tools.review_store import ReviewRecord, ReviewStore
from tools.security import Principal


class ClaimBody(BaseModel):
    reviewer_id: str = Field(..., min_length=1, max_length=500)
    ttl_seconds: float = Field(default=300.0, gt=0.0, le=86400.0)


class LeaseBody(BaseModel):
    reviewer_id: str = Field(..., min_length=1, max_length=500)
    lease_token: int = Field(..., ge=1)
    ttl_seconds: float = Field(default=300.0, gt=0.0, le=86400.0)


class ResolveBody(BaseModel):
    reviewer_id: str = Field(..., min_length=1, max_length=500)
    lease_token: int = Field(..., ge=1)
    resolution: str = Field(..., min_length=1, max_length=500)


class FeedbackBody(BaseModel):
    event_id: str = Field(..., min_length=1, max_length=500)
    kind: str = Field(..., min_length=1, max_length=64)
    subject_id: str = Field(..., min_length=1, max_length=500)
    query: Optional[str] = Field(default=None, max_length=20000)
    evidence: Optional[str] = Field(default=None, max_length=20000)
    weight: float = Field(default=1.0, gt=0.0, le=1000.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _review(record: ReviewRecord) -> dict[str, Any]:
    data = asdict(record)
    data["reasons"] = list(record.reasons)
    data["metadata"] = dict(record.metadata)
    return data


def build_control_router(*, principal_dependency: Callable[..., Any], review_store: ReviewStore,
                         feedback_store: FeedbackStore) -> APIRouter:
    router = APIRouter(tags=["governance"])

    @router.get("/reviews")
    async def reviews(state: Optional[str] = Query(default=None, max_length=32),
                      limit: int = Query(default=100, ge=1, le=1000),
                      principal: Principal = Depends(principal_dependency)) -> list[dict[str, Any]]:
        try:
            return [_review(item) for item in review_store.list(owner_id=principal.owner_id, state=state, limit=limit)]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/reviews/claim")
    async def claim(body: ClaimBody, principal: Principal = Depends(principal_dependency)) -> dict[str, Any] | None:
        item = review_store.claim_next(owner_id=principal.owner_id, reviewer_id=body.reviewer_id,
                                       ttl_seconds=body.ttl_seconds)
        return None if item is None else _review(item)

    def owned_claim(owner: str, request_id: str, reviewer: str, token: int) -> ReviewRecord:
        item = review_store.get(owner_id=owner, request_id=request_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Review request not found.")
        if item.state != "claimed" or item.reviewer_id != reviewer or item.lease_token != token:
            raise HTTPException(status_code=409, detail="Review lease is stale or owned by another reviewer.")
        return item

    @router.post("/reviews/{request_id}/renew")
    async def renew(request_id: str, body: LeaseBody,
                    principal: Principal = Depends(principal_dependency)) -> dict[str, Any]:
        current = owned_claim(principal.owner_id, request_id, body.reviewer_id, body.lease_token)
        item = review_store.renew(current, ttl_seconds=body.ttl_seconds)
        if item is None:
            raise HTTPException(status_code=409, detail="Review lease expired before renewal.")
        return _review(item)

    @router.post("/reviews/{request_id}/resolve")
    async def resolve(request_id: str, body: ResolveBody,
                      principal: Principal = Depends(principal_dependency)) -> dict[str, bool]:
        current = owned_claim(principal.owner_id, request_id, body.reviewer_id, body.lease_token)
        if not review_store.resolve(current, resolution=body.resolution):
            raise HTTPException(status_code=409, detail="Review lease expired before resolution.")
        return {"resolved": True}

    @router.post("/feedback")
    async def feedback(body: FeedbackBody,
                       principal: Principal = Depends(principal_dependency)) -> dict[str, Any]:
        try:
            item = feedback_store.put(owner_id=principal.owner_id, event_id=body.event_id, kind=body.kind,
                                      subject_id=body.subject_id, query=body.query, evidence=body.evidence,
                                      weight=body.weight, metadata=body.metadata)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        data = asdict(item)
        data["metadata"] = dict(item.metadata)
        return data

    @router.get("/feedback")
    async def feedback_list(kind: Optional[str] = Query(default=None, max_length=64),
                            limit: int = Query(default=100, ge=1, le=10000),
                            principal: Principal = Depends(principal_dependency)) -> list[dict[str, Any]]:
        try:
            items = feedback_store.list(owner_id=principal.owner_id, kind=kind, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return [{**asdict(item), "metadata": dict(item.metadata)} for item in items]

    return router


__all__ = ["build_control_router"]
