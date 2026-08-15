"""Owner-scoped human review API for source-trust governance records."""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.source_trust import SourceTrustFeatures, SourceTrustPolicy, evaluate_source_trust
from tools.source_trust_store import SourceTrustRevision, SourceTrustStore


class SourceTrustReviewRequest(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=1000)
    source_type: Literal[
        "primary_study",
        "systematic_review",
        "meta_analysis",
        "guideline",
        "dataset",
        "documentation",
        "web",
        "model_output",
        "other",
    ] = "other"
    status: Literal[
        "active",
        "retracted",
        "superseded",
        "withdrawn",
        "corrected",
        "unknown",
    ] = "unknown"
    provenance_integrity: float = Field(default=1.0, ge=0.0, le=1.0)
    methodological_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    topical_applicability: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness: float = Field(default=0.5, ge=0.0, le=1.0)
    independent_replication: float = Field(default=0.0, ge=0.0, le=1.0)
    conflicts_of_interest_known: bool = False
    notes: list[str] = Field(default_factory=list, max_length=20)
    review_basis: str = Field(..., min_length=1, max_length=5000)


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _payload(revision: SourceTrustRevision) -> Mapping[str, Any]:
    features = revision.features
    return {
        "revision_id": revision.revision_id,
        "source_id": features.source_id,
        "source_type": features.source_type,
        "status": features.status,
        "provenance_integrity": features.provenance_integrity,
        "methodological_quality": features.methodological_quality,
        "topical_applicability": features.topical_applicability,
        "freshness": features.freshness,
        "independent_replication": features.independent_replication,
        "reviewed": features.reviewed,
        "conflicts_of_interest_known": features.conflicts_of_interest_known,
        "notes": list(features.notes),
        "reviewer_id": revision.reviewer_id,
        "review_basis": revision.review_basis,
        "created_at": revision.created_at,
    }


def build_source_trust_router(
    *,
    principal_dependency: Callable[..., Any],
    store: SourceTrustStore,
    policy: SourceTrustPolicy | None = None,
) -> APIRouter:
    if not isinstance(store, SourceTrustStore):
        raise TypeError("store must be SourceTrustStore")
    selected_policy = policy or SourceTrustPolicy()
    router = APIRouter(prefix="/research", tags=["source-trust"])

    @router.post("/source-trust", status_code=201)
    async def review_source(
        request: SourceTrustReviewRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            notes = tuple(str(item).strip()[:500] for item in request.notes if str(item).strip())
            features = SourceTrustFeatures(
                source_id=request.source_id,
                source_type=request.source_type,
                status=request.status,
                provenance_integrity=request.provenance_integrity,
                methodological_quality=request.methodological_quality,
                topical_applicability=request.topical_applicability,
                freshness=request.freshness,
                independent_replication=request.independent_replication,
                reviewed=True,
                conflicts_of_interest_known=request.conflicts_of_interest_known,
                notes=notes,
            )
            revision = store.put(
                owner,
                features,
                reviewer_id=owner,
                review_basis=request.review_basis,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source trust review is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source trust registry is unavailable.") from exc
        return _payload(revision)

    @router.get("/source-trust")
    async def list_source_trust(
        limit: int = Query(default=500, ge=1, le=5000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            values = store.list_latest(owner, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source trust registry is unavailable.") from exc
        return {"sources": [_payload(item) for item in values]}

    @router.get("/source-trust/{source_id}")
    async def source_trust_history(
        source_id: str,
        causal_claim: bool = Query(default=False),
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            history = store.history(owner, source_id, limit=limit)
            latest = history[0] if history else None
            decision = (
                evaluate_source_trust(latest.features, selected_policy, causal_claim=causal_claim)
                if latest is not None
                else None
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source trust query is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source trust registry is unavailable.") from exc
        return {
            "source_id": source_id,
            "history": [_payload(item) for item in history],
            "decision": (
                {
                    "eligible_for_new_claims": decision.eligible_for_new_claims,
                    "trust_score": decision.trust_score,
                    "reasons": list(decision.reasons),
                    "warnings": list(decision.warnings),
                    "decision_fingerprint": decision.decision_fingerprint,
                }
                if decision is not None
                else None
            ),
        }

    return router


__all__ = ["SourceTrustReviewRequest", "build_source_trust_router"]
