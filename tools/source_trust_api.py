"""Owner-scoped human review and reconciliation API for source-trust governance."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Literal, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.dependency_invalidation import DependencyInvalidationStore
from tools.source_trust import SourceTrustFeatures, SourceTrustPolicy, evaluate_source_trust
from tools.source_trust_reconciliation import reconcile_source_trust_activations
from tools.source_trust_store import SourceTrustActivation, SourceTrustRevision, SourceTrustStore


class SourceTrustReviewRequest(BaseModel):
    source_id: str = Field(..., min_length=1, max_length=1000)
    source_type: Literal[
        "primary_study",
        "systematic_review",
        "meta_analysis",
        "guideline",
        "dataset",
        "technical_report",
        "preprint",
        "conference",
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


class SourceTrustReconcileRequest(BaseModel):
    source_id: str | None = Field(default=None, min_length=1, max_length=1000)
    limit: int = Field(default=1000, ge=1, le=10_000)


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


def _activation_payload(activation: SourceTrustActivation) -> Mapping[str, Any]:
    return {
        "activation_id": activation.activation_id,
        "source_id": activation.source_id,
        "previous_revision_id": activation.previous_revision_id,
        "revision_id": activation.revision_id,
        "activated_at": activation.activated_at,
        "invalidation_completed_at": activation.invalidation_completed_at,
        "pending": activation.pending,
        "last_error": activation.last_error,
    }


def _reconcile_payload(summary: Any) -> Mapping[str, Any]:
    return {
        "changed": summary.attempted > 0,
        "attempted": summary.attempted,
        "completed": summary.completed,
        "failed": summary.failed,
        "affected_artifacts": summary.affected_artifacts,
        "recompute_tasks": summary.recompute_tasks,
        "activations": [asdict(item) for item in summary.outcomes],
    }


def build_source_trust_router(
    *,
    principal_dependency: Callable[..., Any],
    store: SourceTrustStore,
    policy: SourceTrustPolicy | None = None,
    invalidation_store: DependencyInvalidationStore | None = None,
) -> APIRouter:
    if not isinstance(store, SourceTrustStore):
        raise TypeError("store must be SourceTrustStore")
    if invalidation_store is not None and not isinstance(invalidation_store, DependencyInvalidationStore):
        raise TypeError("invalidation_store must be DependencyInvalidationStore or null")
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
            if invalidation_store is None:
                pending = store.pending_activations(owner, source_id=request.source_id, limit=1000)
                invalidation: Mapping[str, Any] = {
                    "changed": bool(pending),
                    "pending": len(pending),
                    "reconciliation_configured": False,
                }
            else:
                summary = reconcile_source_trust_activations(
                    store,
                    invalidation_store,
                    owner,
                    source_id=request.source_id,
                    limit=1000,
                    stop_on_error=True,
                )
                invalidation = _reconcile_payload(summary)
                if summary.failed:
                    raise RuntimeError("source trust activation invalidation remains pending")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source trust review is invalid.") from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Source trust revision is durable, but downstream invalidation is not "
                    "fully reconciled yet. Retry reconciliation before relying on the new review."
                ),
            ) from exc
        return {**_payload(revision), "invalidation": invalidation}

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

    # Static routes intentionally precede /source-trust/{source_id}.
    @router.get("/source-trust/pending")
    async def pending_source_trust(
        source_id: str | None = Query(default=None, min_length=1, max_length=1000),
        limit: int = Query(default=1000, ge=1, le=10_000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            values = store.pending_activations(owner, source_id=source_id, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source trust pending query is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source trust registry is unavailable.") from exc
        return {"pending": [_activation_payload(item) for item in values]}

    @router.post("/source-trust/reconcile")
    async def reconcile_source_trust(
        request: SourceTrustReconcileRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        if invalidation_store is None:
            raise HTTPException(status_code=503, detail="Source trust invalidation reconciliation is not configured.")
        try:
            summary = reconcile_source_trust_activations(
                store,
                invalidation_store,
                owner,
                source_id=request.source_id,
                limit=request.limit,
                stop_on_error=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Source trust reconciliation request is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Source trust reconciliation is unavailable.") from exc
        return _reconcile_payload(summary)

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
            latest = store.latest(owner, source_id)
            activations = store.activation_history(owner, source_id, limit=limit)
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
            "active_revision_id": latest.revision_id if latest is not None else None,
            "history": [_payload(item) for item in history],
            "activations": [_activation_payload(item) for item in activations],
            "decision": (
                {
                    "eligible_for_new_claims": decision.eligible_for_new_claims,
                    "trust_score": decision.score,
                    "reasons": list(decision.reasons),
                    "policy_sha256": decision.policy_sha256,
                }
                if decision is not None
                else None
            ),
        }

    return router


__all__ = [
    "SourceTrustReconcileRequest",
    "SourceTrustReviewRequest",
    "build_source_trust_router",
]
