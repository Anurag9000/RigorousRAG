"""ACL-aware, non-executing verification API for immutable research capsules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.hydrology_capsule_verification import verify_stored_capsule_with_hydrology
from tools.hydrology_store import HydrologyArtifactStore
from tools.research_access import ResearchAccessResolver
from tools.research_capsule_store import ResearchCapsuleStore
from tools.research_capsule_verification import verify_stored_capsule
from tools.research_dependencies import stale_reasons
from tools.research_result_store import ResearchResultStore


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def build_research_capsule_verification_router(
    *,
    principal_dependency: Callable[..., Any],
    workspace_store: Any,
    result_store: ResearchResultStore,
    capsule_store: ResearchCapsuleStore,
    code_revision: str,
    invalidation_store: DependencyInvalidationStore | None = None,
    access_resolver: ResearchAccessResolver | None = None,
    hydrology_store: HydrologyArtifactStore | None = None,
) -> APIRouter:
    if not isinstance(result_store, ResearchResultStore):
        raise TypeError("result_store must be ResearchResultStore")
    if not isinstance(capsule_store, ResearchCapsuleStore):
        raise TypeError("capsule_store must be ResearchCapsuleStore")
    if invalidation_store is not None and not isinstance(invalidation_store, DependencyInvalidationStore):
        raise TypeError("invalidation_store must be DependencyInvalidationStore or null")
    if access_resolver is not None and not isinstance(access_resolver, ResearchAccessResolver):
        raise TypeError("access_resolver must be ResearchAccessResolver or null")

    router = APIRouter(prefix="/research/capsules", tags=["research-capsules"])

    @router.get("/{capsule_id}/verify")
    async def verify_capsule_manifest(
        capsule_id: str,
        project_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if project_id is not None and access_resolver is not None:
                access = access_resolver.project(actor, project_id, permission="capsule.read")
                storage_owner = access.storage_owner_id
                project_id = access.project.project_id
            stored = capsule_store.get(storage_owner, capsule_id)
            if project_id is not None and stored.project_id != project_id:
                raise KeyError(capsule_id)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research capsule not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research capsule ID is invalid.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research capsule persistence is unavailable.") from exc

        try:
            if hydrology_store is None:
                verification = verify_stored_capsule(
                    stored,
                    workspace_store=workspace_store,
                    result_store=result_store,
                    deployment_code_revision=code_revision,
                )
            else:
                verification = verify_stored_capsule_with_hydrology(
                    stored,
                    workspace_store=workspace_store,
                    result_store=result_store,
                    hydrology_store=hydrology_store,
                    deployment_code_revision=code_revision,
                )
        except (KeyError, PermissionError) as exc:
            raise HTTPException(
                status_code=409,
                detail="A durable authority required to verify this capsule is no longer available.",
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research capsule verification is unavailable.") from exc

        stale = (
            stale_reasons(
                invalidation_store,
                storage_owner,
                DependencyRef("capsule", stored.capsule_id),
                maximum=100,
            )
            if invalidation_store is not None
            else ()
        )
        receipt = verification.receipt
        hydrology_refs = [item for item in stored.capsule.references if item.ref_id.startswith("hydrology:")]
        return {
            "capsule_id": stored.capsule_id,
            "fingerprint": stored.fingerprint,
            "manifest_verified": verification.manifest_verified,
            "current_evidence": not stale,
            "deployment_compatible": verification.deployment_compatible,
            "code_revision_status": verification.code_revision_status,
            "deployment_code_revision": verification.deployment_code_revision or None,
            "reference_verification": [asdict(item) for item in receipt.references],
            "hydrology_generation_refs": len(hydrology_refs),
            "unavailable_ref_ids": list(receipt.unavailable_ref_ids),
            "mismatched_ref_ids": list(receipt.mismatched_ref_ids),
            "stale": bool(stale),
            "stale_reasons": list(stale),
            "replay_preconditions_met": bool(
                verification.manifest_verified and verification.deployment_compatible and not stale
            ),
            "note": (
                "Verification does not decrypt replay recipes or execute providers. "
                "Replay material availability is reported separately by the replay/capsule APIs."
            ),
        }

    return router


__all__ = ["build_research_capsule_verification_router"]
