"""ACL-scoped hydrology generation history, replacement and replayability projection."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query

from tools.artifact_replacements import ArtifactReplacementStore
from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.hydrology_store import HydrologyArtifactStore
from tools.research_access import ResearchAccessResolver
from tools.research_dependencies import stale_reasons

_KIND_TO_DEP = {
    "topology": "hydrology_topology",
    "engineering_package": "hydrology_package",
    "retrieval_plan": "hydrology_plan",
    "evidence_projection": "hydrology_projection",
    "evidence_report": "hydrology_report",
}
_RECIPE_KINDS = frozenset({"retrieval_plan", "evidence_projection", "evidence_report"})


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _access(resolver: ResearchAccessResolver, actor: str, project_id: str):
    try:
        return resolver.project(actor, project_id, permission="hydrology.read")
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Research project not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research access control is unavailable.") from exc


def _recipe_status(recipe_store: Any, owner_id: str, kind: str, fingerprint: str) -> Mapping[str, Any]:
    if kind not in _RECIPE_KINDS:
        return {"replayable": False, "state": "not_deterministically_derived", "recipe_sha256": None}
    try:
        recipe = recipe_store.for_artifact(owner_id, kind, fingerprint)
    except KeyError:
        return {"replayable": False, "state": "recipe_missing", "recipe_sha256": None}
    except RuntimeError:
        return {"replayable": False, "state": "recipe_ambiguous", "recipe_sha256": None}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology derivation ledger is unavailable.") from exc
    return {"replayable": True, "state": "ready", "recipe_sha256": recipe.recipe_sha256}


def _queued_tasks(
    invalidations: DependencyInvalidationStore,
    owner_id: str,
    dependency_kind: str,
    fingerprint: str,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        for status in ("queued", "claimed", "failed"):
            for task in invalidations.list_recompute(owner_id, status=status, limit=10_000):
                if task.artifact.kind != dependency_kind or task.artifact.resource_id != fingerprint:
                    continue
                rows.append(
                    {
                        "task_id": task.task_id,
                        "status": task.status,
                        "attempts": task.attempts,
                        "created_at": task.created_at,
                        "claimed_at": task.claimed_at,
                        "completed_at": task.completed_at,
                        "error_type": task.error_type or None,
                        "triggering_event_sha256": task.triggering_event_sha256,
                    }
                )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology recompute ledger is unavailable.") from exc
    rows.sort(key=lambda item: (float(item["created_at"]), str(item["task_id"])), reverse=True)
    return rows


def build_hydrology_history_router(
    *,
    principal_dependency: Callable[..., Any],
    store: HydrologyArtifactStore,
    recipe_store: Any,
    access_resolver: ResearchAccessResolver,
    invalidation_store: DependencyInvalidationStore,
    replacement_store: ArtifactReplacementStore,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research-hydrology-history"])

    @router.get("/projects/{project_id}/hydrology/artifacts/{kind}/{logical_id}/history")
    async def artifact_history(
        project_id: str,
        kind: str,
        logical_id: str,
        limit: int = Query(default=200, ge=1, le=5000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        if kind not in _KIND_TO_DEP:
            raise HTTPException(status_code=400, detail="Unsupported hydrology artifact kind.")
        access = _access(access_resolver, _owner(principal), project_id)
        owner, project = access.storage_owner_id, access.project.project_id
        try:
            all_versions = store.list(owner, project, kind=kind, include_history=True, limit=5000)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology persistence is unavailable.") from exc
        versions = [item for item in all_versions if item.logical_id == logical_id][:limit]
        if not versions:
            raise HTTPException(status_code=404, detail="Hydrology artifact not found.")
        dependency_kind = _KIND_TO_DEP[kind]
        output: list[Mapping[str, Any]] = []
        for item in versions:
            ref = DependencyRef(dependency_kind, item.fingerprint)
            try:
                replacement_chain = replacement_store.chain(owner, ref, max_depth=64)
                stale = stale_reasons(invalidation_store, owner, ref, maximum=100)
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=503, detail="Hydrology lifecycle ledgers are unavailable.") from exc
            output.append(
                {
                    "fingerprint": item.fingerprint,
                    "version": item.version,
                    "created_at": item.created_at,
                    "is_current": item.is_current,
                    "stale": bool(stale),
                    "stale_reasons": list(stale),
                    "derivation": _recipe_status(recipe_store, owner, kind, item.fingerprint),
                    "recompute_tasks": _queued_tasks(invalidation_store, owner, dependency_kind, item.fingerprint),
                    "replacement_chain": [
                        {
                            "old_fingerprint": edge.old.resource_id,
                            "new_fingerprint": edge.new.resource_id,
                            "reason": edge.reason,
                            "triggering_event_sha256": edge.triggering_event_sha256,
                            "replacement_sha256": edge.replacement_sha256,
                            "created_at": edge.created_at,
                        }
                        for edge in replacement_chain
                    ],
                    "current_fingerprint": replacement_chain[-1].new.resource_id if replacement_chain else item.fingerprint,
                }
            )
        current = next((item for item in output if item["is_current"]), None)
        return {
            "project_id": project,
            "kind": kind,
            "logical_id": logical_id,
            "current_fingerprint": current["fingerprint"] if current is not None else None,
            "history_count": len(output),
            "history": output,
        }

    return router


__all__ = ["build_hydrology_history_router"]
