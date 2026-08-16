"""ACL-aware research-capsule preflight, creation and inspection APIs.

The API never exposes encrypted replay plaintext and never executes replay. It separates
manifest completeness from executable replay readiness so callers cannot mistake a
content-addressed archival snapshot for a guarantee that exact provider execution is
currently available.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Mapping, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.replay_recipe_store import EncryptedReplayRecipeStore
from tools.research_access import ResearchAccessResolver
from tools.research_capsule_builder import CapsuleBuildContext, assess_capsule, build_capsule
from tools.research_capsule_store import ResearchCapsuleStore, StoredResearchCapsule
from tools.research_dependencies import register_capsule_dependencies, stale_reasons
from tools.research_result_store import ResearchResultStore
from tools.research_workspace import ResearchProject, ResearchSession


class WorkspaceStore(Protocol):
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...


class CapsuleRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=256)
    session_id: str = Field(..., min_length=1, max_length=256)
    result_id: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    supersedes_capsule_id: str = Field(default="", max_length=256)
    require_replay_ready: bool = True


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _storage_scope(
    actor: str,
    request: CapsuleRequest,
    *,
    access_resolver: ResearchAccessResolver | None,
    permission: str,
) -> str:
    if access_resolver is None:
        return actor
    try:
        project_access = access_resolver.project(actor, request.project_id, permission=permission)
        session_access = access_resolver.session(actor, request.session_id, permission=permission)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Research project or session not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if (
        project_access.storage_owner_id != session_access.storage_owner_id
        or project_access.project.project_id != session_access.session.project_id
    ):
        raise HTTPException(status_code=404, detail="Research session is outside the selected project.")
    return project_access.storage_owner_id


def _load_context(
    storage_owner: str,
    request: CapsuleRequest,
    *,
    workspace_store: WorkspaceStore,
    result_store: ResearchResultStore,
    code_revision: str,
) -> CapsuleBuildContext:
    try:
        project = workspace_store.get_project(storage_owner, request.project_id)
        session = workspace_store.get_session(storage_owner, request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Research project or session not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
    if session.project_id != project.project_id:
        raise HTTPException(status_code=404, detail="Research session is outside the selected project.")
    try:
        result = result_store.get(storage_owner, request.result_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Research result not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research result persistence is unavailable.") from exc
    return CapsuleBuildContext(project=project, session=session, result=result, code_revision=code_revision)


def _replay_status(
    storage_owner: str,
    context: CapsuleBuildContext,
    replay_recipe_store: EncryptedReplayRecipeStore | None,
) -> tuple[bool, tuple[str, ...]]:
    if replay_recipe_store is None:
        return False, ("encrypted_replay_not_configured",)
    try:
        recipe = replay_recipe_store.metadata(storage_owner, context.result.result_id)
    except KeyError:
        return False, ("encrypted_replay_recipe_missing",)
    except Exception:
        return False, ("encrypted_replay_recipe_unavailable",)
    reasons: list[str] = []
    if recipe.query_sha256 != context.result.query_sha256:
        reasons.append("encrypted_replay_query_fingerprint_mismatch")
    if recipe.key_id != replay_recipe_store.key_id:
        reasons.append("encrypted_replay_key_unavailable")
    return not reasons, tuple(reasons)


def _preflight_payload(
    storage_owner: str,
    context: CapsuleBuildContext,
    *,
    replay_recipe_store: EncryptedReplayRecipeStore | None,
    invalidation_store: DependencyInvalidationStore | None,
) -> Mapping[str, Any]:
    assessment = assess_capsule(context)
    stale = (
        stale_reasons(invalidation_store, storage_owner, DependencyRef("result", context.result.result_id), maximum=100)
        if invalidation_store is not None
        else ()
    )
    replay_available, replay_reasons = _replay_status(storage_owner, context, replay_recipe_store)
    blockers = list(assessment.blockers)
    if stale:
        blockers.append("result_is_stale")
    manifest_ready = assessment.manifest_ready and not stale
    replay_ready = manifest_ready and replay_available
    return {
        "manifest_ready": manifest_ready,
        "replay_ready": replay_ready,
        "blockers": list(dict.fromkeys(blockers)),
        "replay_blockers": list(replay_reasons),
        "warnings": list(assessment.warnings),
        "bindings": dict(assessment.bindings),
        "stale_reasons": list(stale),
    }


def _stored_payload(
    stored: StoredResearchCapsule,
    *,
    owner_id: str | None = None,
    invalidation_store: DependencyInvalidationStore | None = None,
) -> Mapping[str, Any]:
    capsule = stored.capsule
    stale = ()
    if owner_id is not None and invalidation_store is not None:
        stale = stale_reasons(invalidation_store, owner_id, DependencyRef("capsule", capsule.capsule_id), maximum=100)
    return {
        "capsule_id": capsule.capsule_id,
        "fingerprint": capsule.fingerprint,
        "owner_scoped": True,
        "project_id": stored.project_id,
        "session_id": stored.session_id,
        "result_id": stored.result_id,
        "supersedes_capsule_id": stored.supersedes_capsule_id or None,
        "created_at": capsule.created_at,
        "schema_version": capsule.schema_version,
        "code_revision": capsule.code_revision,
        "references": [asdict(item) for item in capsule.references],
        "replay_steps": [asdict(item) for item in capsule.replay_steps],
        "notes": list(capsule.notes),
        "replayability": dict(capsule.replayability()),
        "stale": bool(stale),
        "stale_reasons": list(stale),
    }


def build_research_capsule_router(
    *,
    principal_dependency: Callable[..., Any],
    workspace_store: WorkspaceStore,
    result_store: ResearchResultStore,
    capsule_store: ResearchCapsuleStore,
    code_revision: str,
    replay_recipe_store: EncryptedReplayRecipeStore | None = None,
    invalidation_store: DependencyInvalidationStore | None = None,
    access_resolver: ResearchAccessResolver | None = None,
) -> APIRouter:
    if not isinstance(result_store, ResearchResultStore):
        raise TypeError("result_store must be ResearchResultStore")
    if not isinstance(capsule_store, ResearchCapsuleStore):
        raise TypeError("capsule_store must be ResearchCapsuleStore")
    if replay_recipe_store is not None and not isinstance(replay_recipe_store, EncryptedReplayRecipeStore):
        raise TypeError("replay_recipe_store must be EncryptedReplayRecipeStore or null")
    if invalidation_store is not None and not isinstance(invalidation_store, DependencyInvalidationStore):
        raise TypeError("invalidation_store must be DependencyInvalidationStore or null")
    if access_resolver is not None and not isinstance(access_resolver, ResearchAccessResolver):
        raise TypeError("access_resolver must be ResearchAccessResolver or null")

    router = APIRouter(prefix="/research/capsules", tags=["research-capsules"])

    @router.post("/preflight")
    async def preflight_capsule(request: CapsuleRequest, principal: Any = Depends(principal_dependency)) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = _storage_scope(actor, request, access_resolver=access_resolver, permission="capsule.write")
        context = _load_context(storage_owner, request, workspace_store=workspace_store, result_store=result_store, code_revision=code_revision)
        return _preflight_payload(storage_owner, context, replay_recipe_store=replay_recipe_store, invalidation_store=invalidation_store)

    @router.post("", status_code=201)
    async def create_capsule(request: CapsuleRequest, principal: Any = Depends(principal_dependency)) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = _storage_scope(actor, request, access_resolver=access_resolver, permission="capsule.write")
        context = _load_context(storage_owner, request, workspace_store=workspace_store, result_store=result_store, code_revision=code_revision)
        preflight = _preflight_payload(storage_owner, context, replay_recipe_store=replay_recipe_store, invalidation_store=invalidation_store)
        if not bool(preflight["manifest_ready"]):
            raise HTTPException(status_code=409, detail={"message": "Research capsule bindings are incomplete.", "preflight": preflight})
        if request.require_replay_ready and not bool(preflight["replay_ready"]):
            raise HTTPException(status_code=409, detail={"message": "Exact replay material is not currently available.", "preflight": preflight})
        try:
            capsule = build_capsule(context)
            stored = capsule_store.put(
                storage_owner,
                project_id=request.project_id,
                session_id=request.session_id,
                result_id=request.result_id.lower(),
                capsule=capsule,
                supersedes_capsule_id=request.supersedes_capsule_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Superseded research capsule not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research capsule persistence is unavailable.") from exc
        if invalidation_store is not None:
            try:
                register_capsule_dependencies(invalidation_store, storage_owner, stored)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=("Research capsule was persisted, but dependency lineage could not be registered. " f"Capsule ID: {stored.capsule_id}")) from exc
        payload = dict(_stored_payload(stored, owner_id=storage_owner, invalidation_store=invalidation_store))
        payload["preflight"] = preflight
        return payload

    @router.get("")
    async def list_capsules(
        project_id: str | None = Query(default=None, min_length=1, max_length=256),
        result_id: str | None = Query(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$"),
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if project_id is not None and access_resolver is not None:
                access = access_resolver.project(actor, project_id, permission="capsule.read")
                storage_owner = access.storage_owner_id
                project_id = access.project.project_id
            values = capsule_store.list(storage_owner, project_id=project_id, result_id=result_id, limit=limit)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research capsule persistence is unavailable.") from exc
        rows = []
        for item in values:
            stale = stale_reasons(invalidation_store, storage_owner, DependencyRef("capsule", item.capsule_id), maximum=20) if invalidation_store is not None else ()
            rows.append({"capsule_id": item.capsule_id, "fingerprint": item.fingerprint, "project_id": item.project_id, "session_id": item.session_id, "result_id": item.result_id, "supersedes_capsule_id": item.supersedes_capsule_id or None, "created_at": item.created_at, "stale": bool(stale), "stale_reasons": list(stale)})
        return {"capsules": rows}

    @router.get("/{capsule_id}")
    async def get_capsule(
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
        return _stored_payload(stored, owner_id=storage_owner, invalidation_store=invalidation_store)

    return router


__all__ = ["CapsuleRequest", "WorkspaceStore", "build_research_capsule_router"]
