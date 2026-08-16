"""Owner-scoped project collaboration ACL management routes."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from tools.project_acl_store import ProjectACLStore, ProjectAccessScope
from tools.research_access import ResearchAccessResolver


class GrantRequest(BaseModel):
    principal_id: str = Field(..., min_length=1, max_length=256)
    role: str = Field(..., min_length=1, max_length=32)
    expires_at: float | None = None


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated principal identity is required.")
    return value


def _payload(scope: ProjectAccessScope) -> Mapping[str, Any]:
    return {
        "project_id": scope.project_id,
        "principal_id": scope.principal_id,
        "role": scope.role,
        "permissions": sorted(scope.permissions),
        "granted_by": scope.granted_by,
        "granted_at": scope.granted_at,
        "expires_at": scope.expires_at,
    }


def build_project_acl_router(
    *,
    principal_dependency: Callable[..., Any],
    acl_store: ProjectACLStore,
    access_resolver: ResearchAccessResolver,
) -> APIRouter:
    if not isinstance(acl_store, ProjectACLStore):
        raise TypeError("acl_store must be ProjectACLStore")
    if not isinstance(access_resolver, ResearchAccessResolver):
        raise TypeError("access_resolver must be ResearchAccessResolver")

    router = APIRouter(prefix="/research/projects", tags=["research-acl"])

    @router.get("/{project_id}/acl")
    async def list_grants(
        project_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        try:
            access = access_resolver.project(actor, project_id, permission="acl.manage")
            grants = acl_store.grants(
                actor,
                project_owner_id=access.storage_owner_id,
                project_id=access.project.project_id,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Project ACL management is not permitted.") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Project ACL request is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Project ACL persistence is unavailable.") from exc
        return {"project_id": access.project.project_id, "grants": [_payload(item) for item in grants]}

    @router.put("/{project_id}/acl")
    async def put_grant(
        project_id: str,
        request: GrantRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        try:
            access = access_resolver.project(actor, project_id, permission="acl.manage")
            grant = acl_store.grant(
                actor,
                project_owner_id=access.storage_owner_id,
                project_id=access.project.project_id,
                principal_id=request.principal_id,
                role=request.role,
                expires_at=request.expires_at,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Project ACL management is not permitted.") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Project ACL persistence is unavailable.") from exc
        return {"grant": _payload(grant)}

    @router.delete("/{project_id}/acl/{principal_id}")
    async def revoke_grant(
        project_id: str,
        principal_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        try:
            access = access_resolver.project(actor, project_id, permission="acl.manage")
            revoked = acl_store.revoke(
                actor,
                project_owner_id=access.storage_owner_id,
                project_id=access.project.project_id,
                principal_id=principal_id,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="Project ACL management is not permitted.") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Project ACL persistence is unavailable.") from exc
        if not revoked:
            raise HTTPException(status_code=404, detail="Active project grant not found.")
        return {"revoked": True, "principal_id": principal_id}

    return router


__all__ = ["GrantRequest", "build_project_acl_router"]
