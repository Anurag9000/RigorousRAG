"""ACL-scoped hydrology derivation endpoints with immutable replay recipes.

These endpoints are the replayable creation path for deterministic plan/projection/report
artifacts. The recipe is persisted before the artifact current pointer advances. An orphan
recipe is harmless; an artifact created successfully through this API always has a recipe.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from tools.hydrology_derivation_store import (
    HydrologyDerivationStore,
    plan_recipe,
    projection_recipe,
    report_recipe,
)
from tools.hydrology_projection import build_hydrology_projection
from tools.hydrology_report import build_hydrology_report, report_payload
from tools.hydrology_retrieval import plan_hydrology_retrieval
from tools.hydrology_store import HydrologyArtifactStore, decode_artifact, make_envelope, query_spec_from_payload
from tools.research_access import ResearchAccessResolver
from tools.spatiotemporal_index import SpatiotemporalIndex

_SHA_PATTERN = r"^[0-9a-fA-F]{64}$"


class DerivePlanRequest(BaseModel):
    plan_id: str = Field(..., min_length=1, max_length=500)
    topology_id: str = Field(..., min_length=1, max_length=500)
    package_id: str = Field(..., min_length=1, max_length=500)
    spec: dict[str, Any]
    reach_travel_seconds: dict[str, float] = Field(default_factory=dict)
    limit: int = Field(default=1000, ge=1, le=10_000)
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


class DeriveProjectionRequest(BaseModel):
    projection_id: str = Field(..., min_length=1, max_length=500)
    package_id: str = Field(..., min_length=1, max_length=500)
    plan_id: str = Field(..., min_length=1, max_length=500)
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


class DeriveReportRequest(BaseModel):
    report_id: str = Field(..., min_length=1, max_length=500)
    projection_id: str = Field(..., min_length=1, max_length=500)
    expected_current_fingerprint: str | None = Field(default=None, pattern=_SHA_PATTERN)


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _access(resolver: ResearchAccessResolver, actor: str, project_id: str, permission: str):
    try:
        return resolver.project(actor, project_id, permission=permission)
    except (KeyError, PermissionError) as exc:
        raise HTTPException(status_code=404, detail="Research project not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Research access control is unavailable.") from exc


def _artifact(store: HydrologyArtifactStore, owner: str, project: str, kind: str, logical_id: str):
    try:
        return store.get(owner, project, kind, logical_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Required hydrology {kind} was not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology persistence is unavailable.") from exc


def _summary(stored: Any, recipe: Any) -> Mapping[str, Any]:
    return {
        "project_id": stored.project_id,
        "kind": stored.kind,
        "logical_id": stored.logical_id,
        "fingerprint": stored.fingerprint,
        "version": stored.version,
        "created_at": stored.created_at,
        "is_current": stored.is_current,
        "recipe_sha256": recipe.recipe_sha256,
        "replayable": True,
    }


def build_hydrology_derivation_router(
    *,
    principal_dependency: Callable[..., Any],
    store: HydrologyArtifactStore,
    recipe_store: Any,
    access_resolver: ResearchAccessResolver,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research-hydrology-derivation"])

    @router.get("/projects/{project_id}/hydrology/derivations")
    async def list_derivations(
        project_id: str,
        limit: int = Query(default=200, ge=1, le=5000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        access = _access(access_resolver, _owner(principal), project_id, "hydrology.read")
        try:
            recipes = recipe_store.list_project(access.storage_owner_id, access.project.project_id, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology derivation ledger is unavailable.") from exc
        return {
            "project_id": access.project.project_id,
            "derivations": [
                {
                    "artifact_kind": item.artifact_kind,
                    "logical_id": item.logical_id,
                    "artifact_fingerprint": item.artifact_fingerprint,
                    "recipe_sha256": item.recipe_sha256,
                    "created_at": item.created_at,
                    "inputs": dict(item.inputs),
                }
                for item in recipes
            ],
        }

    @router.post("/projects/{project_id}/hydrology/derive/plans", status_code=201)
    async def derive_plan(
        project_id: str,
        request: DerivePlanRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        access = _access(access_resolver, _owner(principal), project_id, "hydrology.write")
        owner, project = access.storage_owner_id, access.project.project_id
        topology_envelope = _artifact(store, owner, project, "topology", request.topology_id)
        package_envelope = _artifact(store, owner, project, "engineering_package", request.package_id)
        try:
            if len(request.reach_travel_seconds) > 10_000:
                raise ValueError("reach_travel_seconds exceed the item limit")
            network = decode_artifact("topology", topology_envelope.payload)
            package = decode_artifact("engineering_package", package_envelope.payload)
            if package.topology_fingerprint != network.fingerprint:
                raise RuntimeError("selected package and topology generations are incompatible")
            spec = query_spec_from_payload(request.spec)
            index = SpatiotemporalIndex()
            package.populate_index(index)
            plan = plan_hydrology_retrieval(
                network,
                index,
                spec,
                reach_travel_seconds=request.reach_travel_seconds,
                limit=request.limit,
                package=package,
                expected_index_fingerprint=index.fingerprint,
            )
            recipe = plan_recipe(
                owner,
                project,
                logical_id=request.plan_id,
                artifact_fingerprint=plan.fingerprint,
                topology_id=request.topology_id,
                topology_fingerprint=network.fingerprint,
                package_id=request.package_id,
                package_fingerprint=package.fingerprint,
                spec=request.spec,
                reach_travel_seconds=request.reach_travel_seconds,
                limit=request.limit,
            )
            recipe_store.put(recipe)
            stored = store.put(
                make_envelope(owner, project, "retrieval_plan", request.plan_id, plan),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replayable hydrology plan derivation is unavailable.") from exc
        return {**_summary(stored, recipe), "executable": plan.executable, "record_count": len(plan.record_ids), "unresolved": list(plan.unresolved)}

    @router.post("/projects/{project_id}/hydrology/derive/projections", status_code=201)
    async def derive_projection(
        project_id: str,
        request: DeriveProjectionRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        access = _access(access_resolver, _owner(principal), project_id, "hydrology.write")
        owner, project = access.storage_owner_id, access.project.project_id
        package_envelope = _artifact(store, owner, project, "engineering_package", request.package_id)
        plan_envelope = _artifact(store, owner, project, "retrieval_plan", request.plan_id)
        try:
            package = decode_artifact("engineering_package", package_envelope.payload)
            plan = decode_artifact("retrieval_plan", plan_envelope.payload)
            projection = build_hydrology_projection(package, plan, projection_id=request.projection_id)
            recipe = projection_recipe(
                owner,
                project,
                logical_id=request.projection_id,
                artifact_fingerprint=projection.fingerprint,
                package_id=request.package_id,
                package_fingerprint=package.fingerprint,
                plan_id=request.plan_id,
                plan_fingerprint=plan.fingerprint,
            )
            recipe_store.put(recipe)
            stored = store.put(
                make_envelope(owner, project, "evidence_projection", request.projection_id, projection),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replayable hydrology projection derivation is unavailable.") from exc
        return {**_summary(stored, recipe), "complete": projection.complete, "row_count": len(projection.rows)}

    @router.post("/projects/{project_id}/hydrology/derive/reports", status_code=201)
    async def derive_report(
        project_id: str,
        request: DeriveReportRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        hydrology_access = _access(access_resolver, _owner(principal), project_id, "hydrology.read")
        report_access = _access(access_resolver, _owner(principal), project_id, "report.write")
        if hydrology_access.storage_owner_id != report_access.storage_owner_id:
            raise HTTPException(status_code=409, detail="Project access resolution is inconsistent.")
        owner, project = hydrology_access.storage_owner_id, hydrology_access.project
        projection_envelope = _artifact(store, owner, project.project_id, "evidence_projection", request.projection_id)
        try:
            projection = decode_artifact("evidence_projection", projection_envelope.payload)
            title = f"{project.title} — Hydrology Evidence Report"
            report = build_hydrology_report(
                projection,
                report_id=request.report_id,
                project_id=project.project_id,
                title=title,
                research_question=project.research_question,
            )
            recipe = report_recipe(
                owner,
                project.project_id,
                logical_id=request.report_id,
                artifact_fingerprint=report.fingerprint,
                projection_id=request.projection_id,
                projection_fingerprint=projection.fingerprint,
                title=title,
                research_question=project.research_question,
            )
            recipe_store.put(recipe)
            stored = store.put(
                make_envelope(owner, project.project_id, "evidence_report", request.report_id, report),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Replayable hydrology report derivation is unavailable.") from exc
        return {**_summary(stored, recipe), "complete": report.complete, "summary": report_payload(report)["summary"], "diagnostics": list(report.diagnostics)}

    return router


__all__ = [
    "DerivePlanRequest",
    "DeriveProjectionRequest",
    "DeriveReportRequest",
    "build_hydrology_derivation_router",
]
