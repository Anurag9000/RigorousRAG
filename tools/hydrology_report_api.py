"""ACL-scoped deterministic hydrology evidence report creation and export."""
from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from tools.dependency_invalidation import DependencyRef
from tools.hydrology_api import DependencyStore
from tools.hydrology_report import build_hydrology_report, report_csv, report_markdown, report_payload
from tools.hydrology_store import HydrologyArtifactEnvelope, HydrologyArtifactStore, decode_artifact, make_envelope
from tools.research_access import ResearchAccessResolver
from tools.research_dependencies import stale_reasons

_SHA_PATTERN = r"^[0-9a-fA-F]{64}$"


class CreateHydrologyReportRequest(BaseModel):
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


def _artifact(
    store: HydrologyArtifactStore,
    owner_id: str,
    project_id: str,
    kind: str,
    logical_id: str,
) -> HydrologyArtifactEnvelope:
    try:
        return store.get(owner_id, project_id, kind, logical_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Hydrology artifact not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology persistence is unavailable.") from exc


def _optional_current(store: HydrologyArtifactStore, owner_id: str, project_id: str, kind: str, logical_id: str):
    try:
        return store.get(owner_id, project_id, kind, logical_id)
    except KeyError:
        return None


def _stale(
    invalidation_store: DependencyStore | None,
    owner_id: str,
    kind: str,
    fingerprint: str,
) -> tuple[Mapping[str, Any], ...]:
    if invalidation_store is None:
        return ()
    try:
        return stale_reasons(
            invalidation_store,  # type: ignore[arg-type]
            owner_id,
            DependencyRef(f"hydrology_{kind}", fingerprint),
            maximum=100,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Hydrology stale-state ledger is unavailable.") from exc


def _require_export_access(resolver: ResearchAccessResolver, actor: str, project_id: str):
    hydrology = _access(resolver, actor, project_id, "hydrology.read")
    report = _access(resolver, actor, project_id, "report.read")
    if hydrology.storage_owner_id != report.storage_owner_id:
        raise HTTPException(status_code=409, detail="Project access resolution is inconsistent.")
    return hydrology


def build_hydrology_report_router(
    *,
    principal_dependency: Callable[..., Any],
    store: HydrologyArtifactStore,
    access_resolver: ResearchAccessResolver,
    invalidation_store: DependencyStore | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research-hydrology-report"])

    @router.post("/projects/{project_id}/hydrology/reports", status_code=201)
    async def create_report(
        project_id: str,
        request: CreateHydrologyReportRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        hydrology_access = _access(access_resolver, actor, project_id, "hydrology.read")
        report_access = _access(access_resolver, actor, project_id, "report.write")
        if hydrology_access.storage_owner_id != report_access.storage_owner_id:
            raise HTTPException(status_code=409, detail="Project access resolution is inconsistent.")
        owner = hydrology_access.storage_owner_id
        project = hydrology_access.project
        projection_envelope = _artifact(store, owner, project.project_id, "evidence_projection", request.projection_id)
        projection_stale = _stale(invalidation_store, owner, "projection", projection_envelope.fingerprint)
        if projection_stale:
            raise HTTPException(
                status_code=409,
                detail={"message": "Hydrology projection is stale; rebuild it before creating a report.", "stale_reasons": list(projection_stale)},
            )
        previous = _optional_current(store, owner, project.project_id, "evidence_report", request.report_id)
        try:
            projection = decode_artifact("evidence_projection", projection_envelope.payload)
            report = build_hydrology_report(
                projection,
                report_id=request.report_id,
                project_id=project.project_id,
                title=f"{project.title} — Hydrology Evidence Report",
                research_question=project.research_question,
            )
            stored = store.put(
                make_envelope(owner, project.project_id, "evidence_report", request.report_id, report),
                expected_current_fingerprint=request.expected_current_fingerprint,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Hydrology report persistence is unavailable.") from exc
        try:
            if invalidation_store is not None:
                report_ref = DependencyRef("hydrology_report", stored.fingerprint)
                invalidation_store.register_dependency(
                    owner,
                    upstream=DependencyRef("hydrology_projection", projection.fingerprint),
                    downstream=report_ref,
                    relation="hydrology_projection_report",
                )
                invalidation_store.register_dependency(
                    owner,
                    upstream=DependencyRef("project", project.project_id),
                    downstream=report_ref,
                    relation="hydrology_report_project_scope",
                )
                if previous is not None and previous.fingerprint != stored.fingerprint:
                    invalidation_store.invalidate(
                        owner,
                        root=DependencyRef("hydrology_report", previous.fingerprint),
                        reason="hydrology evidence report generation replaced",
                        event_type="hydrology_generation_replaced",
                        replacement_id=stored.fingerprint,
                        recomputable_kinds=(),
                    )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Hydrology report was persisted ({stored.fingerprint}), but dependency lineage could not be completed.",
            ) from exc
        return {
            "project_id": project.project_id,
            "report_id": request.report_id,
            "fingerprint": stored.fingerprint,
            "version": stored.version,
            "created_at": stored.created_at,
            "projection_id": request.projection_id,
            "projection_fingerprint": projection.fingerprint,
            "complete": report.complete,
            "summary": report_payload(report)["summary"],
            "diagnostics": list(report.diagnostics),
        }

    @router.get("/projects/{project_id}/hydrology/reports/{report_id}/markdown")
    async def export_markdown(
        project_id: str,
        report_id: str,
        allow_stale: bool = Query(default=False),
        principal: Any = Depends(principal_dependency),
    ) -> Response:
        actor = _owner(principal)
        access = _require_export_access(access_resolver, actor, project_id)
        envelope = _artifact(store, access.storage_owner_id, access.project.project_id, "evidence_report", report_id)
        stale = _stale(invalidation_store, access.storage_owner_id, "report", envelope.fingerprint)
        if stale and not allow_stale:
            raise HTTPException(status_code=409, detail={"message": "Hydrology report is stale.", "stale_reasons": list(stale)})
        try:
            report = decode_artifact("evidence_report", envelope.payload)
            body = report_markdown(report)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        headers = {"X-RigorousRAG-Artifact-Fingerprint": envelope.fingerprint}
        if stale:
            headers["X-RigorousRAG-Stale"] = "true"
        return Response(content=body, media_type="text/markdown; charset=utf-8", headers=headers)

    @router.get("/projects/{project_id}/hydrology/reports/{report_id}/csv")
    async def export_csv(
        project_id: str,
        report_id: str,
        allow_stale: bool = Query(default=False),
        principal: Any = Depends(principal_dependency),
    ) -> Response:
        actor = _owner(principal)
        access = _require_export_access(access_resolver, actor, project_id)
        envelope = _artifact(store, access.storage_owner_id, access.project.project_id, "evidence_report", report_id)
        stale = _stale(invalidation_store, access.storage_owner_id, "report", envelope.fingerprint)
        if stale and not allow_stale:
            raise HTTPException(status_code=409, detail={"message": "Hydrology report is stale.", "stale_reasons": list(stale)})
        try:
            report = decode_artifact("evidence_report", envelope.payload)
            body = report_csv(report)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        headers = {"X-RigorousRAG-Artifact-Fingerprint": envelope.fingerprint}
        if stale:
            headers["X-RigorousRAG-Stale"] = "true"
        return Response(content=body, media_type="text/csv; charset=utf-8", headers=headers)

    return router


__all__ = ["CreateHydrologyReportRequest", "build_hydrology_report_router"]
