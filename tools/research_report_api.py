"""ACL-aware report API derived only from stored server-owned research results."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.research_access import ResearchAccessResolver
from tools.research_dependencies import register_report_dependencies, stale_reasons
from tools.research_report import ReportSection, ResearchReport, report_markdown
from tools.research_report_store import ResearchReportStore, StoredResearchReport
from tools.research_result_store import ResearchResultStore
from tools.research_workspace import ResearchProject, ResearchSession


class WorkspaceStore(Protocol):
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...
    def get_session(self, owner_id: str, session_id: str) -> ResearchSession: ...


class CreateReportRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=256)
    result_id: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    session_id: str | None = Field(default=None, min_length=1, max_length=256)


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _session_contains_result(session: ResearchSession, result_id: str) -> bool:
    target = result_id.lower()
    return any(turn.result_sha256 == target for turn in session.turns)


def _stored_payload(
    stored: StoredResearchReport,
    *,
    owner_id: str | None = None,
    invalidation_store: DependencyInvalidationStore | None = None,
) -> Mapping[str, Any]:
    report = stored.report
    stale = ()
    if owner_id is not None and invalidation_store is not None:
        stale = stale_reasons(invalidation_store, owner_id, DependencyRef("report", stored.report_id))
    return {
        "report_id": stored.report_id,
        "result_id": stored.result_id,
        "project_id": stored.project_id,
        "created_at": stored.created_at,
        "fingerprint": report.fingerprint,
        "title": report.title,
        "question": report.question,
        "search_strategy": report.search_strategy,
        "sections": [{"heading": item.heading, "body": item.body, "claim_ids": list(item.claim_ids), "citation_ids": list(item.citation_ids)} for item in report.sections],
        "evidence_matrix": [{"claim_id": item.claim_id, "claim_text": item.claim_text, "support_status": item.support_status, "study_id": item.study_id, "population": item.population, "intervention_or_exposure": item.intervention_or_exposure, "comparator": item.comparator, "outcome": item.outcome, "result": item.result, "uncertainty": item.uncertainty, "limitation": item.limitation, "citation_ids": list(item.citation_ids)} for item in report.evidence_matrix],
        "citations": [item.model_dump(exclude_none=True) for item in report.citations],
        "conflicts": list(report.conflicts),
        "limitations": list(report.limitations),
        "warnings": list(report.warnings),
        "stale": bool(stale),
        "stale_reasons": list(stale),
    }


def build_research_report_router(
    *,
    principal_dependency: Callable[..., Any],
    workspace_store: WorkspaceStore,
    result_store: ResearchResultStore,
    report_store: ResearchReportStore,
    invalidation_store: DependencyInvalidationStore | None = None,
    access_resolver: ResearchAccessResolver | None = None,
) -> APIRouter:
    if invalidation_store is not None and not isinstance(invalidation_store, DependencyInvalidationStore):
        raise TypeError("invalidation_store must be DependencyInvalidationStore or null")
    if access_resolver is not None and not isinstance(access_resolver, ResearchAccessResolver):
        raise TypeError("access_resolver must be ResearchAccessResolver or null")
    router = APIRouter(prefix="/research", tags=["research-reports"])

    @router.post("/reports", status_code=201)
    async def create_report(request: CreateReportRequest, principal: Any = Depends(principal_dependency)) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if access_resolver is None:
                project = workspace_store.get_project(actor, request.project_id)
                session = workspace_store.get_session(actor, request.session_id) if request.session_id else None
            else:
                project_access = access_resolver.project(actor, request.project_id, permission="report.write")
                project = project_access.project
                storage_owner = project_access.storage_owner_id
                if storage_owner != actor and not request.session_id:
                    raise PermissionError("shared report creation requires an accessible source session")
                session = None
                if request.session_id:
                    session_access = access_resolver.session(actor, request.session_id, permission="report.write")
                    if session_access.storage_owner_id != storage_owner or session_access.session.project_id != project.project_id:
                        raise PermissionError("source session is outside the selected project")
                    session = session_access.session
                if storage_owner != actor and (session is None or not _session_contains_result(session, request.result_id)):
                    raise PermissionError("shared report source result is not bound to the accessible session")
                if session is not None and not _session_contains_result(session, request.result_id):
                    raise KeyError(request.result_id)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research project, session, or result not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        try:
            result = result_store.get(storage_owner, request.result_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research result not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research result persistence is unavailable.") from exc
        if invalidation_store is not None:
            result_stale = stale_reasons(invalidation_store, storage_owner, DependencyRef("result", result.result_id))
            if result_stale:
                raise HTTPException(status_code=409, detail="A new report cannot be derived from a stale research result; recompute or explicitly resolve its invalidation first.")
        if len(result.citations) > 100:
            raise HTTPException(status_code=409, detail="The authoritative result exceeds the current 100-citation report limit.")
        report = ResearchReport(
            title=project.title,
            question=project.research_question,
            search_strategy=result.strategy,
            sections=(ReportSection(heading="Synthesis", body=result.answer, citation_ids=result.citation_ids),),
            evidence_matrix=(),
            citations=result.citations,
            warnings=tuple(result.warnings),
        )
        try:
            stored = report_store.put(storage_owner, result_id=result.result_id, project_id=project.project_id, report=report)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc
        if invalidation_store is not None:
            try:
                register_report_dependencies(invalidation_store, storage_owner, stored)
            except Exception as exc:
                raise HTTPException(status_code=503, detail=("Research report was persisted, but dependency lineage could not be registered. " f"Report ID: {stored.report_id}")) from exc
        return _stored_payload(stored, owner_id=storage_owner, invalidation_store=invalidation_store)

    @router.get("/reports")
    async def list_reports(
        project_id: str | None = Query(default=None, min_length=1, max_length=256),
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if project_id is not None and access_resolver is not None:
                access = access_resolver.project(actor, project_id, permission="report.read")
                storage_owner = access.storage_owner_id
                project_id = access.project.project_id
            values = report_store.list(storage_owner, project_id=project_id, limit=limit)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc
        rows = []
        for item in values:
            stale = stale_reasons(invalidation_store, storage_owner, DependencyRef("report", item.report_id), maximum=20) if invalidation_store is not None else ()
            rows.append({"report_id": item.report_id, "result_id": item.result_id, "project_id": item.project_id, "title": item.report.title, "fingerprint": item.report.fingerprint, "created_at": item.created_at, "stale": bool(stale), "stale_reasons": list(stale)})
        return {"reports": rows}

    @router.get("/reports/{report_id}")
    async def get_report(
        report_id: str,
        project_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if project_id is not None and access_resolver is not None:
                access = access_resolver.project(actor, project_id, permission="report.read")
                storage_owner = access.storage_owner_id
            stored = report_store.get(storage_owner, report_id)
            if project_id is not None and stored.project_id != project_id:
                raise KeyError(report_id)
            return _stored_payload(stored, owner_id=storage_owner, invalidation_store=invalidation_store)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research report not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research report ID is invalid.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc

    @router.get("/reports/{report_id}/markdown", response_class=PlainTextResponse)
    async def get_report_markdown(
        report_id: str,
        project_id: str | None = Query(default=None, min_length=1, max_length=256),
        principal: Any = Depends(principal_dependency),
    ) -> PlainTextResponse:
        actor = _owner(principal)
        storage_owner = actor
        try:
            if project_id is not None and access_resolver is not None:
                access = access_resolver.project(actor, project_id, permission="report.read")
                storage_owner = access.storage_owner_id
            stored = report_store.get(storage_owner, report_id)
            if project_id is not None and stored.project_id != project_id:
                raise KeyError(report_id)
        except (KeyError, PermissionError) as exc:
            raise HTTPException(status_code=404, detail="Research report not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research report ID is invalid.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc
        stale = stale_reasons(invalidation_store, storage_owner, DependencyRef("report", stored.report_id)) if invalidation_store is not None else ()
        if stale:
            raise HTTPException(status_code=409, detail="This report is stale; inspect its invalidation state before exporting it as current evidence.")
        return PlainTextResponse(report_markdown(stored.report), media_type="text/markdown; charset=utf-8", headers={"Cache-Control": "no-store"})

    return router


__all__ = ["CreateReportRequest", "WorkspaceStore", "build_research_report_router"]
