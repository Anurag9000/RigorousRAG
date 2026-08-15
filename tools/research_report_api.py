"""Owner-scoped report API derived only from stored server-owned research results."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from tools.research_report import ReportSection, ResearchReport, report_markdown
from tools.research_report_store import ResearchReportStore, StoredResearchReport
from tools.research_result_store import ResearchResultStore
from tools.research_workspace import ResearchProject


class WorkspaceStore(Protocol):
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...


class CreateReportRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=256)
    result_id: str = Field(..., min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")


def _owner(principal: Any) -> str:
    value = getattr(principal, "owner_id", None)
    if not isinstance(value, str) or not value:
        raise HTTPException(status_code=401, detail="Authenticated owner identity is required.")
    return value


def _stored_payload(stored: StoredResearchReport) -> Mapping[str, Any]:
    report = stored.report
    return {
        "report_id": stored.report_id,
        "result_id": stored.result_id,
        "project_id": stored.project_id,
        "created_at": stored.created_at,
        "fingerprint": report.fingerprint,
        "title": report.title,
        "question": report.question,
        "search_strategy": report.search_strategy,
        "sections": [
            {
                "heading": item.heading,
                "body": item.body,
                "claim_ids": list(item.claim_ids),
                "citation_ids": list(item.citation_ids),
            }
            for item in report.sections
        ],
        "evidence_matrix": [
            {
                "claim_id": item.claim_id,
                "claim_text": item.claim_text,
                "support_status": item.support_status,
                "study_id": item.study_id,
                "population": item.population,
                "intervention_or_exposure": item.intervention_or_exposure,
                "comparator": item.comparator,
                "outcome": item.outcome,
                "result": item.result,
                "uncertainty": item.uncertainty,
                "limitation": item.limitation,
                "citation_ids": list(item.citation_ids),
            }
            for item in report.evidence_matrix
        ],
        "citations": [item.model_dump(exclude_none=True) for item in report.citations],
        "conflicts": list(report.conflicts),
        "limitations": list(report.limitations),
        "warnings": list(report.warnings),
    }


def build_research_report_router(
    *,
    principal_dependency: Callable[..., Any],
    workspace_store: WorkspaceStore,
    result_store: ResearchResultStore,
    report_store: ResearchReportStore,
) -> APIRouter:
    router = APIRouter(prefix="/research", tags=["research-reports"])

    @router.post("/reports", status_code=201)
    async def create_report(
        request: CreateReportRequest,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            project = workspace_store.get_project(owner, request.project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research project not found.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research workspace is unavailable.") from exc
        try:
            result = result_store.get(owner, request.result_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research result not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research result ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research result persistence is unavailable.") from exc
        if len(result.citations) > 100:
            raise HTTPException(
                status_code=409,
                detail="The authoritative result exceeds the current 100-citation report limit.",
            )
        citation_ids = result.citation_ids
        report = ResearchReport(
            title=project.title,
            question=project.research_question,
            search_strategy=result.strategy,
            sections=(
                ReportSection(
                    heading="Synthesis",
                    body=result.answer,
                    citation_ids=citation_ids,
                ),
            ),
            evidence_matrix=(),
            citations=result.citations,
            warnings=tuple(result.warnings),
        )
        try:
            stored = report_store.put(
                owner,
                result_id=result.result_id,
                project_id=project.project_id,
                report=report,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc
        return _stored_payload(stored)

    @router.get("/reports")
    async def list_reports(
        project_id: str | None = Query(default=None, min_length=1, max_length=256),
        limit: int = Query(default=100, ge=1, le=1000),
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            values = report_store.list(owner, project_id=project_id, limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc
        return {
            "reports": [
                {
                    "report_id": item.report_id,
                    "result_id": item.result_id,
                    "project_id": item.project_id,
                    "title": item.report.title,
                    "fingerprint": item.report.fingerprint,
                    "created_at": item.created_at,
                }
                for item in values
            ]
        }

    @router.get("/reports/{report_id}")
    async def get_report(
        report_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> Mapping[str, Any]:
        owner = _owner(principal)
        try:
            return _stored_payload(report_store.get(owner, report_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research report not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research report ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc

    @router.get("/reports/{report_id}/markdown", response_class=PlainTextResponse)
    async def get_report_markdown(
        report_id: str,
        principal: Any = Depends(principal_dependency),
    ) -> PlainTextResponse:
        owner = _owner(principal)
        try:
            stored = report_store.get(owner, report_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Research report not found.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Research report ID is invalid.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Research report persistence is unavailable.") from exc
        return PlainTextResponse(
            report_markdown(stored.report),
            media_type="text/markdown; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    return router


__all__ = ["CreateReportRequest", "WorkspaceStore", "build_research_report_router"]
