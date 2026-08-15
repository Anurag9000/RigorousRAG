"""Hardened API plus durable owner-scoped governance and research routes."""
from pathlib import Path
import os

import server as base
from fastapi import Request
from tools.agent_runtime import configure_agent_runtime
from tools.control_api import build_control_router
from tools.feedback_store import FeedbackStore
from tools.postgres_workspace_store import PostgresResearchWorkspaceStore
from tools.research_api import build_research_router
from tools.research_query_api import build_research_query_router
from tools.research_report_api import build_research_report_router
from tools.research_report_store import ResearchReportStore
from tools.research_result_store import ResearchResultStore
from tools.research_workspace_sqlite import SQLiteResearchWorkspaceStore
from tools.review_store import ReviewStore
from tools.runtime_api import build_runtime_router
from tools.runtime_composition import build_runtime_composition
from tools.runtime_providers import runtime_providers

root = Path(os.environ.get("CLASSIC_STORAGE_DIR", "data")).resolve() / "governance"
root.mkdir(parents=True, exist_ok=True)
composition = build_runtime_composition()


def _build_workspace_store():
    backend = composition.config.storage.metadata_backend
    if backend in {"postgres", "postgresql"}:
        connection_factory = runtime_providers.require("postgres.connection_factory")
        return PostgresResearchWorkspaceStore(connection_factory)
    if backend == "sqlite":
        return SQLiteResearchWorkspaceStore(root / "research_workspace.sqlite3")
    raise RuntimeError(f"unsupported research workspace metadata backend: {backend}")


reviews = ReviewStore(root / "reviews.sqlite3")
feedback = FeedbackStore(root / "feedback.sqlite3")
workspace = _build_workspace_store()
results = ResearchResultStore(root / "research_results.sqlite3")
reports = ResearchReportStore(root / "research_reports.sqlite3")
app = base.app

_base_new_agent = base._new_agent


def _production_agent(owner_id: str, model=None):
    agent = _base_new_agent(owner_id, model)
    return configure_agent_runtime(agent, composition, providers=runtime_providers)


# ``server_app`` resolves this module global at request/ingestion time, so replacing the
# trusted factory here gives both legacy /query and ingestion-created agents the same
# composed runtime without changing the compatibility constructor signature.
base._new_agent = _production_agent

_REQUIRED_GOVERNANCE_ROUTES = frozenset({"/reviews", "/reviews/claim", "/feedback"})
_REQUIRED_RESEARCH_ROUTES = frozenset(
    {
        "/research/projects",
        "/research/projects/{project_id}",
        "/research/projects/{project_id}/sessions",
        "/research/sessions/{session_id}",
        "/research/sessions/{session_id}/turns",
        "/research/sessions/{session_id}/close",
        "/research/capabilities",
        "/research/runtime",
        "/research/query",
        "/research/results",
        "/research/results/{result_id}",
        "/research/reports",
        "/research/reports/{report_id}",
        "/research/reports/{report_id}/markdown",
    }
)


def _route_paths() -> set[str]:
    return {
        path
        for route in app.routes
        if isinstance((path := getattr(route, "path", None)), str)
    }


def _append_missing_routes(router) -> None:
    known_paths = _route_paths()
    for route in router.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str) and path not in known_paths:
            app.router.routes.append(route)
            known_paths.add(path)


def _ensure_governance_routes() -> None:
    if not _REQUIRED_GOVERNANCE_ROUTES.issubset(_route_paths()):
        governance = build_control_router(
            principal_dependency=base.get_principal,
            review_store=reviews,
            feedback_store=feedback,
        )
        _append_missing_routes(governance)
    missing = _REQUIRED_GOVERNANCE_ROUTES.difference(_route_paths())
    if missing:
        raise RuntimeError(
            "Production governance routes failed to mount: " + ", ".join(sorted(missing))
        )


def _ensure_research_routes() -> None:
    if not _REQUIRED_RESEARCH_ROUTES.issubset(_route_paths()):
        research = build_research_router(
            principal_dependency=base.get_rate_limited_principal,
            workspace_store=workspace,
            capability_registry=composition.capabilities,
            domain_registry=composition.domains,
        )
        runtime = build_runtime_router(
            principal_dependency=base.get_rate_limited_principal,
            composition=composition,
        )
        query = build_research_query_router(
            principal_dependency=base.get_rate_limited_principal,
            agent_factory=_production_agent,
            run_research_task=base._run_research_task,
            result_store=results,
            workspace_store=workspace,
            composition=composition,
        )
        report = build_research_report_router(
            principal_dependency=base.get_rate_limited_principal,
            workspace_store=workspace,
            result_store=results,
            report_store=reports,
        )
        _append_missing_routes(research)
        _append_missing_routes(runtime)
        _append_missing_routes(query)
        _append_missing_routes(report)
    missing = _REQUIRED_RESEARCH_ROUTES.difference(_route_paths())
    if missing:
        raise RuntimeError(
            "Production research routes failed to mount: " + ", ".join(sorted(missing))
        )


_ensure_governance_routes()
_ensure_research_routes()


@app.middleware("http")
async def governance_no_store(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/reviews", "/feedback", "/research")):
        response.headers["Cache-Control"] = "no-store"
    return response


__all__ = [
    "app",
    "composition",
    "feedback",
    "reports",
    "results",
    "reviews",
    "workspace",
]
