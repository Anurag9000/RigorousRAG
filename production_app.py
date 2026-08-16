"""Hardened API plus durable owner-scoped governance and research routes."""
from pathlib import Path
import os

import server as base
from fastapi import Request
from tools.agent_runtime import configure_agent_runtime
from tools.artifact_lineage_api import build_artifact_lineage_router
from tools.artifact_replacements import ArtifactReplacementStore
from tools.control_api import build_control_router
from tools.dependency_invalidation import DependencyInvalidationStore
from tools.feedback_store import FeedbackStore
from tools.invalidation_api import build_invalidation_router
from tools.postgres_workspace_store import PostgresResearchWorkspaceStore
from tools.project_acl_api import build_project_acl_router
from tools.project_acl_store import ProjectACLStore
from tools.recompute_executor import ResearchRecomputeExecutor
from tools.replay_api import build_replay_router
from tools.replay_runtime import build_replay_recipe_store
from tools.research_access import ResearchAccessResolver
from tools.research_answer_history_api import build_research_answer_history_router
from tools.research_api import build_research_router
from tools.research_capsule_api import build_research_capsule_router
from tools.research_capsule_store import ResearchCapsuleStore
from tools.research_capsule_verification_api import build_research_capsule_verification_router
from tools.research_query_api import build_research_query_router
from tools.research_report_api import build_research_report_router
from tools.research_report_store import ResearchReportStore
from tools.research_result_store import ResearchResultStore
from tools.research_workspace_sqlite import SQLiteResearchWorkspaceStore
from tools.review_store import ReviewStore
from tools.runtime_api import build_runtime_router
from tools.runtime_composition import build_runtime_composition
from tools.runtime_providers import runtime_providers
from tools.source_trust_api import build_source_trust_router
from tools.source_trust_store import SourceTrustStore

root = Path(os.environ.get("CLASSIC_STORAGE_DIR", "data")).resolve() / "governance"
root.mkdir(parents=True, exist_ok=True)
composition = build_runtime_composition()
code_revision = os.environ.get("RIGOROUSRAG_CODE_REVISION", "").strip()


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
project_acls = ProjectACLStore(root / "research_project_acl.sqlite3")
access_resolver = ResearchAccessResolver(workspace, project_acls)
results = ResearchResultStore(root / "research_results.sqlite3")
reports = ResearchReportStore(root / "research_reports.sqlite3")
capsules = ResearchCapsuleStore(root / "research_capsules.sqlite3")
invalidations = DependencyInvalidationStore(root / "research_invalidation.sqlite3")
replacements = ArtifactReplacementStore(root / "research_replacements.sqlite3")
source_trust = SourceTrustStore(root / "source_trust.sqlite3")
replay_recipes = build_replay_recipe_store(
    root / "research_replay.sqlite3", providers=runtime_providers
)
app = base.app

_base_new_agent = base._new_agent


def _production_agent(owner_id: str, model=None):
    agent = _base_new_agent(owner_id, model)
    configured = configure_agent_runtime(agent, composition, providers=runtime_providers)
    configured.source_status_store = invalidations
    configured.source_trust_store = source_trust
    return configured


base._new_agent = _production_agent

# Deliberately not mounted as a public HTTP action: recomputation can incur model/provider
# cost and is an operator/deployment-worker responsibility. The queue state remains
# owner-visible through /research/recompute.
recompute_executor = ResearchRecomputeExecutor(
    invalidations=invalidations,
    replacements=replacements,
    results=results,
    reports=reports,
    workspace=workspace,
    composition=composition,
    agent_factory=_production_agent,
    replay_recipes=replay_recipes,
)

_REQUIRED_GOVERNANCE_ROUTES = frozenset({"/reviews", "/reviews/claim", "/feedback"})
_REQUIRED_RESEARCH_ROUTES = frozenset(
    {
        "/research/projects",
        "/research/projects/{project_id}",
        "/research/projects/{project_id}/sessions",
        "/research/projects/{project_id}/acl",
        "/research/projects/{project_id}/acl/{principal_id}",
        "/research/sessions/{session_id}",
        "/research/sessions/{session_id}/turns",
        "/research/sessions/{session_id}/close",
        "/research/capabilities",
        "/research/runtime",
        "/research/query",
        "/research/results",
        "/research/results/{result_id}",
        "/research/results/{result_id}/history",
        "/research/reports",
        "/research/reports/{report_id}",
        "/research/reports/{report_id}/markdown",
        "/research/replay",
        "/research/replay/{result_id}",
        "/research/capsules",
        "/research/capsules/preflight",
        "/research/capsules/{capsule_id}",
        "/research/capsules/{capsule_id}/verify",
        "/research/source-status",
        "/research/source-status/{source_id}",
        "/research/source-trust",
        "/research/source-trust/pending",
        "/research/source-trust/reconcile",
        "/research/source-trust/{source_id}",
        "/research/invalidate",
        "/research/stale",
        "/research/stale/acknowledge",
        "/research/recompute",
        "/research/artifacts/{kind}/{resource_id}/lineage",
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
            access_resolver=access_resolver,
            result_store=results,
        )
        acl = build_project_acl_router(
            principal_dependency=base.get_rate_limited_principal,
            acl_store=project_acls,
            access_resolver=access_resolver,
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
            invalidation_store=invalidations,
            replay_recipe_store=replay_recipes,
            access_resolver=access_resolver,
        )
        answer_history = build_research_answer_history_router(
            principal_dependency=base.get_rate_limited_principal,
            result_store=results,
            replacement_store=replacements,
            access_resolver=access_resolver,
        )
        report = build_research_report_router(
            principal_dependency=base.get_rate_limited_principal,
            workspace_store=workspace,
            result_store=results,
            report_store=reports,
            invalidation_store=invalidations,
            access_resolver=access_resolver,
        )
        replay = build_replay_router(
            principal_dependency=base.get_rate_limited_principal,
            replay_recipe_store=replay_recipes,
            access_resolver=access_resolver,
        )
        capsule = build_research_capsule_router(
            principal_dependency=base.get_rate_limited_principal,
            workspace_store=workspace,
            result_store=results,
            capsule_store=capsules,
            code_revision=code_revision,
            replay_recipe_store=replay_recipes,
            invalidation_store=invalidations,
            access_resolver=access_resolver,
        )
        capsule_verification = build_research_capsule_verification_router(
            principal_dependency=base.get_rate_limited_principal,
            workspace_store=workspace,
            result_store=results,
            capsule_store=capsules,
            code_revision=code_revision,
            invalidation_store=invalidations,
            access_resolver=access_resolver,
        )
        invalidation = build_invalidation_router(
            principal_dependency=base.get_rate_limited_principal, store=invalidations
        )
        lineage = build_artifact_lineage_router(
            principal_dependency=base.get_rate_limited_principal,
            replacements=replacements,
        )
        trust = build_source_trust_router(
            principal_dependency=base.get_rate_limited_principal,
            store=source_trust,
            invalidation_store=invalidations,
        )
        for router in (
            research,
            acl,
            runtime,
            query,
            answer_history,
            report,
            replay,
            capsule,
            capsule_verification,
            invalidation,
            lineage,
            trust,
        ):
            _append_missing_routes(router)
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
    "access_resolver",
    "app",
    "capsules",
    "code_revision",
    "composition",
    "feedback",
    "invalidations",
    "project_acls",
    "recompute_executor",
    "replacements",
    "replay_recipes",
    "reports",
    "results",
    "reviews",
    "source_trust",
    "workspace",
]
