"""Hardened API plus durable governance and research routes."""
from pathlib import Path
import os

import search_agent_legacy as agent_legacy
import server as base
from fastapi import Request
from tools.agent_runtime import configure_agent_runtime
from tools.artifact_lineage_api import build_artifact_lineage_router
from tools.control_api import build_control_router
from tools.hydrology_agent_tools import register_hydrology_agent_tools
from tools.hydrology_api import build_hydrology_router
from tools.hydrology_derivation_api import build_hydrology_derivation_router
from tools.hydrology_recompute_executor import HydrologyRecomputeExecutor
from tools.hydrology_report_api import build_hydrology_report_router
from tools.hydrology_status_api import build_hydrology_status_router
from tools.invalidation_api import build_invalidation_router
from tools.production_persistence import build_production_persistence
from tools.production_recompute_executor import ProductionResearchRecomputeExecutor
from tools.project_acl_api import build_project_acl_router
from tools.replay_api import build_replay_router
from tools.research_access import ResearchAccessResolver
from tools.research_answer_history_api import build_research_answer_history_router
from tools.research_api import build_research_router
from tools.research_capsule_api import build_research_capsule_router
from tools.research_capsule_verification_api import build_research_capsule_verification_router
from tools.research_query_api import build_research_query_router
from tools.research_report_api import build_research_report_router
from tools.routed_recompute_executor import RoutedRecomputeExecutor
from tools.runtime_api import build_runtime_router
from tools.runtime_composition import build_runtime_composition
from tools.runtime_providers import runtime_providers
from tools.source_trust_api import build_source_trust_router

root = Path(os.environ.get("CLASSIC_STORAGE_DIR", "data")).resolve() / "governance"
root.mkdir(parents=True, exist_ok=True)
composition = build_runtime_composition()
code_revision = os.environ.get("RIGOROUSRAG_CODE_REVISION", "").strip()
persistence = build_production_persistence(
    root,
    metadata_backend=composition.config.storage.metadata_backend,
    providers=runtime_providers,
)
metadata_backend = persistence.backend
workspace = persistence.workspace
project_acls = persistence.project_acls
results = persistence.results
reports = persistence.reports
capsules = persistence.capsules
invalidations = persistence.invalidations
replacements = persistence.replacements
source_trust = persistence.source_trust
replay_recipes = persistence.replay_recipes
reviews = persistence.reviews
feedback = persistence.feedback
hydrology = persistence.hydrology
hydrology_recipes = persistence.hydrology_recipes
access_resolver = ResearchAccessResolver(workspace, project_acls)
register_hydrology_agent_tools(
    agent_legacy,
    store=hydrology,
    access_resolver=access_resolver,
    invalidation_store=invalidations,
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

# Recompute execution remains operator/worker-only. The public API exposes queue/stale
# metadata but never triggers model/provider work. Deterministic hydrology recomputation
# is routed through the same authoritative task ledger without model calls.
research_recompute_executor = ProductionResearchRecomputeExecutor(
    invalidations=invalidations,
    replacements=replacements,
    results=results,
    reports=reports,
    workspace=workspace,
    composition=composition,
    agent_factory=_production_agent,
    replay_recipes=replay_recipes,
)
hydrology_recompute_executor = HydrologyRecomputeExecutor(
    invalidations=invalidations,
    replacements=replacements,
    store=hydrology,
    recipes=hydrology_recipes,
)
distributed_recompute_executor = RoutedRecomputeExecutor(
    invalidations=invalidations,
    research=research_recompute_executor,
    hydrology=hydrology_recompute_executor,
)
# Compatibility alias retained for existing local research-only operator callers.
recompute_executor = research_recompute_executor

_REQUIRED_GOVERNANCE_ROUTES = frozenset({"/reviews", "/reviews/claim", "/feedback"})
_REQUIRED_RESEARCH_ROUTES = frozenset(
    {
        "/research/projects",
        "/research/projects/{project_id}",
        "/research/projects/{project_id}/sessions",
        "/research/projects/{project_id}/acl",
        "/research/projects/{project_id}/acl/{principal_id}",
        "/research/projects/{project_id}/hydrology/artifacts",
        "/research/projects/{project_id}/hydrology/artifacts/{kind}/{logical_id}",
        "/research/projects/{project_id}/hydrology/topologies/{topology_id}",
        "/research/projects/{project_id}/hydrology/packages/{package_id}",
        "/research/projects/{project_id}/hydrology/plans",
        "/research/projects/{project_id}/hydrology/projections",
        "/research/projects/{project_id}/hydrology/derivations",
        "/research/projects/{project_id}/hydrology/derive/plans",
        "/research/projects/{project_id}/hydrology/derive/projections",
        "/research/projects/{project_id}/hydrology/derive/reports",
        "/research/projects/{project_id}/hydrology/reports",
        "/research/projects/{project_id}/hydrology/reports/{report_id}/markdown",
        "/research/projects/{project_id}/hydrology/reports/{report_id}/csv",
        "/research/projects/{project_id}/hydrology/status",
        "/research/projects/{project_id}/hydrology/status/{kind}/{logical_id}",
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
            persistence_metadata={
                "metadata_backend": metadata_backend,
                "distributed_shared_state": metadata_backend == "postgres",
                "encrypted_replay_configured": replay_recipes is not None,
                "hydrology_derivation_recipes": True,
                "hydrology_capsule_generation_verification": True,
                "routed_recompute": True,
                "code_revision_configured": bool(code_revision),
            },
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
            hydrology_store=hydrology,
        )
        invalidation = build_invalidation_router(
            principal_dependency=base.get_rate_limited_principal,
            store=invalidations,
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
        # The governed hydrology store owns dependency registration and generation
        # invalidation. Pass no route-level lifecycle store here to avoid duplicate events.
        hydrology_router = build_hydrology_router(
            principal_dependency=base.get_rate_limited_principal,
            store=hydrology,
            access_resolver=access_resolver,
            invalidation_store=None,
        )
        hydrology_derivation_router = build_hydrology_derivation_router(
            principal_dependency=base.get_rate_limited_principal,
            store=hydrology,
            recipe_store=hydrology_recipes,
            access_resolver=access_resolver,
        )
        hydrology_report_router = build_hydrology_report_router(
            principal_dependency=base.get_rate_limited_principal,
            store=hydrology,
            access_resolver=access_resolver,
            invalidation_store=invalidations,
        )
        hydrology_status_router = build_hydrology_status_router(
            principal_dependency=base.get_rate_limited_principal,
            store=hydrology,
            access_resolver=access_resolver,
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
            hydrology_router,
            hydrology_derivation_router,
            hydrology_report_router,
            hydrology_status_router,
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
    "distributed_recompute_executor",
    "feedback",
    "hydrology",
    "hydrology_recipes",
    "hydrology_recompute_executor",
    "invalidations",
    "metadata_backend",
    "persistence",
    "project_acls",
    "recompute_executor",
    "replacements",
    "replay_recipes",
    "reports",
    "research_recompute_executor",
    "results",
    "reviews",
    "source_trust",
    "workspace",
]
