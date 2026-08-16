"""Single-point production persistence composition for governance and research state.

The selected metadata backend is an all-or-nothing deployment boundary. This prevents a
multi-replica Postgres workspace from accidentally pairing with node-local result, ACL,
review, invalidation, trust, replay, or hydrology ledgers. No driver or credential
discovery occurs here: Postgres uses the already-injected ``postgres.connection_factory``
provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.artifact_replacements import ArtifactReplacementStore
from tools.dependency_invalidation import DependencyInvalidationStore
from tools.feedback_store import FeedbackStore
from tools.hydrology_derivation_store import HydrologyDerivationStore
from tools.hydrology_derivation_store_postgres import PostgresHydrologyDerivationStore
from tools.hydrology_guarded_store import GuardedHydrologyArtifactStore
from tools.hydrology_store import HydrologyArtifactStore
from tools.hydrology_store_postgres import PostgresHydrologyArtifactStore
from tools.hydrology_store_sqlite import SQLiteHydrologyArtifactStore
from tools.hydrology_versioned_store import VersionedHydrologyArtifactStore
from tools.postgres_governance_stores import PostgresFeedbackStore, PostgresReviewStore
from tools.postgres_invalidation_store import PostgresDependencyInvalidationStore
from tools.postgres_research_stores import (
    PostgresArtifactReplacementStore,
    PostgresProjectACLStore,
    PostgresResearchCapsuleStore,
    PostgresResearchReportStore,
    PostgresResearchResultStore,
)
from tools.postgres_source_trust_store import PostgresSourceTrustStore
from tools.postgres_workspace_store import PostgresResearchWorkspaceStore
from tools.project_acl_store import ProjectACLStore
from tools.replay_recipe_store import EncryptedReplayRecipeStore
from tools.replay_runtime import build_replay_recipe_store
from tools.research_capsule_store import ResearchCapsuleStore
from tools.research_report_store import ResearchReportStore
from tools.research_result_store import ResearchResultStore
from tools.research_workspace_sqlite import SQLiteResearchWorkspaceStore
from tools.review_store import ReviewStore
from tools.runtime_providers import RuntimeProviderRegistry
from tools.source_trust_store import SourceTrustStore


@dataclass(frozen=True)
class ProductionPersistence:
    backend: str
    workspace: Any
    project_acls: ProjectACLStore
    results: ResearchResultStore
    reports: ResearchReportStore
    capsules: ResearchCapsuleStore
    invalidations: DependencyInvalidationStore
    replacements: ArtifactReplacementStore
    source_trust: SourceTrustStore
    replay_recipes: EncryptedReplayRecipeStore | None
    reviews: ReviewStore
    feedback: FeedbackStore
    hydrology: HydrologyArtifactStore
    hydrology_recipes: Any


def _root(value: str | Path) -> Path:
    root = Path(value).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _governed_hydrology(raw: HydrologyArtifactStore, invalidations: DependencyInvalidationStore) -> HydrologyArtifactStore:
    return VersionedHydrologyArtifactStore(
        GuardedHydrologyArtifactStore(raw, invalidations),
        invalidations,
    )


def build_production_persistence(
    root: str | Path,
    *,
    metadata_backend: str,
    providers: RuntimeProviderRegistry,
    postgres_schema: str = "rigorousrag",
) -> ProductionPersistence:
    selected_root = _root(root)
    backend = str(metadata_backend).strip().lower()
    if backend in {"postgres", "postgresql"}:
        connection_factory = providers.require("postgres.connection_factory")
        workspace = PostgresResearchWorkspaceStore(connection_factory, schema=postgres_schema)
        project_acls = PostgresProjectACLStore(connection_factory, schema=postgres_schema)
        results = PostgresResearchResultStore(connection_factory, schema=postgres_schema)
        reports = PostgresResearchReportStore(connection_factory, schema=postgres_schema)
        capsules = PostgresResearchCapsuleStore(connection_factory, schema=postgres_schema)
        invalidations = PostgresDependencyInvalidationStore(connection_factory, schema=postgres_schema)
        replacements = PostgresArtifactReplacementStore(connection_factory, schema=postgres_schema)
        source_trust = PostgresSourceTrustStore(connection_factory, schema=postgres_schema)
        hydrology_raw = PostgresHydrologyArtifactStore(connection_factory, schema=postgres_schema)
        hydrology_raw.initialize()
        hydrology = _governed_hydrology(hydrology_raw, invalidations)
        hydrology_recipes = PostgresHydrologyDerivationStore(connection_factory, schema=postgres_schema)
        replay_recipes = build_replay_recipe_store(
            selected_root / "research_replay.sqlite3",
            providers=providers,
            metadata_backend=backend,
            connection_factory=connection_factory,
            schema=postgres_schema,
        )
        reviews = PostgresReviewStore(connection_factory, schema=postgres_schema)
        feedback = PostgresFeedbackStore(connection_factory, schema=postgres_schema)
        return ProductionPersistence(
            backend="postgres",
            workspace=workspace,
            project_acls=project_acls,
            results=results,
            reports=reports,
            capsules=capsules,
            invalidations=invalidations,
            replacements=replacements,
            source_trust=source_trust,
            replay_recipes=replay_recipes,
            reviews=reviews,
            feedback=feedback,
            hydrology=hydrology,
            hydrology_recipes=hydrology_recipes,
        )

    if backend == "sqlite":
        invalidations = DependencyInvalidationStore(selected_root / "research_invalidation.sqlite3")
        hydrology_raw = SQLiteHydrologyArtifactStore(selected_root / "research_hydrology.sqlite3")
        hydrology = _governed_hydrology(hydrology_raw, invalidations)
        hydrology_recipes = HydrologyDerivationStore(selected_root / "research_hydrology_recipes.sqlite3")
        return ProductionPersistence(
            backend="sqlite",
            workspace=SQLiteResearchWorkspaceStore(selected_root / "research_workspace.sqlite3"),
            project_acls=ProjectACLStore(selected_root / "research_project_acl.sqlite3"),
            results=ResearchResultStore(selected_root / "research_results.sqlite3"),
            reports=ResearchReportStore(selected_root / "research_reports.sqlite3"),
            capsules=ResearchCapsuleStore(selected_root / "research_capsules.sqlite3"),
            invalidations=invalidations,
            replacements=ArtifactReplacementStore(selected_root / "research_replacements.sqlite3"),
            source_trust=SourceTrustStore(selected_root / "source_trust.sqlite3"),
            replay_recipes=build_replay_recipe_store(
                selected_root / "research_replay.sqlite3",
                providers=providers,
                metadata_backend="sqlite",
            ),
            reviews=ReviewStore(selected_root / "reviews.sqlite3"),
            feedback=FeedbackStore(selected_root / "feedback.sqlite3"),
            hydrology=hydrology,
            hydrology_recipes=hydrology_recipes,
        )

    raise RuntimeError(f"unsupported production metadata backend: {backend}")


__all__ = ["ProductionPersistence", "build_production_persistence"]
