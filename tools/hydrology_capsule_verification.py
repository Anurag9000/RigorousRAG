"""Hydrology-aware private-safe verification of research capsule references."""
from __future__ import annotations

from tools.capsule_replay import verify_capsule
from tools.hydrology_store import HydrologyArtifactStore
from tools.research_capsule import CapsuleReference
from tools.research_capsule_store import StoredResearchCapsule
from tools.research_capsule_verification import (
    ResearchCapsuleDigestAuthority,
    StoredCapsuleVerification,
    WorkspaceStore,
    _revision,
)
from tools.research_result_store import ResearchResultStore

_ALLOWED = frozenset({"evidence_projection", "evidence_report"})


class HydrologyCapsuleDigestAuthority(ResearchCapsuleDigestAuthority):
    def __init__(
        self,
        stored: StoredResearchCapsule,
        *,
        workspace_store: WorkspaceStore,
        result_store: ResearchResultStore,
        hydrology_store: HydrologyArtifactStore,
    ) -> None:
        super().__init__(stored, workspace_store=workspace_store, result_store=result_store)
        self.hydrology_store = hydrology_store

    def _hydrology_digest(self, reference: CapsuleReference) -> str | None:
        if not reference.ref_id.startswith("hydrology:"):
            return None
        kind = reference.metadata.get("artifact_kind", "")
        if kind not in _ALLOWED:
            return None
        logical_id = reference.version
        if not logical_id:
            return None
        try:
            envelope = self.hydrology_store.get(
                self.stored.owner_id,
                self.stored.project_id,
                kind,
                logical_id,
                fingerprint=reference.content_sha256,
            )
        except (KeyError, ValueError, RuntimeError):
            return None
        return envelope.fingerprint if envelope.fingerprint == reference.content_sha256 else None

    def digest(self, reference: CapsuleReference) -> str | None:
        if reference.ref_id.startswith("hydrology:"):
            return self._hydrology_digest(reference)
        return super().digest(reference)


def verify_stored_capsule_with_hydrology(
    stored: StoredResearchCapsule,
    *,
    workspace_store: WorkspaceStore,
    result_store: ResearchResultStore,
    hydrology_store: HydrologyArtifactStore,
    deployment_code_revision: str = "",
) -> StoredCapsuleVerification:
    authority = HydrologyCapsuleDigestAuthority(
        stored,
        workspace_store=workspace_store,
        result_store=result_store,
        hydrology_store=hydrology_store,
    )
    receipt = verify_capsule(stored.capsule, authority=authority)
    deployment_revision = _revision(deployment_code_revision)
    capsule_revision = _revision(stored.capsule.code_revision)
    if not deployment_revision:
        code_status = "unavailable"
    elif deployment_revision == capsule_revision:
        code_status = "matched"
    else:
        code_status = "mismatch"
    return StoredCapsuleVerification(
        receipt=receipt,
        code_revision_status=code_status,
        deployment_code_revision=deployment_revision,
        manifest_verified=receipt.verified,
        deployment_compatible=receipt.verified and code_status == "matched",
    )


__all__ = ["HydrologyCapsuleDigestAuthority", "verify_stored_capsule_with_hydrology"]
