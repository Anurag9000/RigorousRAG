"""Research capsule-store decorator that binds exact hydrology derivation generations."""
from __future__ import annotations

from typing import Any, Mapping

from tools.research_capsule import CapsuleReference, ReplayStep, ResearchCapsule
from tools.research_capsule_store import ResearchCapsuleStore, StoredResearchCapsule
from tools.research_result_store import ResearchResultStore

_ALLOWED_ARTIFACTS = frozenset({"evidence_projection", "evidence_report"})


def _sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    digest = value.strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        return ""
    return digest


def _hydrology_references(result: Any) -> tuple[CapsuleReference, ...]:
    bindings: dict[tuple[str, str, str], CapsuleReference] = {}
    for citation in result.citations:
        metadata = getattr(citation, "metadata", None)
        if not isinstance(metadata, Mapping) or metadata.get("derived_evidence") is not True:
            continue
        kind = str(metadata.get("artifact_kind") or "").strip()
        if kind not in _ALLOWED_ARTIFACTS:
            continue
        logical_id = str(metadata.get("artifact_id") or "").strip()
        fingerprint = _sha(metadata.get("artifact_fingerprint"))
        if not logical_id or len(logical_id) > 500 or not fingerprint:
            raise ValueError("derived hydrology citation is missing an exact artifact identity")
        key = (kind, logical_id, fingerprint)
        if key in bindings:
            continue
        ref_id = f"hydrology:{len(bindings)}"
        bindings[key] = CapsuleReference(
            ref_id,
            "generation",
            fingerprint,
            version=logical_id,
            metadata={"artifact_kind": kind, "binding": "derived_hydrology_generation"},
        )
    return tuple(bindings.values())


def augment_capsule_with_hydrology(capsule: ResearchCapsule, result: Any) -> ResearchCapsule:
    additions = _hydrology_references(result)
    if not additions:
        return capsule
    existing = {item.ref_id for item in capsule.references}
    if any(item.ref_id in existing for item in additions):
        raise RuntimeError("hydrology capsule reference ID collision")
    hydrology_ids = tuple(item.ref_id for item in additions)
    steps: list[ReplayStep] = []
    attached = False
    for step in capsule.replay_steps:
        if step.operation == "research_query" and not attached:
            steps.append(
                ReplayStep(
                    step_id=step.step_id,
                    operation=step.operation,
                    input_ref_ids=tuple((*step.input_ref_ids, *hydrology_ids)),
                    output_ref_ids=step.output_ref_ids,
                    capability_ref_id=step.capability_ref_id,
                    policy_ref_id=step.policy_ref_id,
                    deterministic=step.deterministic,
                    seed=step.seed,
                )
            )
            attached = True
        else:
            steps.append(step)
    if not attached:
        raise RuntimeError("research capsule has no research_query replay step")
    return ResearchCapsule(
        capsule_id=capsule.capsule_id,
        project_id=capsule.project_id,
        run_id=capsule.run_id,
        code_revision=capsule.code_revision,
        references=tuple((*capsule.references, *additions)),
        replay_steps=tuple(steps),
        created_at=capsule.created_at,
        schema_version=capsule.schema_version,
        notes=tuple((*capsule.notes, "Exact hydrology derivation generations are bound as replay inputs.")),
    )


class HydrologyAwareCapsuleStore(ResearchCapsuleStore):
    """Delegate persistence while enriching new manifests from immutable result citations."""

    def __init__(self, inner: ResearchCapsuleStore, results: ResearchResultStore) -> None:
        if inner is None or results is None:
            raise ValueError("capsule and result stores are required")
        self._inner = inner
        self._results = results

    def put(
        self,
        owner_id: str,
        *,
        project_id: str,
        session_id: str,
        result_id: str,
        capsule: ResearchCapsule,
        supersedes_capsule_id: str = "",
    ) -> StoredResearchCapsule:
        result = self._results.get(owner_id, result_id)
        enriched = augment_capsule_with_hydrology(capsule, result)
        return self._inner.put(
            owner_id,
            project_id=project_id,
            session_id=session_id,
            result_id=result_id,
            capsule=enriched,
            supersedes_capsule_id=supersedes_capsule_id,
        )

    def get(self, owner_id: str, capsule_id: str) -> StoredResearchCapsule:
        return self._inner.get(owner_id, capsule_id)

    def list(self, owner_id: str, *, project_id: str | None = None, result_id: str | None = None, limit: int = 100) -> tuple[StoredResearchCapsule, ...]:
        return self._inner.list(owner_id, project_id=project_id, result_id=result_id, limit=limit)


__all__ = ["HydrologyAwareCapsuleStore", "augment_capsule_with_hydrology"]
