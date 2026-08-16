"""Dependency registration helpers for finalized research results, reports and capsules."""

from __future__ import annotations

from typing import Any, Mapping

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.research_capsule_store import StoredResearchCapsule
from tools.research_report_store import StoredResearchReport
from tools.research_result_provenance import session_binding
from tools.research_result_store import StoredResearchResult
from tools.runtime_composition import RuntimeComposition

_HYDROLOGY_CITATION_KINDS = {
    "evidence_projection": "hydrology_projection",
    "evidence_report": "hydrology_report",
}


def _safe_metadata_id(value: Any, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        return ""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        return ""
    return cleaned


def _register_unique(
    upstreams: list[tuple[DependencyRef, str]],
    seen: set[tuple[str, str]],
    ref: DependencyRef,
    relation: str,
) -> None:
    key = (ref.kind, ref.resource_id)
    if key in seen:
        return
    seen.add(key)
    upstreams.append((ref, relation))


def _hydrology_dependency_identity(metadata: Mapping[str, Any]) -> tuple[str, str] | None:
    if metadata.get("derived_evidence") is not True:
        return None
    artifact_kind = _safe_metadata_id(metadata.get("artifact_kind"), 64)
    dependency_kind = _HYDROLOGY_CITATION_KINDS.get(artifact_kind)
    if dependency_kind is None:
        return None
    fingerprint = _safe_metadata_id(metadata.get("artifact_fingerprint"), 64).lower()
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        raise RuntimeError("server-derived hydrology evidence is missing an exact artifact fingerprint")
    return dependency_kind, fingerprint


def _register_hydrology_citation_dependency(
    upstreams: list[tuple[DependencyRef, str]],
    seen: set[tuple[str, str]],
    citation: Any,
) -> None:
    metadata = getattr(citation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return
    identity = _hydrology_dependency_identity(metadata)
    if identity is None:
        return
    dependency_kind, fingerprint = identity
    _register_unique(
        upstreams,
        seen,
        DependencyRef(dependency_kind, fingerprint),
        "derived_from_hydrology_artifact",
    )


def register_result_dependencies(
    store: DependencyInvalidationStore,
    owner_id: str,
    result: StoredResearchResult,
    *,
    composition: RuntimeComposition,
) -> None:
    if not isinstance(store, DependencyInvalidationStore):
        raise TypeError("store must be DependencyInvalidationStore")
    if not isinstance(result, StoredResearchResult):
        raise TypeError("result must be StoredResearchResult")
    downstream = DependencyRef("result", result.result_id)
    upstreams: list[tuple[DependencyRef, str]] = []
    seen: set[tuple[str, str]] = set()

    for citation in result.citations:
        source_id = _safe_metadata_id(citation.source_id or citation.url)
        if source_id:
            _register_unique(upstreams, seen, DependencyRef("source", source_id), "cites")
        doc_id = _safe_metadata_id(citation.doc_id or "", 256)
        if doc_id:
            _register_unique(upstreams, seen, DependencyRef("document", doc_id), "retrieved_from")
        _register_hydrology_citation_dependency(upstreams, seen, citation)

    model_id = _safe_metadata_id(result.model, 256)
    if model_id:
        _register_unique(upstreams, seen, DependencyRef("model", model_id), "generated_with")

    selected_policy = composition.selected_capabilities.get("policy", "")
    if selected_policy:
        descriptor = composition.capabilities.active(selected_policy)
        if descriptor is not None:
            _register_unique(upstreams, seen, DependencyRef("policy", descriptor.fingerprint), "routed_with")

    _register_unique(upstreams, seen, DependencyRef("runtime_config", composition.config.fingerprint), "configured_with")
    _register_unique(upstreams, seen, DependencyRef("capability_registry", composition.capabilities.fingerprint), "resolved_with")

    metadata: Mapping[str, Any] = result.metadata
    binding = session_binding(metadata)
    if binding is not None:
        _register_unique(upstreams, seen, DependencyRef("project", binding["project_id"]), "created_in_project")
        _register_unique(upstreams, seen, DependencyRef("session", binding["session_id"]), "created_in_session")
        _register_unique(
            upstreams,
            seen,
            DependencyRef("session_snapshot", binding["session_fingerprint_before"]),
            "executed_against_session_snapshot",
        )

    for key in (
        "generation_id",
        "generation_fingerprint",
        "index_generation",
        "index_generation_sha256",
        "retrieval_generation",
    ):
        value = _safe_metadata_id(metadata.get(key), 256)
        if value:
            _register_unique(upstreams, seen, DependencyRef("index_generation", value), "retrieved_from_generation")
    plan = _safe_metadata_id(metadata.get("plan_fingerprint") or metadata.get("plan_sha256"), 256)
    if plan:
        _register_unique(upstreams, seen, DependencyRef("plan", plan), "executed_plan")

    admissibility = metadata.get("admissibility_gate")
    if isinstance(admissibility, Mapping):
        policy_sha = _safe_metadata_id(admissibility.get("policy_sha256"), 64)
        if len(policy_sha) == 64:
            _register_unique(upstreams, seen, DependencyRef("admissibility_policy", policy_sha), "published_under_admissibility_policy")
        revision_ids = admissibility.get("trust_revision_ids", ())
        if isinstance(revision_ids, (list, tuple)):
            for raw_revision in revision_ids[:100]:
                revision_id = _safe_metadata_id(raw_revision, 64)
                if len(revision_id) != 64:
                    continue
                _register_unique(upstreams, seen, DependencyRef("source_trust_revision", revision_id), "admitted_under_source_review")
        evaluated_source_ids = admissibility.get("evaluated_source_ids", ())
        if isinstance(evaluated_source_ids, (list, tuple)):
            for raw_source in evaluated_source_ids[:100]:
                source_id = _safe_metadata_id(raw_source, 1000)
                if not source_id:
                    continue
                _register_unique(upstreams, seen, DependencyRef("source_trust_subject", source_id), "evaluated_for_admissibility")

    store.register_dependencies(owner_id, downstream=downstream, upstreams=tuple(upstreams))


def register_report_dependencies(store: DependencyInvalidationStore, owner_id: str, report: StoredResearchReport) -> None:
    if not isinstance(store, DependencyInvalidationStore):
        raise TypeError("store must be DependencyInvalidationStore")
    if not isinstance(report, StoredResearchReport):
        raise TypeError("report must be StoredResearchReport")
    store.register_dependencies(
        owner_id,
        downstream=DependencyRef("report", report.report_id),
        upstreams=(
            (DependencyRef("result", report.result_id), "derived_from_result"),
            (DependencyRef("project", report.project_id), "scoped_by_project"),
        ),
    )


def _register_capsule_hydrology_reference(
    upstreams: list[tuple[DependencyRef, str]],
    seen: set[tuple[str, str]],
    reference: Any,
) -> bool:
    if not str(reference.ref_id).startswith("hydrology:"):
        return False
    if reference.kind != "generation":
        raise RuntimeError("hydrology capsule reference must be a generation reference")
    metadata = reference.metadata
    if not isinstance(metadata, Mapping):
        raise RuntimeError("hydrology capsule reference metadata is invalid")
    artifact_kind = _safe_metadata_id(metadata.get("artifact_kind"), 64)
    dependency_kind = _HYDROLOGY_CITATION_KINDS.get(artifact_kind)
    if dependency_kind is None:
        raise RuntimeError("hydrology capsule reference has an unsupported artifact kind")
    _register_unique(
        upstreams,
        seen,
        DependencyRef(dependency_kind, reference.content_sha256),
        "captures_hydrology_generation",
    )
    return True


def register_capsule_dependencies(store: DependencyInvalidationStore, owner_id: str, capsule: StoredResearchCapsule) -> None:
    """Attach an immutable capsule to the stale-artifact graph."""

    if not isinstance(store, DependencyInvalidationStore):
        raise TypeError("store must be DependencyInvalidationStore")
    if not isinstance(capsule, StoredResearchCapsule):
        raise TypeError("capsule must be StoredResearchCapsule")

    downstream = DependencyRef("capsule", capsule.capsule_id)
    upstreams: list[tuple[DependencyRef, str]] = [
        (DependencyRef("result", capsule.result_id), "captures_result"),
        (DependencyRef("project", capsule.project_id), "scoped_by_project"),
        (DependencyRef("session", capsule.session_id), "captures_session"),
        (DependencyRef("code_revision", capsule.capsule.code_revision), "executed_code"),
    ]
    seen = {(item.kind, item.resource_id) for item, _ in upstreams}

    for reference in capsule.capsule.references:
        if _register_capsule_hydrology_reference(upstreams, seen, reference):
            continue
        if reference.ref_id == "runtime-config":
            _register_unique(upstreams, seen, DependencyRef("runtime_config", reference.content_sha256), "configured_with")
        elif reference.ref_id == "capability-registry":
            _register_unique(upstreams, seen, DependencyRef("capability_registry", reference.content_sha256), "resolved_with")
        elif reference.kind == "generation":
            _register_unique(upstreams, seen, DependencyRef("index_generation", reference.content_sha256), "captures_generation")
        elif reference.kind == "policy":
            _register_unique(upstreams, seen, DependencyRef("policy", reference.content_sha256), "captures_policy")
        elif reference.ref_id.endswith(":retrieval-profile"):
            _register_unique(upstreams, seen, DependencyRef("retrieval_profile", reference.content_sha256), "captures_retrieval_profile")
        elif reference.ref_id == "model-identifier":
            _register_unique(upstreams, seen, DependencyRef("model_identity", reference.content_sha256), "captures_model_identity")
        elif reference.kind == "source":
            _register_unique(upstreams, seen, DependencyRef("evidence_content", reference.content_sha256), "captures_evidence_content")

    store.register_dependencies(owner_id, downstream=downstream, upstreams=tuple(upstreams))


def stale_reasons(store: DependencyInvalidationStore, owner_id: str, artifact: DependencyRef, *, maximum: int = 100) -> tuple[Mapping[str, Any], ...]:
    if not 1 <= maximum <= 1000:
        raise ValueError("maximum is invalid")
    rows = store.list_stale(owner_id, kind=artifact.kind, limit=10_000)
    output: list[Mapping[str, Any]] = []
    for row in rows:
        if row.artifact != artifact:
            continue
        output.append({"event_sha256": row.triggering_event_sha256, "reason": row.reason, "stale_at": row.stale_at, "replacement_id": row.replacement_id})
        if len(output) >= maximum:
            break
    return tuple(output)


__all__ = ["register_capsule_dependencies", "register_report_dependencies", "register_result_dependencies", "stale_reasons"]
