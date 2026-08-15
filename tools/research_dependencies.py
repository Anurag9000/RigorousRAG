"""Dependency registration helpers for finalized research results and reports."""

from __future__ import annotations

from typing import Any, Mapping

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef
from tools.research_report_store import StoredResearchReport
from tools.research_result_store import StoredResearchResult
from tools.runtime_composition import RuntimeComposition


def _safe_metadata_id(value: Any, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        return ""
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        return ""
    return cleaned


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
        if source_id and ("source", source_id) not in seen:
            seen.add(("source", source_id))
            upstreams.append((DependencyRef("source", source_id), "cites"))
        doc_id = _safe_metadata_id(citation.doc_id or "", 256)
        if doc_id and ("document", doc_id) not in seen:
            seen.add(("document", doc_id))
            upstreams.append((DependencyRef("document", doc_id), "retrieved_from"))

    model_id = _safe_metadata_id(result.model, 256)
    if model_id:
        upstreams.append((DependencyRef("model", model_id), "generated_with"))

    selected_policy = composition.selected_capabilities.get("policy", "")
    if selected_policy:
        descriptor = composition.capabilities.active(selected_policy)
        if descriptor is not None:
            upstreams.append((DependencyRef("policy", descriptor.fingerprint), "routed_with"))

    upstreams.append(
        (DependencyRef("runtime_config", composition.config.fingerprint), "configured_with")
    )
    upstreams.append(
        (
            DependencyRef("capability_registry", composition.capabilities.fingerprint),
            "resolved_with",
        )
    )

    metadata: Mapping[str, Any] = result.metadata
    for key in (
        "generation_id",
        "generation_fingerprint",
        "index_generation",
        "index_generation_sha256",
        "retrieval_generation",
    ):
        value = _safe_metadata_id(metadata.get(key), 256)
        if value:
            upstreams.append((DependencyRef("index_generation", value), "retrieved_from_generation"))
    plan = _safe_metadata_id(
        metadata.get("plan_fingerprint") or metadata.get("plan_sha256"), 256
    )
    if plan:
        upstreams.append((DependencyRef("plan", plan), "executed_plan"))

    store.register_dependencies(owner_id, downstream=downstream, upstreams=tuple(upstreams))


def register_report_dependencies(
    store: DependencyInvalidationStore,
    owner_id: str,
    report: StoredResearchReport,
) -> None:
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


def stale_reasons(
    store: DependencyInvalidationStore,
    owner_id: str,
    artifact: DependencyRef,
    *,
    maximum: int = 100,
) -> tuple[Mapping[str, Any], ...]:
    if not 1 <= maximum <= 1000:
        raise ValueError("maximum is invalid")
    # ``list_stale`` is bounded; filtering here avoids exposing SQL internals to callers.
    rows = store.list_stale(owner_id, kind=artifact.kind, limit=10_000)
    output: list[Mapping[str, Any]] = []
    for row in rows:
        if row.artifact != artifact:
            continue
        output.append(
            {
                "event_sha256": row.triggering_event_sha256,
                "reason": row.reason,
                "stale_at": row.stale_at,
                "replacement_id": row.replacement_id,
            }
        )
        if len(output) >= maximum:
            break
    return tuple(output)


__all__ = [
    "register_report_dependencies",
    "register_result_dependencies",
    "stale_reasons",
]
