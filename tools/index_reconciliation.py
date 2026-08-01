"""Bounded dry-run repair planning for vector/sparse/manifest drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tools.three_store_coordinator import (
    AuthoritativeIndexCoordinator,
    ReconciliationReport,
)


@dataclass(frozen=True)
class RepairAction:
    doc_id: str
    action: str
    reason: str
    automatic: bool
    destructive: bool


def _actions(
    documents: Iterable[str],
    action: str,
    reason: str,
    *,
    automatic: bool,
    destructive: bool,
) -> list[RepairAction]:
    return [
        RepairAction(doc_id, action, reason, automatic, destructive)
        for doc_id in sorted(set(documents))
    ]


def plan_repairs(report: ReconciliationReport) -> tuple[RepairAction, ...]:
    """Return deterministic recommendations; no state is modified."""

    actions: list[RepairAction] = []
    actions += _actions(
        report.deleted_but_present,
        "delete_store_residue",
        "The durable current generation is deleted but store rows remain.",
        automatic=True,
        destructive=True,
    )
    actions += _actions(
        report.vector_only,
        "reindex_from_retained_source",
        "The vector store has no paired sparse generation.",
        automatic=False,
        destructive=False,
    )
    actions += _actions(
        report.sparse_only,
        "reindex_from_retained_source",
        "The sparse store has no paired vector generation.",
        automatic=False,
        destructive=False,
    )
    actions += _actions(
        report.store_pair_without_manifest,
        "reindex_or_adopt_after_source_verification",
        "Both stores exist but no durable manifest proves their generation.",
        automatic=False,
        destructive=False,
    )
    actions += _actions(
        report.manifest_without_store_pair,
        "reindex_from_retained_source",
        "The manifest is active but one or both stores are absent.",
        automatic=False,
        destructive=False,
    )
    actions += _actions(
        report.metadata_mismatch,
        "shadow_reindex_then_cutover",
        "Store counts, hashes, profile fingerprints, or generations disagree.",
        automatic=False,
        destructive=False,
    )
    actions += _actions(
        report.inspection_failed,
        "operator_corruption_inspection",
        "The document could not be safely inspected for reconciliation.",
        automatic=False,
        destructive=False,
    )
    return tuple(sorted(actions, key=lambda item: (item.doc_id, item.action)))


def apply_deleted_residue_repairs(
    coordinator: AuthoritativeIndexCoordinator,
    report: ReconciliationReport,
    *,
    confirmation: str,
    maximum: int = 100,
) -> tuple[str, ...]:
    """Apply only the narrow safe repair class with exact confirmation."""

    if confirmation != "DELETE_DELETED_GENERATION_RESIDUE":
        raise ValueError(
            "The exact deleted-residue confirmation phrase is required."
        )
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or not 1 <= maximum <= 1_000
    ):
        raise ValueError("maximum must be an integer between 1 and 1000.")
    repaired: list[str] = []
    for doc_id in report.deleted_but_present[:maximum]:
        if coordinator.delete_document(
            owner_id=report.owner_id,
            doc_id=doc_id,
            audit_metadata={"repair": "deleted_generation_residue"},
        ):
            repaired.append(doc_id)
    return tuple(repaired)


__all__ = ["RepairAction", "apply_deleted_residue_repairs", "plan_repairs"]
