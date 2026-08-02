"""Exact-generation reconciliation for the derived provenance evidence graph.

This module is intentionally operator driven. It derives only document/section
structure from the authoritative sparse snapshot and never invents semantic
support, contradiction, entity, method, dataset, or citation relations.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from tools.evidence_graph_jobs import EvidenceGraphJob, EvidenceGraphJobJournal
from tools.evidence_graph_types import (
    EvidenceEdge,
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_edge_id,
    deterministic_node_id,
)
from tools.index_coordinator import _document_lock


class EvidenceGraphReconciliationError(RuntimeError):
    """Raised when an exact-generation derived graph cannot be reconciled."""


def _same_generation(job: EvidenceGraphJob, generation: Any) -> bool:
    return bool(
        generation is not None
        and getattr(generation, "owner_id", None) == job.owner_id
        and getattr(generation, "doc_id", None) == job.doc_id
        and getattr(generation, "sequence", None) == job.source_sequence
        and getattr(generation, "state", None) == job.source_state
        and getattr(generation, "content_sha256", None) == job.content_sha256
        and getattr(generation, "profile_fingerprint", None)
        == job.profile_fingerprint
        and getattr(generation, "sparse_generation", None)
        == job.sparse_generation
    )


def _document_node(job: EvidenceGraphJob, *, metadata: Mapping[str, Any]) -> EvidenceNode:
    node_id = deterministic_node_id(
        owner_id=job.owner_id,
        doc_id=job.doc_id,
        generation=job.source_sequence,
        node_type="document",
        natural_key="document",
    )
    return EvidenceNode(
        node_id=node_id,
        owner_id=job.owner_id,
        doc_id=job.doc_id,
        generation=job.source_sequence,
        node_type="document",
        natural_key="document",
        label=job.doc_id,
        text="",
        page_number=None,
        section=None,
        metadata=dict(metadata),
    )


def _section_node(job: EvidenceGraphJob, field: Any, index: int) -> EvidenceNode:
    field_id = getattr(field, "field_id", None)
    field_type = getattr(field, "field_type", None)
    text = getattr(field, "text", None)
    position = getattr(field, "position", None)
    token_count = getattr(field, "token_count", None)
    page_number = getattr(field, "page_number", None)
    section = getattr(field, "section", None)
    metadata = getattr(field, "metadata", {})
    if not isinstance(field_id, str) or not field_id.strip():
        raise RuntimeError("authoritative sparse field ID is invalid.")
    if not isinstance(field_type, str) or not field_type.strip():
        raise RuntimeError("authoritative sparse field type is invalid.")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("authoritative sparse field text is invalid.")
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise RuntimeError("authoritative sparse field position is invalid.")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise RuntimeError("authoritative sparse token count is invalid.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise RuntimeError("authoritative sparse field metadata is invalid.")
    natural_key = f"sparse-field:{field_id.strip()}"
    label = (
        section.strip()
        if isinstance(section, str) and section.strip()
        else f"{field_type.strip()} {index + 1}"
    )
    return EvidenceNode(
        node_id=deterministic_node_id(
            owner_id=job.owner_id,
            doc_id=job.doc_id,
            generation=job.source_sequence,
            node_type="section",
            natural_key=natural_key,
        ),
        owner_id=job.owner_id,
        doc_id=job.doc_id,
        generation=job.source_sequence,
        node_type="section",
        natural_key=natural_key,
        label=label,
        text=text.strip(),
        page_number=page_number,
        section=section.strip() if isinstance(section, str) and section.strip() else None,
        metadata={
            "derived_from_authoritative_sparse": True,
            "field_id": field_id.strip(),
            "field_type": field_type.strip(),
            "position": position,
            "token_count": token_count,
            **({} if metadata is None else dict(metadata)),
        },
    )


def build_structural_graph(
    job: EvidenceGraphJob,
    *,
    sparse_snapshot: Any | None,
    now: float | None = None,
) -> EvidenceGraphBatch:
    """Build document/section structure only for one immutable generation job."""

    if not isinstance(job, EvidenceGraphJob):
        raise ValueError("job must be EvidenceGraphJob.")
    created_at = time.time() if now is None else now
    if job.source_state == "deleted":
        if sparse_snapshot is not None:
            raise RuntimeError("deleted authoritative generation retained a sparse snapshot.")
        document = _document_node(
            job,
            metadata={
                "authoritative_state": "deleted",
                "derived_tombstone": True,
                "source_sequence": job.source_sequence,
            },
        )
        return EvidenceGraphBatch(
            owner_id=job.owner_id,
            doc_id=job.doc_id,
            generation=job.source_sequence,
            content_sha256=job.content_sha256,
            profile_fingerprint=job.profile_fingerprint,
            nodes=(document,),
            edges=(),
            created_at=created_at,
        )

    if sparse_snapshot is None:
        raise RuntimeError("active authoritative generation has no sparse snapshot.")
    if (
        getattr(sparse_snapshot, "owner_id", None) != job.owner_id
        or getattr(sparse_snapshot, "doc_id", None) != job.doc_id
        or getattr(sparse_snapshot, "generation", None) != job.sparse_generation
        or getattr(sparse_snapshot, "profile_fingerprint", None)
        != job.profile_fingerprint
    ):
        raise RuntimeError("authoritative sparse snapshot identity differs from job.")
    sparse_metadata = getattr(sparse_snapshot, "metadata", {})
    if sparse_metadata is not None and not isinstance(sparse_metadata, Mapping):
        raise RuntimeError("authoritative sparse document metadata is invalid.")
    for key in ("content_sha256", "source_content_sha256"):
        declared = (sparse_metadata or {}).get(key)
        if declared is not None and declared != job.content_sha256:
            raise RuntimeError("authoritative sparse content hash differs from job.")
    fields = tuple(getattr(sparse_snapshot, "fields", ()))
    if not fields:
        raise RuntimeError("active authoritative sparse snapshot has no fields.")

    document = _document_node(
        job,
        metadata={
            "authoritative_state": job.source_state,
            "content_sha256": job.content_sha256,
            "profile_fingerprint": job.profile_fingerprint,
            "sparse_generation": job.sparse_generation,
            "derived_semantic_edges": False,
        },
    )
    nodes: list[EvidenceNode] = [document]
    edges: list[EvidenceEdge] = []
    seen_fields: set[str] = set()
    for index, field in enumerate(
        sorted(
            fields,
            key=lambda item: (
                getattr(item, "position", 0),
                str(getattr(item, "field_id", "")),
            ),
        )
    ):
        node = _section_node(job, field, index)
        if node.natural_key in seen_fields:
            raise RuntimeError("authoritative sparse snapshot contains duplicate field IDs.")
        seen_fields.add(node.natural_key)
        nodes.append(node)
        relation_key = f"document-contains:{node.natural_key}"
        edges.append(
            EvidenceEdge(
                edge_id=deterministic_edge_id(
                    owner_id=job.owner_id,
                    doc_id=job.doc_id,
                    generation=job.source_sequence,
                    source_node_id=document.node_id,
                    target_node_id=node.node_id,
                    edge_type="contains",
                    relation_key=relation_key,
                ),
                owner_id=job.owner_id,
                doc_id=job.doc_id,
                generation=job.source_sequence,
                source_node_id=document.node_id,
                target_node_id=node.node_id,
                edge_type="contains",
                relation_key=relation_key,
                weight=1.0,
                metadata={"derived_from_authoritative_sparse": True},
            )
        )
    return EvidenceGraphBatch(
        owner_id=job.owner_id,
        doc_id=job.doc_id,
        generation=job.source_sequence,
        content_sha256=job.content_sha256,
        profile_fingerprint=job.profile_fingerprint,
        nodes=tuple(nodes),
        edges=tuple(edges),
        created_at=created_at,
    )


def seed_current_graph_job(
    *,
    owner_id: str,
    doc_id: str,
    generations: Any,
    journal: EvidenceGraphJobJournal,
    max_attempts: int = 3,
    now: float | None = None,
) -> EvidenceGraphJob:
    generation = generations.current(owner_id=owner_id, doc_id=doc_id)
    if generation is None:
        raise KeyError((owner_id, doc_id))
    return journal.seed(
        EvidenceGraphJob.from_generation(
            generation,
            max_attempts=max_attempts,
            now=now,
        )
    )


def reconcile_graph_job(
    job: EvidenceGraphJob,
    *,
    worker_id: str,
    journal: EvidenceGraphJobJournal,
    generations: Any,
    sparse: Any,
    graphs: Any,
    now: float | None = None,
) -> EvidenceGraphJob:
    """Reconcile a leased job against the exact authoritative current generation."""

    if not isinstance(job, EvidenceGraphJob) or job.state != "running":
        raise ValueError("reconciliation requires a running EvidenceGraphJob.")
    if job.lease_owner != worker_id:
        raise ValueError("worker_id does not own the job lease.")
    timestamp = time.time() if now is None else now
    try:
        with _document_lock(job.owner_id, job.doc_id):
            before = generations.current(owner_id=job.owner_id, doc_id=job.doc_id)
            if not _same_generation(job, before):
                raise RuntimeError("authoritative generation changed before graph reconciliation.")
            sparse_snapshot = (
                None
                if job.source_state == "deleted"
                else sparse.snapshot_document(owner_id=job.owner_id, doc_id=job.doc_id)
            )
            batch = build_structural_graph(job, sparse_snapshot=sparse_snapshot, now=timestamp)
            middle = generations.current(owner_id=job.owner_id, doc_id=job.doc_id)
            if not _same_generation(job, middle):
                raise RuntimeError("authoritative generation changed during graph construction.")
            current_graph = graphs.current(owner_id=job.owner_id, doc_id=job.doc_id)
            if current_graph is not None and current_graph.generation > job.source_sequence:
                raise RuntimeError("evidence graph current generation is newer than the job.")
            expected_current = 0 if current_graph is None else current_graph.generation
            if (
                current_graph is not None
                and current_graph.generation == job.source_sequence
                and current_graph.graph_digest == batch.graph_digest
            ):
                published = current_graph
            else:
                published = graphs.commit(
                    batch,
                    make_current=True,
                    expected_current_generation=expected_current,
                    now=timestamp,
                )
            after = generations.current(owner_id=job.owner_id, doc_id=job.doc_id)
            if not _same_generation(job, after):
                raise RuntimeError("authoritative generation changed after graph publication.")
            return journal.complete(
                job.job_id,
                worker_id=worker_id,
                graph_digest=published.graph_digest,
                now=timestamp,
            )
    except Exception as exc:
        failure_type = type(exc).__name__
        try:
            journal.fail(
                job.job_id,
                worker_id=worker_id,
                failure_type=failure_type,
                now=timestamp,
            )
        except Exception as journal_exc:
            raise EvidenceGraphReconciliationError(
                "evidence graph reconciliation and failure recording both failed."
            ) from journal_exc
        raise EvidenceGraphReconciliationError(
            f"evidence graph reconciliation failed ({failure_type})."
        ) from exc


def execute_next_graph_job(
    *,
    owner_id: str,
    worker_id: str,
    journal: EvidenceGraphJobJournal,
    generations: Any,
    sparse: Any,
    graphs: Any,
    lease_seconds: int = 60,
    now: float | None = None,
) -> EvidenceGraphJob | None:
    job = journal.claim(
        owner_id=owner_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        now=now,
    )
    if job is None:
        return None
    return reconcile_graph_job(
        job,
        worker_id=worker_id,
        journal=journal,
        generations=generations,
        sparse=sparse,
        graphs=graphs,
        now=now,
    )


__all__ = [
    "EvidenceGraphReconciliationError",
    "build_structural_graph",
    "execute_next_graph_job",
    "reconcile_graph_job",
    "seed_current_graph_job",
]
