from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tools.evidence_graph_jobs import EvidenceGraphJob, EvidenceGraphJobJournal
from tools.evidence_graph_reconcile import (
    EvidenceGraphReconciliationError,
    build_structural_graph,
    execute_next_graph_job,
    seed_current_graph_job,
)

A = "a" * 64
B = "b" * 64


@dataclass
class Field:
    field_id: str
    field_type: str
    text: str
    position: int
    token_count: int
    page_number: int | None = None
    section: str | None = None
    metadata: dict | None = None


class Generations:
    def __init__(self, record):
        self.record = record

    def current(self, **kwargs):
        return self.record


class Sparse:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def snapshot_document(self, **kwargs):
        return self.snapshot


class Graphs:
    def __init__(self):
        self.value = None
        self.commits = 0

    def current(self, **kwargs):
        return self.value

    def commit(self, batch, **kwargs):
        self.commits += 1
        self.value = batch
        return batch


def record(state="active", sequence=4, sparse=9):
    return SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        sequence=sequence,
        state=state,
        content_sha256=A,
        profile_fingerprint=B,
        vector_rows=2 if state != "deleted" else 0,
        sparse_generation=sparse if state != "deleted" else 0,
    )


def snapshot():
    return SimpleNamespace(
        owner_id="alice",
        doc_id="doc-1",
        generation=9,
        profile_fingerprint=B,
        metadata={"content_sha256": A},
        fields=(
            Field("f1", "title", "Title", 0, 1, 1, "Intro", {}),
            Field("f2", "body", "Body text", 1, 2, 2, "Methods", {}),
        ),
    )


def test_structural_active_graph_has_only_contains_edges():
    job = EvidenceGraphJob.from_generation(record(), now=1.0)
    batch = build_structural_graph(job, sparse_snapshot=snapshot(), now=2.0)
    assert len(batch.nodes) == 3 and len(batch.edges) == 2
    assert {edge.edge_type for edge in batch.edges} == {"contains"}
    assert all(node.node_type in {"document", "section"} for node in batch.nodes)


def test_deleted_generation_builds_tombstone_graph():
    job = EvidenceGraphJob.from_generation(
        record(state="deleted", sparse=0), now=1.0
    )
    batch = build_structural_graph(job, sparse_snapshot=None, now=2.0)
    assert len(batch.nodes) == 1 and batch.edges == ()
    assert batch.nodes[0].metadata["derived_tombstone"] is True


def test_reconcile_exact_generation_is_idempotent(tmp_path):
    generations = Generations(record())
    sparse = Sparse(snapshot())
    graphs = Graphs()
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    seeded = seed_current_graph_job(
        owner_id="alice",
        doc_id="doc-1",
        generations=generations,
        journal=journal,
        now=1.0,
    )
    result = execute_next_graph_job(
        owner_id="alice",
        worker_id="worker",
        journal=journal,
        generations=generations,
        sparse=sparse,
        graphs=graphs,
        now=2.0,
    )
    assert result and result.state == "completed" and result.graph_digest
    assert graphs.commits == 1
    assert (
        journal.seed(EvidenceGraphJob.from_generation(record(), now=5.0)).job_id
        == seeded.job_id
    )


def test_stale_generation_fails_with_generic_type(tmp_path):
    generations = Generations(record())
    sparse = Sparse(snapshot())
    graphs = Graphs()
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    job = seed_current_graph_job(
        owner_id="alice",
        doc_id="doc-1",
        generations=generations,
        journal=journal,
        now=1.0,
    )
    generations.record = record(sequence=5, sparse=10)
    with pytest.raises(EvidenceGraphReconciliationError):
        execute_next_graph_job(
            owner_id="alice",
            worker_id="worker",
            journal=journal,
            generations=generations,
            sparse=sparse,
            graphs=graphs,
            now=2.0,
        )
    stored = journal.get(job.job_id)
    assert stored and stored.state == "failed" and stored.failure_type == "RuntimeError"
    assert graphs.commits == 0


def test_sparse_identity_mismatch_fails_before_publish(tmp_path):
    bad = snapshot()
    bad.generation = 10
    generations = Generations(record())
    graphs = Graphs()
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    seed_current_graph_job(
        owner_id="alice",
        doc_id="doc-1",
        generations=generations,
        journal=journal,
        now=1.0,
    )
    with pytest.raises(EvidenceGraphReconciliationError):
        execute_next_graph_job(
            owner_id="alice",
            worker_id="worker",
            journal=journal,
            generations=generations,
            sparse=Sparse(bad),
            graphs=graphs,
            now=2.0,
        )
    assert graphs.commits == 0
