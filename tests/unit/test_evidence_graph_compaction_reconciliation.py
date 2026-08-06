from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evidence_graph_compaction import EvidenceGraphCompactionStore
from tools.evidence_graph_compaction_reconciliation import (
    reconcile_evidence_graph_compactions,
    recover_reconciled_compaction_receipts,
)
from tools.evidence_graph_jobs import EvidenceGraphJob, deterministic_graph_job_id

A = "a" * 64
B = "b" * 64


def job(sequence: int, state: str = "completed", digest: str | None = None):
    return EvidenceGraphJob(
        job_id=deterministic_graph_job_id(
            owner_id="alice",
            doc_id="doc-1",
            source_sequence=sequence,
            source_state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=sequence + 10,
        ),
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=sequence,
        source_state="active",
        content_sha256=A,
        profile_fingerprint=B,
        sparse_generation=sequence + 10,
        state=state,
        attempt_count=0,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        graph_digest=digest if state == "completed" else None,
        failure_type=None,
        created_at=1.0,
        updated_at=2.0,
    )


class Journal:
    def __init__(self, *values):
        self.values = {value.job_id: value for value in values}

    def get(self, job_id):
        return self.values.get(job_id)


class Generations:
    def __init__(self, current_sequence=9):
        self.current_sequence = current_sequence

    def current(self, **_kwargs):
        return SimpleNamespace(
            owner_id="alice",
            doc_id="doc-1",
            sequence=self.current_sequence,
            state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=self.current_sequence + 10,
        )


class Graphs:
    def __init__(self, values=None, current_sequence=9):
        self.values = dict(values or {})
        self.current_value = SimpleNamespace(
            generation=current_sequence,
            graph_digest="9" * 64,
        )

    def current(self, **_kwargs):
        return self.current_value

    def get(self, **kwargs):
        try:
            return self.values[kwargs["generation"]]
        except KeyError as exc:
            raise KeyError(kwargs["generation"]) from exc


def report(store, selected_job, graphs, *, now=5.0):
    return reconcile_evidence_graph_compactions(
        owner_id="alice",
        compactions=store,
        journal=Journal(selected_job),
        generations=Generations(),
        graphs=graphs,
        now=now,
    )


def test_planned_missing_graph_is_recoverable_and_exactly_completed(tmp_path):
    selected = job(4, digest="4" * 64)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)
    graphs = Graphs()

    value = report(store, selected, graphs)

    assert value.healthy is True
    assert value.recoverable_job_ids == (selected.job_id,)
    assert value.findings[0].status == "completion_pending_after_delete"
    result = recover_reconciled_compaction_receipts(
        report=value,
        compactions=store,
        journal=Journal(selected),
        generations=Generations(),
        graphs=graphs,
        confirm_report_digest=value.report_digest,
        confirm_job_ids=[selected.job_id],
        now=6.0,
    )
    assert result.completed_job_ids == (selected.job_id,)
    assert store.get(selected.job_id).phase == "completed"


def test_audit_only_planned_receipt_is_recoverable_without_graph_access(tmp_path):
    selected = job(4, state="cancelled")
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)

    value = report(store, selected, Graphs())

    assert value.findings[0].status == "audit_only_completion_pending"
    assert value.findings[0].graph_present is None


def test_planned_existing_graph_remains_deletion_pending(tmp_path):
    selected = job(4, digest="4" * 64)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)
    graphs = Graphs({4: SimpleNamespace(generation=4, graph_digest="4" * 64)})

    value = report(store, selected, graphs)

    assert value.findings[0].status == "deletion_pending"
    assert value.recoverable_job_ids == ()


def test_completed_receipt_with_present_graph_is_a_conflict(tmp_path):
    selected = job(4, digest="4" * 64)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)
    store.complete(selected.job_id, owner_id="alice", plan_digest="1" * 64, now=4.0)
    graphs = Graphs({4: SimpleNamespace(generation=4, graph_digest="4" * 64)})

    value = report(store, selected, graphs)

    assert value.healthy is False
    assert value.conflict_job_ids == (selected.job_id,)
    assert value.findings[0].status == "completed_graph_present"


def test_digest_and_authority_conflicts_fail_closed(tmp_path):
    selected = job(4, digest="4" * 64)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)

    mismatch = report(
        store,
        selected,
        Graphs({4: SimpleNamespace(generation=4, graph_digest="5" * 64)}),
    )
    assert mismatch.findings[0].status == "graph_digest_conflict"

    authority = reconcile_evidence_graph_compactions(
        owner_id="alice",
        compactions=store,
        journal=Journal(selected),
        generations=Generations(current_sequence=4),
        graphs=Graphs(current_sequence=4),
        now=5.0,
    )
    assert authority.findings[0].status == "authoritative_current_conflict"


def test_recovery_requires_exact_report_and_job_confirmations(tmp_path):
    selected = job(4, digest="4" * 64)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)
    value = report(store, selected, Graphs())
    kwargs = dict(
        report=value,
        compactions=store,
        journal=Journal(selected),
        generations=Generations(),
        graphs=Graphs(),
        now=6.0,
    )

    with pytest.raises(ValueError, match="report_digest"):
        recover_reconciled_compaction_receipts(
            **kwargs,
            confirm_report_digest="f" * 64,
            confirm_job_ids=[selected.job_id],
        )
    with pytest.raises(ValueError, match="every recoverable"):
        recover_reconciled_compaction_receipts(
            **kwargs,
            confirm_report_digest=value.report_digest,
            confirm_job_ids=[],
        )


def test_missing_journal_and_malformed_graph_digest_are_conflicts(tmp_path):
    selected = job(4, digest="4" * 64)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=selected, plan_digest="1" * 64, now=3.0)

    missing = reconcile_evidence_graph_compactions(
        owner_id="alice",
        compactions=store,
        journal=Journal(),
        generations=Generations(),
        graphs=Graphs(),
        now=5.0,
    )
    assert missing.findings[0].status == "journal_missing_conflict"
    assert missing.conflict_count == 1
    assert missing.recoverable_count == 0

    malformed = report(
        store,
        selected,
        Graphs({4: SimpleNamespace(generation=4, graph_digest="not-a-digest")}),
    )
    assert malformed.findings[0].status == "record_corrupt"
    assert malformed.status_counts == (("record_corrupt", 1),)
