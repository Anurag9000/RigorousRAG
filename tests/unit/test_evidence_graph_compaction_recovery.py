from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evidence_graph_compaction import EvidenceGraphCompactionStore
from tools.evidence_graph_compaction_reconciliation import (
    reconcile_evidence_graph_compactions,
)
from tools.evidence_graph_compaction_recovery import (
    EvidenceGraphCompactionRecoveryJournal,
    recover_reconciled_compaction_receipts_durable,
)
from tools.evidence_graph_jobs import EvidenceGraphJob, deterministic_graph_job_id

A = "a" * 64
B = "b" * 64


def job(sequence: int) -> EvidenceGraphJob:
    doc_id = f"doc-{sequence}"
    return EvidenceGraphJob(
        job_id=deterministic_graph_job_id(
            owner_id="alice",
            doc_id=doc_id,
            source_sequence=sequence,
            source_state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=sequence + 10,
        ),
        owner_id="alice",
        doc_id=doc_id,
        source_sequence=sequence,
        source_state="active",
        content_sha256=A,
        profile_fingerprint=B,
        sparse_generation=sequence + 10,
        state="completed",
        attempt_count=0,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        graph_digest=f"{sequence}" * 64,
        failure_type=None,
        created_at=1.0,
        updated_at=2.0,
    )


class Journal:
    def __init__(self, *values: EvidenceGraphJob) -> None:
        self.values = {value.job_id: value for value in values}

    def get(self, job_id: str):
        return self.values.get(job_id)


class Generations:
    def current(self, **kwargs):
        return SimpleNamespace(
            owner_id="alice",
            doc_id=kwargs["doc_id"],
            sequence=99,
            state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=109,
        )


class Graphs:
    def current(self, **_kwargs):
        return SimpleNamespace(generation=99, graph_digest="9" * 64)

    def get(self, **kwargs):
        raise KeyError(kwargs["generation"])


def prepare(tmp_path, *values: EvidenceGraphJob):
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    for selected in values:
        store.begin(job=selected, plan_digest="1" * 64, now=3.0)
    report = reconcile_evidence_graph_compactions(
        owner_id="alice",
        compactions=store,
        journal=Journal(*values),
        generations=Generations(),
        graphs=Graphs(),
        now=5.0,
    )
    recovery = EvidenceGraphCompactionRecoveryJournal(store.path)
    return store, recovery, report


def recover(store, recovery, report, values, *, now=6.0):
    return recover_reconciled_compaction_receipts_durable(
        report=report,
        compactions=store,
        journal=Journal(*values),
        generations=Generations(),
        graphs=Graphs(),
        recovery_journal=recovery,
        confirm_report_digest=report.report_digest,
        confirm_job_ids=report.recoverable_job_ids,
        actor_id="operator-1",
        reason="resume-after-verified-delete",
        now=now,
    )


def test_recovery_persists_operator_intent_and_terminal_receipt(tmp_path):
    selected = job(4)
    store, recovery, report = prepare(tmp_path, selected)

    receipt = recover(store, recovery, report, [selected])

    assert receipt.phase == "completed"
    assert receipt.attempt_count == 1
    assert receipt.confirmed_job_ids == (selected.job_id,)
    assert receipt.completed_job_ids == (selected.job_id,)
    assert receipt.already_completed_job_ids == ()
    assert receipt.actor_id == "operator-1"
    assert receipt.reason == "resume-after-verified-delete"
    assert receipt.result_digest is not None
    assert len(receipt.receipt_digest) == 64
    assert store.get(selected.job_id).phase == "completed"
    assert recovery.get(receipt.recovery_id) == receipt


def test_completed_recovery_replay_is_idempotent(tmp_path):
    selected = job(4)
    store, recovery, report = prepare(tmp_path, selected)
    first = recover(store, recovery, report, [selected], now=6.0)

    second = recover(store, recovery, report, [selected], now=7.0)

    assert second == first
    assert second.attempt_count == 1
    assert len(recovery.list(owner_id="alice", phase="completed")) == 1


def test_partial_failure_is_durable_and_retry_accounts_for_completed_work(
    tmp_path, monkeypatch
):
    values = [job(4), job(5)]
    store, recovery, report = prepare(tmp_path, *values)
    failure_id = sorted(report.recoverable_job_ids)[1]
    original_complete = store.complete

    def flaky_complete(job_id, **kwargs):
        if job_id == failure_id:
            raise RuntimeError("injected failure")
        return original_complete(job_id, **kwargs)

    monkeypatch.setattr(store, "complete", flaky_complete)
    with pytest.raises(RuntimeError, match="injected failure"):
        recover(store, recovery, report, values, now=6.0)

    planned = recovery.list(owner_id="alice", phase="planned")
    assert len(planned) == 1
    assert planned[0].attempt_count == 1
    assert planned[0].last_error_type == "RuntimeError"
    assert (
        sum(
            store.get(job_id).phase == "completed"
            for job_id in report.recoverable_job_ids
        )
        == 1
    )

    monkeypatch.setattr(store, "complete", original_complete)
    receipt = recover(store, recovery, report, values, now=7.0)

    assert receipt.phase == "completed"
    assert receipt.attempt_count == 2
    assert receipt.last_error_type is None
    assert tuple(
        sorted(receipt.completed_job_ids + receipt.already_completed_job_ids)
    ) == report.recoverable_job_ids
    assert len(receipt.completed_job_ids) == 1
    assert len(receipt.already_completed_job_ids) == 1


def test_recovery_intent_requires_exact_confirmations_before_persistence(tmp_path):
    values = [job(4), job(5)]
    store, recovery, report = prepare(tmp_path, *values)

    with pytest.raises(ValueError, match="every recoverable"):
        recovery.begin(
            report=report,
            confirm_report_digest=report.report_digest,
            confirm_job_ids=[report.recoverable_job_ids[0]],
            actor_id="operator-1",
            reason="resume-after-verified-delete",
            now=6.0,
        )

    assert recovery.list(owner_id="alice") == ()
    assert all(store.get(item.job_id).phase == "planned" for item in values)


def test_recovery_journal_detects_database_substitution(tmp_path):
    selected = job(4)
    store, recovery, report = prepare(tmp_path, selected)
    recover(store, recovery, report, [selected])
    path = recovery.path
    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"")

    with pytest.raises(RuntimeError, match="identity changed"):
        recovery.list(owner_id="alice")
