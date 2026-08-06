from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evidence_graph_compaction import (
    EvidenceGraphCompactionStore,
    compact_evidence_graph_retention_plan,
)
from tools.evidence_graph_jobs import EvidenceGraphJob, deterministic_graph_job_id
from tools.evidence_graph_operations import (
    EvidenceGraphRetentionCandidate,
    EvidenceGraphRetentionPlan,
)

A = "a" * 64
B = "b" * 64


def _job(
    sequence: int,
    state: str,
    *,
    graph_digest: str | None = None,
    updated_at: float = 10.0,
) -> EvidenceGraphJob:
    job_id = deterministic_graph_job_id(
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=sequence,
        source_state="active",
        content_sha256=A,
        profile_fingerprint=B,
        sparse_generation=sequence + 10,
    )
    return EvidenceGraphJob(
        job_id=job_id,
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
        graph_digest=graph_digest if state == "completed" else None,
        failure_type=None,
        created_at=1.0,
        updated_at=updated_at,
    )


class Journal:
    def __init__(self, jobs: tuple[EvidenceGraphJob, ...]) -> None:
        self.jobs = {job.job_id: job for job in jobs}

    def get(self, job_id: str) -> EvidenceGraphJob | None:
        return self.jobs.get(job_id)


class Generations:
    def current(self, **_kwargs):
        return SimpleNamespace(
            owner_id="alice",
            doc_id="doc-1",
            sequence=5,
            state="active",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=15,
        )


class Graphs:
    def __init__(self) -> None:
        self.values = {
            4: SimpleNamespace(generation=4, graph_digest="4" * 64),
        }
        self.current_value = SimpleNamespace(generation=5, graph_digest="5" * 64)
        self.deleted: list[int] = []

    def current(self, **_kwargs):
        return self.current_value

    def get(self, **kwargs):
        try:
            return self.values[kwargs["generation"]]
        except KeyError as exc:
            raise KeyError(kwargs["generation"]) from exc

    def delete_generation(self, **kwargs) -> bool:
        value = self.values.get(kwargs["generation"])
        if value is None:
            return False
        if value.graph_digest != kwargs["confirm_graph_digest"]:
            raise RuntimeError("digest mismatch")
        del self.values[kwargs["generation"]]
        self.deleted.append(kwargs["generation"])
        return True


def _plan(*jobs: EvidenceGraphJob) -> EvidenceGraphRetentionPlan:
    return EvidenceGraphRetentionPlan(
        owner_id="alice",
        min_age_seconds=20.0,
        scanned_count=len(jobs),
        candidates=tuple(
            EvidenceGraphRetentionCandidate(
                job_id=job.job_id,
                state=job.state,
                source_sequence=job.source_sequence,
                age_seconds=90.0,
                reason_codes=("terminal",),
            )
            for job in jobs
        ),
        retained_current_or_recent_job_ids=(),
        retained_failed_or_running_job_ids=(),
        retained_missing_or_mismatched_graph_job_ids=(),
        generated_at=100.0,
    )


def _compact(
    *,
    job: EvidenceGraphJob,
    plan: EvidenceGraphRetentionPlan,
    graphs: Graphs,
    store: EvidenceGraphCompactionStore,
):
    return compact_evidence_graph_retention_plan(
        plan=plan,
        journal=Journal((job,)),
        generations=Generations(),
        graphs=graphs,
        compactions=store,
        confirm_plan_digest=plan.plan_digest,
        confirm_job_ids=[job.job_id],
        now=101.0,
    )


def test_verified_historical_graph_is_deleted_and_receipted(tmp_path):
    job = _job(4, "completed", graph_digest="4" * 64)
    plan = _plan(job)
    graphs = Graphs()
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")

    result = _compact(job=job, plan=plan, graphs=graphs, store=store)

    assert graphs.deleted == [4]
    assert result.deleted_graph_generation_job_ids == (job.job_id,)
    assert store.get(job.job_id).phase == "completed"


def test_cancelled_job_keeps_audit_row_and_records_no_graph_deletion(tmp_path):
    job = _job(6, "cancelled")
    plan = _plan(job)
    graphs = Graphs()
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")

    result = _compact(job=job, plan=plan, graphs=graphs, store=store)

    assert graphs.deleted == []
    assert result.retained_job_audit_only_ids == (job.job_id,)


def test_exact_plan_and_job_confirmations_are_required(tmp_path):
    job = _job(4, "completed", graph_digest="4" * 64)
    plan = _plan(job)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")

    with pytest.raises(ValueError, match="plan_digest"):
        compact_evidence_graph_retention_plan(
            plan=plan,
            journal=Journal((job,)),
            generations=Generations(),
            graphs=Graphs(),
            compactions=store,
            confirm_plan_digest="f" * 64,
            confirm_job_ids=[job.job_id],
            now=101.0,
        )
    with pytest.raises(ValueError, match="every plan candidate"):
        compact_evidence_graph_retention_plan(
            plan=plan,
            journal=Journal((job,)),
            generations=Generations(),
            graphs=Graphs(),
            compactions=store,
            confirm_plan_digest=plan.plan_digest,
            confirm_job_ids=["9" * 64],
            now=101.0,
        )


def test_resume_after_graph_delete_before_receipt_completion(tmp_path):
    job = _job(4, "completed", graph_digest="4" * 64)
    plan = _plan(job)
    graphs = Graphs()
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    store.begin(job=job, plan_digest=plan.plan_digest, now=100.0)
    graphs.delete_generation(
        owner_id="alice",
        doc_id="doc-1",
        generation=4,
        confirm_graph_digest="4" * 64,
    )

    result = _compact(job=job, plan=plan, graphs=graphs, store=store)

    assert result.completed_job_ids == (job.job_id,)
    assert store.get(job.job_id).phase == "completed"


def test_current_authoritative_job_is_refused(tmp_path):
    job = _job(5, "completed", graph_digest="5" * 64)
    plan = _plan(job)
    graphs = Graphs()
    graphs.values[5] = graphs.current_value
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")

    with pytest.raises(RuntimeError, match="authoritative-current"):
        _compact(job=job, plan=plan, graphs=graphs, store=store)


def test_completed_compaction_is_idempotent(tmp_path):
    job = _job(4, "completed", graph_digest="4" * 64)
    plan = _plan(job)
    graphs = Graphs()
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")

    first = _compact(job=job, plan=plan, graphs=graphs, store=store)
    second = _compact(job=job, plan=plan, graphs=graphs, store=store)

    assert first.completed_job_ids == (job.job_id,)
    assert second.completed_job_ids == ()
    assert second.already_completed_job_ids == (job.job_id,)
    assert graphs.deleted == [4]


def test_missing_graph_without_prior_intent_is_refused(tmp_path):
    job = _job(4, "completed", graph_digest="4" * 64)
    plan = _plan(job)
    graphs = Graphs()
    del graphs.values[4]
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")

    with pytest.raises(RuntimeError, match="without a resumable compaction intent"):
        _compact(job=job, plan=plan, graphs=graphs, store=store)


def test_compaction_receipts_can_be_filtered_by_owner_and_phase(tmp_path):
    job = _job(6, "cancelled")
    plan = _plan(job)
    store = EvidenceGraphCompactionStore(tmp_path / "compaction.sqlite3")
    _compact(job=job, plan=plan, graphs=Graphs(), store=store)

    values = store.list(owner_id="alice", phase="completed")

    assert tuple(value.job_id for value in values) == (job.job_id,)
    assert store.list(owner_id="bob", phase="completed") == ()
