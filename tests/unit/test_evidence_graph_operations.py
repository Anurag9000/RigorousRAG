from __future__ import annotations

from types import SimpleNamespace

from tools.evidence_graph_jobs import EvidenceGraphJob, deterministic_graph_job_id
from tools.evidence_graph_operations import (
    audit_evidence_graph_jobs,
    plan_evidence_graph_job_retention,
)

A = "a" * 64
B = "b" * 64


def job(sequence, state, *, attempts=0, maximum=3, updated=10.0, digest=None, lease=None):
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
        attempt_count=attempts,
        max_attempts=maximum,
        lease_owner="worker" if state == "running" else None,
        lease_expires_at=lease if state == "running" else None,
        graph_digest=digest if state == "completed" else None,
        failure_type="RuntimeError" if state == "failed" else None,
        created_at=1.0,
        updated_at=updated,
    )


class Journal:
    def __init__(self, jobs):
        self.jobs = tuple(jobs)

    def list(self, **kwargs):
        return self.jobs[: kwargs["limit"]]


class Generations:
    def current(self, **kwargs):
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
    def __init__(self, historical):
        self.historical = historical
        self.current_value = historical[5]

    def get(self, **kwargs):
        value = self.historical.get(kwargs["generation"])
        if value is None:
            raise KeyError(kwargs["generation"])
        return value

    def current(self, **kwargs):
        return self.current_value


def fixtures():
    current = job(5, "completed", digest="5" * 64)
    stale = job(4, "completed", digest="4" * 64)
    missing = job(3, "completed", digest="3" * 64)
    retryable = job(2, "failed", attempts=1)
    dead = job(1, "failed", attempts=3)
    expired = job(6, "running", attempts=1, lease=50.0)
    planned = job(7, "planned")
    cancelled = job(8, "cancelled")
    graphs = Graphs(
        {
            5: SimpleNamespace(generation=5, graph_digest="5" * 64),
            4: SimpleNamespace(generation=4, graph_digest="4" * 64),
        }
    )
    return (
        Journal((current, stale, missing, retryable, dead, expired, planned, cancelled)),
        Generations(),
        graphs,
        {item.source_sequence: item for item in (current, stale, missing, retryable, dead, expired, planned, cancelled)},
    )


def test_operational_audit_classifies_queue_and_artifact_health():
    journal, generations, graphs, jobs = fixtures()
    report = audit_evidence_graph_jobs(
        owner_id="alice",
        journal=journal,
        generations=generations,
        graphs=graphs,
        now=100.0,
    )
    assert report.scanned_count == 8
    assert report.state_counts == {
        "planned": 1,
        "running": 1,
        "completed": 3,
        "failed": 2,
        "cancelled": 1,
    }
    assert report.expired_running_job_ids == (jobs[6].job_id,)
    assert report.retryable_failed_job_ids == (jobs[2].job_id,)
    assert report.dead_letter_job_ids == (jobs[1].job_id,)
    assert set(report.superseded_nonterminal_job_ids) == {
        jobs[1].job_id,
        jobs[2].job_id,
        jobs[6].job_id,
        jobs[7].job_id,
    }
    assert report.current_completed_job_ids == (jobs[5].job_id,)
    assert report.stale_completed_job_ids == (jobs[4].job_id,)
    assert report.missing_or_mismatched_graph_job_ids == (jobs[3].job_id,)
    assert len(report.report_digest) == 64


def test_retention_plan_is_conservative_and_non_destructive():
    journal, generations, graphs, jobs = fixtures()
    plan = plan_evidence_graph_job_retention(
        owner_id="alice",
        journal=journal,
        generations=generations,
        graphs=graphs,
        min_age_seconds=20.0,
        now=100.0,
    )
    candidate_ids = {item.job_id for item in plan.candidates}
    assert candidate_ids == {jobs[4].job_id, jobs[8].job_id}
    assert jobs[5].job_id in plan.retained_current_or_recent_job_ids
    assert jobs[3].job_id in plan.retained_missing_or_mismatched_graph_job_ids
    assert jobs[1].job_id in plan.retained_failed_or_running_job_ids
    stale = next(item for item in plan.candidates if item.job_id == jobs[4].job_id)
    assert "historical_graph_digest_verified" in stale.reason_codes
    assert len(plan.plan_digest) == 64


def test_recent_terminal_jobs_are_not_candidates():
    journal, generations, graphs, _jobs = fixtures()
    plan = plan_evidence_graph_job_retention(
        owner_id="alice",
        journal=journal,
        generations=generations,
        graphs=graphs,
        min_age_seconds=1000.0,
        now=100.0,
    )
    assert plan.candidates == ()
