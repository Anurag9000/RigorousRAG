from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tools.evidence_graph_jobs import (
    EvidenceGraphJob,
    EvidenceGraphJobJournal,
    deterministic_graph_job_id,
)

A = "a" * 64
B = "b" * 64


def generation(state="active", sequence=3, sparse=7):
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


def test_job_identity_and_timestamp_hardening():
    job = EvidenceGraphJob.from_generation(generation(), now=10.0)
    assert job.job_id == deterministic_graph_job_id(
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=3,
        source_state="active",
        content_sha256=A,
        profile_fingerprint=B,
        sparse_generation=7,
    )
    for bad in (math.nan, math.inf, -1.0, True):
        with pytest.raises(ValueError):
            EvidenceGraphJob.from_generation(generation(), now=bad)


def test_seed_claim_complete_and_idempotent(tmp_path):
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    job = EvidenceGraphJob.from_generation(generation(), now=1.0)
    assert journal.seed(job).job_id == job.job_id
    assert journal.seed(job).job_id == job.job_id
    claimed = journal.claim(
        owner_id="alice", worker_id="w1", lease_seconds=10, now=2.0
    )
    assert claimed and claimed.state == "running" and claimed.attempt_count == 1
    completed = journal.complete(
        claimed.job_id, worker_id="w1", graph_digest="c" * 64, now=3.0
    )
    assert completed.state == "completed" and completed.graph_digest == "c" * 64
    assert journal.claim(owner_id="alice", worker_id="w2", now=4.0) is None


def test_expired_lease_reclaim_and_attempt_ceiling(tmp_path):
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    job = EvidenceGraphJob.from_generation(generation(), max_attempts=2, now=1.0)
    journal.seed(job)
    first = journal.claim(
        owner_id="alice", worker_id="w1", lease_seconds=2, now=2.0
    )
    assert first
    assert journal.claim(owner_id="alice", worker_id="w2", now=3.0) is None
    second = journal.claim(owner_id="alice", worker_id="w2", now=5.0)
    assert second and second.attempt_count == 2 and second.lease_owner == "w2"
    journal.fail(
        second.job_id, worker_id="w2", failure_type="RuntimeError", now=6.0
    )
    assert journal.claim(owner_id="alice", worker_id="w3", now=7.0) is None


def test_failed_retry_and_cancel_are_owner_scoped(tmp_path):
    journal = EvidenceGraphJobJournal(tmp_path / "jobs.sqlite3")
    job = journal.seed(EvidenceGraphJob.from_generation(generation(), now=1.0))
    running = journal.claim(owner_id="alice", worker_id="w", now=2.0)
    assert running
    failed = journal.fail(
        job.job_id, worker_id="w", failure_type="ValueError", now=3.0
    )
    assert failed.failure_type == "ValueError"
    with pytest.raises(RuntimeError):
        journal.retry_failed(job.job_id, owner_id="bob", now=4.0)
    planned = journal.retry_failed(job.job_id, owner_id="alice", now=4.0)
    assert planned.state == "planned" and planned.attempt_count == 0
    cancelled = journal.cancel(job.job_id, owner_id="alice", now=5.0)
    assert cancelled.state == "cancelled"


def test_deleted_generation_job_contract():
    job = EvidenceGraphJob.from_generation(
        generation(state="deleted", sparse=0), now=1.0
    )
    assert job.source_state == "deleted" and job.sparse_generation == 0
    with pytest.raises(ValueError):
        deterministic_graph_job_id(
            owner_id="alice",
            doc_id="doc-1",
            source_sequence=1,
            source_state="deleted",
            content_sha256=A,
            profile_fingerprint=B,
            sparse_generation=1,
        )
