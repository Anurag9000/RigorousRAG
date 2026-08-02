from __future__ import annotations

import os

import pytest

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
    deterministic_signed_retirement_restore_id,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


def attempt(*, now: float = 1.0, max_attempts: int = 3):
    return SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=2,
        max_attempts=max_attempts,
        now=now,
    )


def test_restore_identity_is_deterministic_and_scope_bound():
    first = attempt()
    second = attempt(now=9.0)
    assert first.restore_id == second.restore_id
    assert first.restore_id == deterministic_signed_retirement_restore_id(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
    )
    changed = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="3" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=2,
        now=1.0,
    )
    assert changed.restore_id != first.restore_id


def test_restore_journal_runs_monotonic_lifecycle(tmp_path):
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    seeded = journal.seed(attempt())
    claimed = journal.claim(
        seeded.restore_id,
        worker_id="worker",
        lease_seconds=30,
        now=2.0,
    )
    assert claimed.state == "running"
    assert claimed.attempt_count == 1
    assert claimed.lease_expires_at == 32.0

    committed = journal.record_target_committed(
        seeded.restore_id,
        worker_id="worker",
        target_verification_digest="3" * 64,
        now=3.0,
    )
    assert committed.phase == "target_committed"
    completed = journal.complete(
        seeded.restore_id,
        worker_id="worker",
        target_verification_digest="3" * 64,
        now=4.0,
    )
    assert completed.state == "completed"
    assert completed.phase == "verified"
    assert completed.completed_at == 4.0
    assert completed.lease_owner is None
    with pytest.raises(RuntimeError, match="not claimable"):
        journal.claim(
            seeded.restore_id,
            worker_id="other",
            lease_seconds=30,
            now=40.0,
        )


def test_restore_retry_preserves_committed_phase_and_ceiling(tmp_path):
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    seeded = journal.seed(attempt(max_attempts=2))
    journal.claim(seeded.restore_id, worker_id="worker", now=2.0)
    journal.record_target_committed(
        seeded.restore_id,
        worker_id="worker",
        target_verification_digest="3" * 64,
        now=3.0,
    )
    failed = journal.fail(
        seeded.restore_id,
        worker_id="worker",
        failure_type="Interrupted",
        now=4.0,
    )
    assert failed.phase == "target_committed"

    retried = journal.retry(
        seeded.restore_id,
        owner_id="alice",
        confirm_restore_id=seeded.restore_id,
        now=5.0,
    )
    assert retried.state == "planned"
    assert retried.phase == "target_committed"
    journal.claim(retried.restore_id, worker_id="worker", now=6.0)
    journal.fail(
        retried.restore_id,
        worker_id="worker",
        failure_type="Again",
        now=7.0,
    )
    with pytest.raises(RuntimeError, match="attempt ceiling"):
        journal.retry(
            retried.restore_id,
            owner_id="alice",
            confirm_restore_id=retried.restore_id,
            now=8.0,
        )


def test_restore_cancel_is_exact_and_only_before_target_work(tmp_path):
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    seeded = journal.seed(attempt())
    with pytest.raises(ValueError, match="confirmation"):
        journal.cancel(
            seeded.restore_id,
            owner_id="alice",
            confirm_restore_id="f" * 64,
            now=2.0,
        )
    cancelled = journal.cancel(
        seeded.restore_id,
        owner_id="alice",
        confirm_restore_id=seeded.restore_id,
        now=2.0,
    )
    assert cancelled.state == "cancelled"

    second = journal.seed(
        SignedRetirementRestoreAttempt.create(
            owner_id="alice",
            snapshot_digest="4" * 64,
            target_path_digest="2" * 64,
            snapshot_record_count=1,
            now=1.0,
        )
    )
    journal.claim(second.restore_id, worker_id="worker", now=2.0)
    journal.record_target_committed(
        second.restore_id,
        worker_id="worker",
        target_verification_digest="5" * 64,
        now=3.0,
    )
    journal.fail(
        second.restore_id,
        worker_id="worker",
        failure_type="Stopped",
        now=4.0,
    )
    with pytest.raises(RuntimeError, match="unstarted"):
        journal.cancel(
            second.restore_id,
            owner_id="alice",
            confirm_restore_id=second.restore_id,
            now=5.0,
        )


def test_restore_reclaim_tamper_and_database_identity_fail_closed(tmp_path):
    path = tmp_path / "restores.sqlite3"
    journal = SignedRetirementRestoreJournal(path)
    seeded = journal.seed(attempt())
    journal.claim(
        seeded.restore_id,
        worker_id="one",
        lease_seconds=1,
        now=2.0,
    )
    recovered = journal.claim(
        seeded.restore_id,
        worker_id="two",
        lease_seconds=10,
        now=4.0,
    )
    assert recovered.attempt_count == 2
    assert recovered.lease_owner == "two"

    with journal._lock, journal._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_retirement_restores "
            "SET snapshot_digest=? WHERE restore_id=?",
            ("f" * 64, seeded.restore_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        journal.get(seeded.restore_id)

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        journal.list(owner_id="alice")
