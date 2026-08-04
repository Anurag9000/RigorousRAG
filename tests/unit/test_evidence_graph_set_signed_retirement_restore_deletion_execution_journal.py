from __future__ import annotations

import os

import pytest

from tools.evidence_graph_set_signed_retirement_restore_deletion_execution_contracts import (
    SignedRetirementRestoreDeletionAttempt,
    deterministic_restore_deletion_id,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_journal import (
    SignedRetirementRestoreDeletionJournal,
)


def attempt(*, now: float = 1.0, max_attempts: int = 3):
    return SignedRetirementRestoreDeletionAttempt.create(
        authorization_id="1" * 64,
        authorization_digest="2" * 64,
        owner_id="alice",
        restore_id="3" * 64,
        snapshot_digest="4" * 64,
        target_path_digest="5" * 64,
        restore_state="completed",
        restore_phase="verified",
        restore_record_digest="6" * 64,
        custody_id="7" * 64,
        custody_manifest_digest="8" * 64,
        max_attempts=max_attempts,
        now=now,
    )


def test_deletion_identity_is_deterministic_and_terminal_scope_bound():
    first = attempt(now=1.0)
    second = attempt(now=9.0)
    assert first.deletion_id == second.deletion_id
    assert first.deletion_id == deterministic_restore_deletion_id(
        authorization_id="1" * 64,
        authorization_digest="2" * 64,
        owner_id="alice",
        restore_id="3" * 64,
        snapshot_digest="4" * 64,
        target_path_digest="5" * 64,
        restore_record_digest="6" * 64,
        custody_manifest_digest="8" * 64,
    )
    with pytest.raises(ValueError, match="terminal"):
        SignedRetirementRestoreDeletionAttempt.create(
            authorization_id="1" * 64,
            authorization_digest="2" * 64,
            owner_id="alice",
            restore_id="3" * 64,
            snapshot_digest="4" * 64,
            target_path_digest="5" * 64,
            restore_state="running",
            restore_phase="planned",
            restore_record_digest="6" * 64,
            custody_id=None,
            custody_manifest_digest=None,
            now=1.0,
        )
    with pytest.raises(ValueError, match="custody"):
        SignedRetirementRestoreDeletionAttempt.create(
            authorization_id="1" * 64,
            authorization_digest="2" * 64,
            owner_id="alice",
            restore_id="3" * 64,
            snapshot_digest="4" * 64,
            target_path_digest="5" * 64,
            restore_state="completed",
            restore_phase="verified",
            restore_record_digest="6" * 64,
            custody_id=None,
            custody_manifest_digest=None,
            now=1.0,
        )


def test_journal_runs_monotonic_lifecycle_and_exact_completion(tmp_path):
    journal = SignedRetirementRestoreDeletionJournal(
        tmp_path / "deletions.sqlite3"
    )
    seeded = journal.seed(attempt())
    claimed = journal.claim(
        seeded.deletion_id,
        worker_id="worker",
        lease_seconds=30,
        now=2.0,
    )
    assert claimed.state == "running"
    assert claimed.attempt_count == 1
    marker = journal.record_marker_active(
        seeded.deletion_id,
        worker_id="worker",
        marker_digest="9" * 64,
        now=3.0,
    )
    assert marker.phase == "marker_active"
    deleted = journal.record_restore_deleted(
        seeded.deletion_id,
        worker_id="worker",
        marker_digest="9" * 64,
        tombstone_digest="a" * 64,
        now=4.0,
    )
    assert deleted.phase == "restore_deleted"
    completed = journal.complete(
        seeded.deletion_id,
        worker_id="worker",
        marker_digest="9" * 64,
        tombstone_digest="a" * 64,
        now=5.0,
    )
    assert completed.state == "completed"
    assert completed.phase == "verified"
    assert completed.completed_at == 5.0
    with pytest.raises(RuntimeError, match="claimable"):
        journal.claim(
            seeded.deletion_id,
            worker_id="other",
            now=40.0,
        )


def test_failure_retry_preserves_phase_and_attempt_ceiling(tmp_path):
    journal = SignedRetirementRestoreDeletionJournal(
        tmp_path / "deletions.sqlite3"
    )
    seeded = journal.seed(attempt(max_attempts=2))
    journal.claim(seeded.deletion_id, worker_id="one", now=2.0)
    journal.record_marker_active(
        seeded.deletion_id,
        worker_id="one",
        marker_digest="9" * 64,
        now=3.0,
    )
    failed = journal.fail(
        seeded.deletion_id,
        worker_id="one",
        failure_type="Stopped",
        now=4.0,
    )
    assert failed.phase == "marker_active"
    retried = journal.retry(
        seeded.deletion_id,
        owner_id="alice",
        confirm_deletion_id=seeded.deletion_id,
        now=5.0,
    )
    assert retried.phase == "marker_active"
    journal.claim(retried.deletion_id, worker_id="two", now=6.0)
    journal.fail(
        retried.deletion_id,
        worker_id="two",
        failure_type="Again",
        now=7.0,
    )
    with pytest.raises(RuntimeError, match="retryable"):
        journal.retry(
            retried.deletion_id,
            owner_id="alice",
            confirm_deletion_id=retried.deletion_id,
            now=8.0,
        )


def test_cancel_is_exact_and_only_before_marker_work(tmp_path):
    journal = SignedRetirementRestoreDeletionJournal(
        tmp_path / "deletions.sqlite3"
    )
    seeded = journal.seed(attempt())
    with pytest.raises(ValueError, match="confirmation"):
        journal.cancel(
            seeded.deletion_id,
            owner_id="alice",
            confirm_deletion_id="f" * 64,
            now=2.0,
        )
    cancelled = journal.cancel(
        seeded.deletion_id,
        owner_id="alice",
        confirm_deletion_id=seeded.deletion_id,
        now=2.0,
    )
    assert cancelled.state == "cancelled"

    other = SignedRetirementRestoreDeletionAttempt.create(
        authorization_id="a" * 64,
        authorization_digest="b" * 64,
        owner_id="alice",
        restore_id="c" * 64,
        snapshot_digest="d" * 64,
        target_path_digest="e" * 64,
        restore_state="cancelled",
        restore_phase="planned",
        restore_record_digest="f" * 64,
        custody_id=None,
        custody_manifest_digest=None,
        now=1.0,
    )
    journal.seed(other)
    journal.claim(other.deletion_id, worker_id="worker", now=2.0)
    journal.record_marker_active(
        other.deletion_id,
        worker_id="worker",
        marker_digest="1" * 64,
        now=3.0,
    )
    journal.fail(
        other.deletion_id,
        worker_id="worker",
        failure_type="Stopped",
        now=4.0,
    )
    with pytest.raises(RuntimeError, match="unstarted"):
        journal.cancel(
            other.deletion_id,
            owner_id="alice",
            confirm_deletion_id=other.deletion_id,
            now=5.0,
        )


def test_database_identity_and_stored_row_tampering_fail_closed(tmp_path):
    path = tmp_path / "deletions.sqlite3"
    journal = SignedRetirementRestoreDeletionJournal(path)
    value = journal.seed(attempt())
    with journal._lock, journal._connect() as connection:
        connection.execute(
            "UPDATE signed_retirement_restore_deletions "
            "SET restore_record_digest=? WHERE deletion_id=?",
            ("f" * 64, value.deletion_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        journal.get(value.deletion_id)

    clean = tmp_path / "clean.sqlite3"
    second = SignedRetirementRestoreDeletionJournal(clean)
    second.seed(attempt())
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(clean.read_bytes())
    os.replace(replacement, clean)
    with pytest.raises(RuntimeError, match="identity changed"):
        second.list(owner_id="alice")
