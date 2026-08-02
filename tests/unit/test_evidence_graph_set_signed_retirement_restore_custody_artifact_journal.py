from __future__ import annotations

import os
import sqlite3
from dataclasses import replace

import pytest

from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_contracts import (
    RestoreCustodyArtifactAttempt,
    deterministic_custody_artifact_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_journal_boundary import (
    GovernedRestoreCustodyArtifactJournal,
)


def attempt(*, now: float = 1.0, max_attempts: int = 3):
    return RestoreCustodyArtifactAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        backup_path_digest="3" * 64,
        receipt_path_digest="4" * 64,
        max_attempts=max_attempts,
        now=now,
    )


def test_artifact_identity_is_deterministic_and_scope_bound():
    first = attempt()
    second = attempt(now=9.0)
    assert first.artifact_id == second.artifact_id
    assert first.artifact_id == deterministic_custody_artifact_id(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        backup_path_digest="3" * 64,
        receipt_path_digest="4" * 64,
    )
    changed = RestoreCustodyArtifactAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        backup_path_digest="3" * 64,
        receipt_path_digest="5" * 64,
        now=1.0,
    )
    assert changed.artifact_id != first.artifact_id
    with pytest.raises(ValueError, match="immutable artifact scope"):
        replace(first, artifact_id="f" * 64)


def test_phase_guarded_journal_runs_completed_lifecycle(tmp_path):
    journal = GovernedRestoreCustodyArtifactJournal(tmp_path / "artifacts.sqlite3")
    seeded = journal.seed(attempt())
    claimed = journal.claim(
        seeded.artifact_id,
        worker_id="worker",
        lease_seconds=30,
        now=2.0,
    )
    assert claimed.state == "running"
    assert claimed.phase == "planned"
    assert claimed.attempt_count == 1

    with pytest.raises(RuntimeError, match="publication intent"):
        journal.complete(
            seeded.artifact_id,
            worker_id="worker",
            backup_sha256="5" * 64,
            backup_size_bytes=100,
            receipt_digest="6" * 64,
            receipt_actor_id="actor",
            receipt_binding_method="process_environment",
            receipt_binding_digest="7" * 64,
            now=3.0,
        )

    intent = journal.record_publication_intent(
        seeded.artifact_id,
        worker_id="worker",
        now=3.0,
    )
    assert intent.phase == "publication_intent"
    completed = journal.complete(
        seeded.artifact_id,
        worker_id="worker",
        backup_sha256="5" * 64,
        backup_size_bytes=100,
        receipt_digest="6" * 64,
        receipt_actor_id="actor",
        receipt_binding_method="process_environment",
        receipt_binding_digest="7" * 64,
        now=4.0,
    )
    assert completed.state == "completed"
    assert completed.phase == "verified"
    assert completed.disposition == "paired"
    assert completed.lease_owner is None
    assert journal.next_claimable_id(owner_id="alice", now=40.0) is None


def test_orphan_lifecycle_requires_intent_and_is_terminal(tmp_path):
    journal = GovernedRestoreCustodyArtifactJournal(tmp_path / "artifacts.sqlite3")
    seeded = journal.seed(attempt())
    journal.claim(seeded.artifact_id, worker_id="worker", now=2.0)
    with pytest.raises(RuntimeError, match="publication intent"):
        journal.orphan(
            seeded.artifact_id,
            worker_id="worker",
            disposition="backup_without_receipt",
            backup_sha256="5" * 64,
            backup_size_bytes=100,
            now=3.0,
        )
    journal.record_publication_intent(
        seeded.artifact_id,
        worker_id="worker",
        now=3.0,
    )
    orphaned = journal.orphan(
        seeded.artifact_id,
        worker_id="worker",
        disposition="backup_without_receipt",
        backup_sha256="5" * 64,
        backup_size_bytes=100,
        now=4.0,
    )
    assert orphaned.state == "orphaned"
    assert orphaned.phase == "observed"
    assert orphaned.disposition == "backup_without_receipt"
    with pytest.raises(RuntimeError, match="not retryable"):
        journal.retry(
            orphaned.artifact_id,
            owner_id="alice",
            confirm_artifact_id=orphaned.artifact_id,
            now=5.0,
        )


def test_expired_claim_recovery_and_retry_ceiling_preserve_phase(tmp_path):
    journal = GovernedRestoreCustodyArtifactJournal(tmp_path / "artifacts.sqlite3")
    seeded = journal.seed(attempt(max_attempts=2))
    journal.claim(
        seeded.artifact_id,
        worker_id="one",
        lease_seconds=2,
        now=2.0,
    )
    journal.record_publication_intent(
        seeded.artifact_id,
        worker_id="one",
        now=3.0,
    )
    reclaimed = journal.claim(
        seeded.artifact_id,
        worker_id="two",
        lease_seconds=10,
        now=5.0,
    )
    assert reclaimed.phase == "publication_intent"
    assert reclaimed.attempt_count == 2
    failed = journal.fail(
        seeded.artifact_id,
        worker_id="two",
        failure_type="StorageFailure",
        now=6.0,
    )
    assert failed.state == "failed"
    assert failed.phase == "publication_intent"
    with pytest.raises(RuntimeError, match="attempt ceiling"):
        journal.retry(
            seeded.artifact_id,
            owner_id="alice",
            confirm_artifact_id=seeded.artifact_id,
            now=7.0,
        )


def test_exact_cancel_only_before_publication_intent(tmp_path):
    journal = GovernedRestoreCustodyArtifactJournal(tmp_path / "artifacts.sqlite3")
    seeded = journal.seed(attempt())
    with pytest.raises(ValueError, match="confirmation"):
        journal.cancel(
            seeded.artifact_id,
            owner_id="alice",
            confirm_artifact_id="f" * 64,
            now=2.0,
        )
    cancelled = journal.cancel(
        seeded.artifact_id,
        owner_id="alice",
        confirm_artifact_id=seeded.artifact_id,
        now=2.0,
    )
    assert cancelled.state == "cancelled"

    second = journal.seed(
        RestoreCustodyArtifactAttempt.create(
            owner_id="alice",
            snapshot_digest="a" * 64,
            target_path_digest="b" * 64,
            backup_path_digest="c" * 64,
            receipt_path_digest="d" * 64,
            now=1.0,
        )
    )
    journal.claim(second.artifact_id, worker_id="worker", now=2.0)
    journal.record_publication_intent(
        second.artifact_id,
        worker_id="worker",
        now=3.0,
    )
    journal.fail(
        second.artifact_id,
        worker_id="worker",
        failure_type="Stopped",
        now=4.0,
    )
    with pytest.raises(RuntimeError, match="cannot be cancelled"):
        journal.cancel(
            second.artifact_id,
            owner_id="alice",
            confirm_artifact_id=second.artifact_id,
            now=5.0,
        )


def test_database_identity_and_row_tampering_fail_closed(tmp_path):
    path = tmp_path / "artifacts.sqlite3"
    journal = GovernedRestoreCustodyArtifactJournal(path)
    value = journal.seed(attempt())
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        journal.get(value.artifact_id)

    second_path = tmp_path / "second.sqlite3"
    second = GovernedRestoreCustodyArtifactJournal(second_path)
    value = second.seed(attempt())
    with second._lock, second._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_restore_custody_artifacts "
            "SET target_path_digest=? WHERE artifact_id=?",
            ("f" * 64, value.artifact_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        second.get(value.artifact_id)


def test_list_is_owner_scoped_state_filtered_and_bounded(tmp_path):
    journal = GovernedRestoreCustodyArtifactJournal(tmp_path / "artifacts.sqlite3")
    first = journal.seed(attempt())
    second = journal.seed(
        RestoreCustodyArtifactAttempt.create(
            owner_id="alice",
            snapshot_digest="a" * 64,
            target_path_digest="b" * 64,
            backup_path_digest="c" * 64,
            receipt_path_digest="d" * 64,
            now=2.0,
        )
    )
    assert {value.artifact_id for value in journal.list(owner_id="alice", limit=10)} == {
        first.artifact_id,
        second.artifact_id,
    }
    assert journal.list(owner_id="alice", state="failed", limit=10) == ()
    with pytest.raises(ValueError, match="between"):
        journal.list(owner_id="alice", limit=0)
