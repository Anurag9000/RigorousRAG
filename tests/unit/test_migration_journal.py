import os

import pytest

from tools.migration_journal import MigrationJournal
from tools.migration_types import MigrationCandidate


def candidate(*, doc_id="doc-1", eligible=True, retained=True):
    return MigrationCandidate(
        owner_id="alice",
        doc_id=doc_id,
        source_sequence=1,
        source_profile_fingerprint="a" * 64,
        target_profile_name="e5-base-v2",
        target_profile_fingerprint="b" * 64,
        retained_source=retained,
        eligible=eligible,
        reason="ready" if eligible else "retained_source_unavailable",
    )


def test_seed_is_idempotent_and_skips_ineligible_candidates(tmp_path):
    journal = MigrationJournal(tmp_path / "migration.sqlite3")
    first = journal.seed(
        [candidate(), candidate(doc_id="missing", eligible=False, retained=False)],
        now=1.0,
    )
    second = journal.seed([candidate()], now=2.0)
    assert len(first) == 1
    assert second == first
    assert first[0].state == "planned"
    assert first[0].attempt == 0
    assert "/private" not in repr(first)


def test_claim_validate_commit_and_renew_lifecycle(tmp_path):
    journal = MigrationJournal(tmp_path / "migration.sqlite3")
    task = journal.seed([candidate()], now=1.0)[0]
    claimed = journal.claim(
        owner_id="alice",
        worker_id="worker-1",
        lease_seconds=10,
        now=2.0,
    )
    assert claimed is not None
    assert claimed.state == "running" and claimed.attempt == 1
    renewed = journal.renew(
        task_id=task.task_id,
        worker_id="worker-1",
        lease_seconds=20,
        now=3.0,
    )
    assert renewed.lease_expires_at == 23.0
    validated = journal.mark_validated(
        task_id=task.task_id,
        worker_id="worker-1",
        validation_digest="c" * 64,
        now=4.0,
    )
    assert validated.state == "validated"
    committed = journal.mark_committed(
        task_id=task.task_id,
        worker_id="worker-1",
        now=5.0,
    )
    assert committed.state == "committed"
    assert committed.validation_digest == "c" * 64
    assert committed.lease_owner is None
    assert journal.claim(owner_id="alice", worker_id="worker-2", now=6.0) is None


def test_commit_requires_validation_and_active_lease(tmp_path):
    journal = MigrationJournal(tmp_path / "migration.sqlite3")
    task = journal.seed([candidate()], now=1.0)[0]
    journal.claim(owner_id="alice", worker_id="worker-1", lease_seconds=2, now=2.0)
    with pytest.raises(RuntimeError, match="validation"):
        journal.mark_committed(
            task_id=task.task_id,
            worker_id="worker-1",
            now=3.0,
        )
    with pytest.raises(RuntimeError, match="lease"):
        journal.mark_validated(
            task_id=task.task_id,
            worker_id="worker-1",
            validation_digest="c" * 64,
            now=5.0,
        )


def test_failed_and_expired_running_tasks_are_reclaimed_with_attempt_budget(tmp_path):
    journal = MigrationJournal(tmp_path / "migration.sqlite3")
    task = journal.seed([candidate()], now=1.0)[0]
    journal.claim(owner_id="alice", worker_id="worker-1", lease_seconds=10, now=2.0)
    failed = journal.mark_failed(
        task_id=task.task_id,
        worker_id="worker-1",
        failure_type="ParserError",
        now=3.0,
    )
    assert failed.state == "failed" and failed.failure_type == "ParserError"
    retried = journal.claim(
        owner_id="alice",
        worker_id="worker-2",
        max_attempts=2,
        now=4.0,
    )
    assert retried is not None and retried.attempt == 2
    journal.mark_failed(
        task_id=task.task_id,
        worker_id="worker-2",
        failure_type="ParserError",
        now=5.0,
    )
    assert journal.claim(
        owner_id="alice",
        worker_id="worker-3",
        max_attempts=2,
        now=6.0,
    ) is None

    other = journal.seed([candidate(doc_id="doc-2")], now=7.0)[0]
    journal.claim(
        owner_id="alice",
        worker_id="worker-1",
        lease_seconds=2,
        max_attempts=2,
        now=8.0,
    )
    reclaimed = journal.claim(
        owner_id="alice",
        worker_id="worker-2",
        max_attempts=3,
        now=11.0,
    )
    assert reclaimed is not None
    assert reclaimed.task_id == other.task_id
    assert reclaimed.attempt == 2
    assert reclaimed.lease_owner == "worker-2"


def test_expired_validated_task_preserves_digest_for_cutover(tmp_path):
    journal = MigrationJournal(tmp_path / "migration.sqlite3")
    task = journal.seed([candidate()], now=1.0)[0]
    journal.claim(
        owner_id="alice",
        worker_id="worker-1",
        lease_seconds=2,
        max_attempts=2,
        now=2.0,
    )
    journal.mark_validated(
        task_id=task.task_id,
        worker_id="worker-1",
        validation_digest="d" * 64,
        now=3.0,
    )
    reclaimed = journal.claim(
        owner_id="alice",
        worker_id="worker-2",
        max_attempts=2,
        now=5.0,
    )
    assert reclaimed is not None
    assert reclaimed.state == "validated"
    assert reclaimed.attempt == 1
    assert reclaimed.validation_digest == "d" * 64
    assert reclaimed.lease_owner == "worker-2"


def test_cancel_and_database_identity_replacement_fail_closed(tmp_path):
    path = tmp_path / "migration.sqlite3"
    journal = MigrationJournal(path)
    task = journal.seed([candidate()], now=1.0)[0]
    cancelled = journal.cancel(task_id=task.task_id, now=2.0)
    assert cancelled.state == "cancelled"
    with pytest.raises(RuntimeError, match="planned or failed"):
        journal.cancel(task_id=task.task_id, now=3.0)

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        journal.list_tasks(owner_id="alice")
