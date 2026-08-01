from dataclasses import replace

import pytest

from tools.migration_cutover_control import CutoverPreparation
from tools.migration_cutover_journal import MigrationCutoverJournal

D = "a" * 64


def preparation(now=1.0, task_id="b" * 64):
    return CutoverPreparation(
        task_id=task_id,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=4,
        source_profile_fingerprint=D,
        target_profile_fingerprint="c" * 64,
        source_content_sha256="d" * 64,
        validation_digest="e" * 64,
        promotion_report_digest="f" * 64,
        benchmark_fingerprint="1" * 64,
        preflight_digest="2" * 64,
        rollback_identity_digest="3" * 64,
        rollback_artifact_digest="4" * 64,
        rollback_key_id="key-1",
        staging_verification_digest="5" * 64,
        target_artifact_digest="6" * 64,
        vector_snapshot_digest="7" * 64,
        sparse_snapshot_digest="8" * 64,
        source_vector_rows=2,
        source_sparse_generation=7,
        source_sparse_fields=2,
        target_vector_rows=3,
        target_sparse_rows=3,
        prepared_at=now,
    )


def test_seed_is_idempotent_across_timestamp_only_repreparation(tmp_path):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    first = journal.seed(preparation(1), now=1)
    second = journal.seed(preparation(2), now=2)
    assert first.operation_id == second.operation_id
    assert second.preparation.prepared_at == 1
    assert second.state == "planned"


def test_claim_ready_and_ready_is_terminal_in_preparation_journal(tmp_path):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    seeded = journal.seed(preparation(), now=1)
    running = journal.claim(
        seeded.operation_id,
        worker_id="worker",
        lease_seconds=10,
        now=2,
    )
    assert running.state == "running" and running.attempt == 1
    ready = journal.mark_ready(
        seeded.operation_id,
        worker_id="worker",
        now=3,
    )
    assert ready.state == "ready" and ready.lease_owner is None
    with pytest.raises(RuntimeError, match="unavailable"):
        journal.claim(seeded.operation_id, worker_id="other", now=4)
    with pytest.raises(RuntimeError, match="planned or failed"):
        journal.cancel(seeded.operation_id, now=4)


def test_expired_running_and_failed_operations_can_retry_with_ceiling(tmp_path):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    seeded = journal.seed(preparation(), now=1)
    first = journal.claim(
        seeded.operation_id,
        worker_id="one",
        lease_seconds=5,
        max_attempts=2,
        now=2,
    )
    assert first.attempt == 1
    second = journal.claim(
        seeded.operation_id,
        worker_id="two",
        lease_seconds=5,
        max_attempts=2,
        now=8,
    )
    assert second.attempt == 2
    failed = journal.mark_failed(
        seeded.operation_id,
        worker_id="two",
        failure_type="RuntimeError",
        now=9,
    )
    assert failed.state == "failed" and failed.failure_type == "RuntimeError"
    with pytest.raises(RuntimeError, match="unavailable"):
        journal.claim(
            seeded.operation_id,
            worker_id="three",
            max_attempts=2,
            now=10,
        )
    cancelled = journal.cancel(seeded.operation_id, now=11)
    assert cancelled.state == "cancelled"


def test_owner_scoped_listing_and_database_identity(tmp_path):
    path = tmp_path / "cutovers.sqlite3"
    journal = MigrationCutoverJournal(path)
    journal.seed(preparation(task_id="b" * 64), now=1)
    other = replace(preparation(task_id="9" * 64), owner_id="bob")
    journal.seed(other, now=2)
    assert len(journal.list_operations(owner_id="alice")) == 1
    assert len(journal.list_operations(owner_id="bob")) == 1
    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        journal.list_operations(owner_id="alice")
