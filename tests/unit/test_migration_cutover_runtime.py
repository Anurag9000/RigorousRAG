from dataclasses import replace

import pytest

from tools.migration_cutover_journal import MigrationCutoverJournal
from tools.migration_cutover_runtime import prepare_cutover_operation
from tests.unit.test_migration_cutover_journal import preparation


def test_prepare_resolves_twice_under_lease_and_marks_ready(tmp_path):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    calls = []

    def resolver(task_id):
        calls.append(task_id)
        return preparation(now=float(len(calls)), task_id=task_id)

    result = prepare_cutover_operation(
        "b" * 64,
        worker_id="worker",
        journal=journal,
        resolver=resolver,
    )
    assert result.state == "ready"
    assert calls == ["b" * 64, "b" * 64]
    again = prepare_cutover_operation(
        "b" * 64,
        worker_id="worker",
        journal=journal,
        resolver=resolver,
    )
    assert again.state == "ready"
    assert len(calls) == 3


def test_changed_prerequisites_during_lease_mark_failed(tmp_path):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    calls = 0

    def resolver(task_id):
        nonlocal calls
        calls += 1
        value = preparation(now=float(calls), task_id=task_id)
        if calls == 2:
            value = replace(value, rollback_artifact_digest="9" * 64)
        return value

    with pytest.raises(RuntimeError, match="changed"):
        prepare_cutover_operation(
            "b" * 64,
            worker_id="worker",
            journal=journal,
            resolver=resolver,
        )
    operations = journal.list_operations(owner_id="alice")
    assert len(operations) == 1
    assert operations[0].state == "failed"
    assert operations[0].failure_type == "RuntimeError"


def test_invalid_resolver_result_fails_before_journal_mutation(tmp_path):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    with pytest.raises(RuntimeError, match="invalid preparation"):
        prepare_cutover_operation(
            "b" * 64,
            worker_id="worker",
            journal=journal,
            resolver=lambda task_id: object(),
        )
    assert journal.list_operations(owner_id="alice") == ()
