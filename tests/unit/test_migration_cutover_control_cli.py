import json

from tools import migration_cutover_control_cli as cli
from tools.migration_cutover_journal import MigrationCutoverJournal
from tests.unit.test_migration_cutover_journal import preparation


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_prepare_output_is_path_free_and_nonmutating(tmp_path, monkeypatch, capsys):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    seeded = journal.seed(preparation(), now=1)
    running = journal.claim(seeded.operation_id, worker_id="worker", now=2)
    ready = journal.mark_ready(
        seeded.operation_id,
        worker_id="worker",
        fencing_token=running.fencing_token,
        now=3,
    )
    monkeypatch.setattr(
        cli,
        "prepare_cutover_operation",
        lambda *args, **kwargs: ready,
    )
    assert (
        cli.main(
            [
                "prepare",
                ready.preparation.task_id,
                "--worker-id",
                "worker",
            ]
        )
        == 0
    )
    output, error = parse(capsys)
    assert error is None
    assert output["state"] == "ready"
    assert output["fencing_token"] == running.fencing_token
    assert output["authoritative_mutation_performed"] is False
    assert output["restore_performed"] is False
    assert output["cutover_performed"] is False
    assert "source_path" not in json.dumps(output)


def test_status_list_and_cancel_boundaries(tmp_path, monkeypatch, capsys):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    seeded = journal.seed(preparation(), now=1)
    monkeypatch.setattr(cli, "get_migration_cutover_journal", lambda: journal)
    assert cli.main(["status", seeded.operation_id]) == 0
    status, error = parse(capsys)
    assert error is None and status["state"] == "planned"
    assert status["fencing_token"] == 0
    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing, error = parse(capsys)
    assert error is None and listing["count"] == 1
    assert (
        cli.main(
            [
                "cancel",
                seeded.operation_id,
                "--confirm-operation-id",
                "9" * 64,
            ]
        )
        == 2
    )
    parse(capsys)
    assert (
        cli.main(
            [
                "cancel",
                seeded.operation_id,
                "--confirm-operation-id",
                seeded.operation_id,
            ]
        )
        == 0
    )
    cancelled, error = parse(capsys)
    assert error is None and cancelled["state"] == "cancelled"


def test_not_found_and_file_not_found_are_bounded(tmp_path, monkeypatch, capsys):
    journal = MigrationCutoverJournal(tmp_path / "cutovers.sqlite3")
    monkeypatch.setattr(cli, "get_migration_cutover_journal", lambda: journal)
    assert cli.main(["status", "9" * 64]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "operation_id": "9" * 64}
    monkeypatch.setattr(
        cli,
        "prepare_cutover_operation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("8" * 64)
        ),
    )
    assert (
        cli.main(
            [
                "prepare",
                "8" * 64,
                "--worker-id",
                "worker",
            ]
        )
        == 1
    )
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": "8" * 64}