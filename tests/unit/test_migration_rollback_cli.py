import base64
import json
from types import SimpleNamespace

from tools import migration_rollback_cli as cli
from tools.migration_rollback_store import MigrationRollbackStore
from tests.unit.test_migration_rollback_artifact import aligned_preflight


def install(monkeypatch, tmp_path, state="validated"):
    preflight, snapshot = aligned_preflight()
    task = SimpleNamespace(
        task_id=preflight.task_id,
        owner_id=preflight.owner_id,
        doc_id=preflight.doc_id,
        state=state,
    )
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    monkeypatch.setattr(
        cli,
        "get_migration_journal",
        lambda: SimpleNamespace(get=lambda task_id: task if task_id == task.task_id else None),
    )
    monkeypatch.setattr(
        cli,
        "get_migration_cutover_preflight_store",
        lambda: SimpleNamespace(
            read=lambda task_id, preflight_digest=None: preflight
        ),
    )
    monkeypatch.setattr(cli, "get_migration_rollback_store", lambda: store)
    monkeypatch.setattr(cli, "_capture_snapshot", lambda current: snapshot)
    monkeypatch.setenv(
        "MIGRATION_ROLLBACK_KEY_B64",
        base64.b64encode(b"k" * 32).decode("ascii"),
    )
    monkeypatch.setenv("MIGRATION_ROLLBACK_KEY_ID", "key-1")
    return task, preflight, store


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_capture_status_verify_never_return_plaintext_or_restore(
    tmp_path, monkeypatch, capsys
):
    _task, preflight, store = install(monkeypatch, tmp_path)
    assert cli.main(["capture", preflight.task_id]) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["verified"] is True
    assert output["restore_performed"] is False
    assert output["mutation_performed"] is False
    assert "one" not in json.dumps(output)
    assert "source_path" not in json.dumps(output)
    assert store.read_manifest(preflight.task_id, preflight.preflight_digest).key_id == "key-1"

    assert cli.main(["status", preflight.task_id]) == 0
    status, error = parse(capsys)
    assert error is None and status["verified"] is False
    assert cli.main(["verify", preflight.task_id]) == 0
    verified, error = parse(capsys)
    assert error is None and verified["verified"] is True
    assert verified["restore_performed"] is False


def test_missing_key_capture_and_verify_fail_generic_but_status_does_not_need_key(
    tmp_path, monkeypatch, capsys
):
    _task, preflight, _store = install(monkeypatch, tmp_path)
    assert cli.main(["capture", preflight.task_id]) == 0
    parse(capsys)
    monkeypatch.delenv("MIGRATION_ROLLBACK_KEY_B64")
    monkeypatch.delenv("MIGRATION_ROLLBACK_KEY_ID")
    assert cli.main(["status", preflight.task_id]) == 0
    parse(capsys)
    assert cli.main(["verify", preflight.task_id]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}


def test_remove_requires_failed_state_and_double_exact_confirmation(
    tmp_path, monkeypatch, capsys
):
    task, preflight, _store = install(monkeypatch, tmp_path)
    assert cli.main(["capture", preflight.task_id]) == 0
    parse(capsys)
    args = [
        "remove",
        preflight.task_id,
        "--preflight-digest",
        preflight.preflight_digest,
        "--confirm-task-id",
        preflight.task_id,
        "--confirm-preflight-digest",
        preflight.preflight_digest,
    ]
    assert cli.main(args) == 2
    parse(capsys)
    task.state = "failed"
    wrong = list(args)
    wrong[-1] = "9" * 64
    assert cli.main(wrong) == 2
    parse(capsys)
    assert cli.main(args) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["removed"] is True
    assert output["restore_performed"] is False
    assert output["mutation_performed"] is True


def test_not_found_is_bounded(tmp_path, monkeypatch, capsys):
    _task, preflight, _store = install(monkeypatch, tmp_path)
    assert cli.main(["status", preflight.task_id]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": preflight.task_id}
    assert cli.main(["capture", "9" * 64]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": "9" * 64}


def test_missing_preflight_is_bounded_not_found(tmp_path, monkeypatch, capsys):
    _task, preflight, _store = install(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli,
        "get_migration_cutover_preflight_store",
        lambda: SimpleNamespace(
            read=lambda task_id, preflight_digest=None: (_ for _ in ()).throw(
                FileNotFoundError(task_id)
            )
        ),
    )
    for command in ("capture", "status", "verify"):
        assert cli.main([command, preflight.task_id]) == 1
        _output, error = parse(capsys)
        assert error == {"error": "not_found", "task_id": preflight.task_id}
