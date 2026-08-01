import base64
import json
from types import SimpleNamespace

from tools import migration_rollback_staging_cli as cli
from tools.migration_rollback_artifact import (
    RollbackEncryptionKey,
    capture_rollback_payload,
)
from tools.migration_rollback_store import MigrationRollbackStore
from tests.unit.test_migration_rollback_artifact import aligned_preflight


def install(monkeypatch, tmp_path):
    preflight, snapshot = aligned_preflight()
    store = MigrationRollbackStore(tmp_path / "rollbacks")
    store.write(
        preflight=preflight,
        payload=capture_rollback_payload(preflight, snapshot),
        key=RollbackEncryptionKey("key-1", b"k" * 32),
        now=1,
    )
    monkeypatch.setattr(
        cli,
        "get_migration_cutover_preflight_store",
        lambda: SimpleNamespace(
            read=lambda task_id, preflight_digest=None: preflight
        ),
    )
    monkeypatch.setattr(cli, "get_migration_rollback_store", lambda: store)
    monkeypatch.setenv(
        "MIGRATION_ROLLBACK_KEY_B64",
        base64.b64encode(b"k" * 32).decode("ascii"),
    )
    monkeypatch.setenv("MIGRATION_ROLLBACK_KEY_ID", "key-1")
    return preflight


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_staging_verify_is_non_authoritative_and_path_free(
    tmp_path, monkeypatch, capsys
):
    preflight = install(monkeypatch, tmp_path)
    assert cli.main([preflight.task_id]) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["staging_verified"] is True
    assert output["staging_scope"] == "process_local_non_authoritative"
    assert output["staging_mutation_performed"] is True
    assert output["authoritative_mutation_performed"] is False
    assert output["restore_performed"] is False
    assert output["cutover_performed"] is False
    assert "one" not in json.dumps(output)
    assert "source_path" not in json.dumps(output)
    assert len(output["verification_digest"]) == 64


def test_missing_key_and_missing_preflight_are_bounded(
    tmp_path, monkeypatch, capsys
):
    preflight = install(monkeypatch, tmp_path)
    monkeypatch.delenv("MIGRATION_ROLLBACK_KEY_B64")
    assert cli.main([preflight.task_id]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}

    monkeypatch.setattr(
        cli,
        "get_migration_cutover_preflight_store",
        lambda: SimpleNamespace(
            read=lambda task_id, preflight_digest=None: (_ for _ in ()).throw(
                FileNotFoundError(task_id)
            )
        ),
    )
    assert cli.main([preflight.task_id]) == 1
    _output, error = parse(capsys)
    assert error == {"error": "not_found", "task_id": preflight.task_id}
