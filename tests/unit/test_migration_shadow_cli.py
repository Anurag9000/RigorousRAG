from __future__ import annotations

import json
from types import SimpleNamespace

from tools import migration_shadow_cli
from tools.migration_shadow_executor import ShadowExecutionResult

TASK_ID = "e" * 64


def output(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_execute_one_outputs_public_result_and_failure_exit(monkeypatch, capsys):
    monkeypatch.setattr(
        migration_shadow_cli,
        "execute_next_shadow_build",
        lambda **kwargs: ShadowExecutionResult(
            task_id=TASK_ID,
            outcome="validated",
            task_state="validated",
            validation_digest="a" * 64,
            vector_count=3,
            sparse_count=3,
        ),
    )
    assert migration_shadow_cli.main(
        [
            "execute-one",
            "--owner-id",
            "alice",
            "--worker-id",
            "builder",
        ]
    ) == 0
    result, error = output(capsys)
    assert error is None
    assert result["outcome"] == "validated"
    assert result["vector_count"] == 3
    assert "source_path" not in result

    monkeypatch.setattr(
        migration_shadow_cli,
        "execute_next_shadow_build",
        lambda **kwargs: ShadowExecutionResult(
            task_id=TASK_ID,
            outcome="failed",
            task_state="failed",
            failure_type="RuntimeError",
        ),
    )
    assert migration_shadow_cli.main(
        [
            "execute-one",
            "--owner-id",
            "alice",
            "--worker-id",
            "builder",
        ]
    ) == 1
    result, _error = output(capsys)
    assert result["failure_type"] == "RuntimeError"


def test_execute_one_reports_no_buildable_task(monkeypatch, capsys):
    monkeypatch.setattr(
        migration_shadow_cli,
        "execute_next_shadow_build",
        lambda **kwargs: None,
    )
    assert migration_shadow_cli.main(
        [
            "execute-one",
            "--owner-id",
            "alice",
            "--worker-id",
            "builder",
        ]
    ) == 0
    result, error = output(capsys)
    assert error is None
    assert result == {"outcome": "no_buildable_task", "owner_id": "alice"}


def test_validate_compares_manifest_and_journal_without_paths(monkeypatch, capsys):
    task = SimpleNamespace(
        task_id=TASK_ID,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=2,
        source_profile_fingerprint="a" * 64,
        target_profile_name="bge-m3",
        target_profile_fingerprint="b" * 64,
        validation_digest="c" * 64,
        state="validated",
    )
    manifest = SimpleNamespace(
        task_id=TASK_ID,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=2,
        source_profile_fingerprint="a" * 64,
        target_profile_name="bge-m3",
        target_profile_fingerprint="b" * 64,
        validation_digest="c" * 64,
        vector_count=4,
        sparse_count=4,
        content_sha256="d" * 64,
        parser_fingerprint="f" * 64,
    )
    monkeypatch.setattr(
        migration_shadow_cli,
        "get_migration_journal",
        lambda: SimpleNamespace(get=lambda task_id: task),
    )
    monkeypatch.setattr(
        migration_shadow_cli,
        "get_migration_shadow_store",
        lambda: SimpleNamespace(validate=lambda task_id: manifest),
    )
    assert migration_shadow_cli.main(["validate", TASK_ID]) == 0
    result, error = output(capsys)
    assert error is None
    assert result["validation_digest"] == "c" * 64
    assert result["vector_count"] == 4
    assert "path" not in json.dumps(result).lower()


def test_remove_requires_exact_confirmation_and_terminal_state(monkeypatch, capsys):
    failed = SimpleNamespace(state="failed")
    removed = []
    monkeypatch.setattr(
        migration_shadow_cli,
        "get_migration_journal",
        lambda: SimpleNamespace(get=lambda task_id: failed),
    )
    monkeypatch.setattr(
        migration_shadow_cli,
        "get_migration_shadow_store",
        lambda: SimpleNamespace(remove=lambda task_id: removed.append(task_id) or True),
    )
    assert migration_shadow_cli.main(
        ["remove", TASK_ID, "--confirm-task-id", "f" * 64]
    ) == 2
    _result, error = output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    assert removed == []

    assert migration_shadow_cli.main(
        ["remove", TASK_ID, "--confirm-task-id", TASK_ID]
    ) == 0
    result, error = output(capsys)
    assert error is None
    assert result == {"removed": True, "task_id": TASK_ID}
    assert removed == [TASK_ID]


def test_validate_mismatch_and_remove_validated_are_generic(monkeypatch, capsys):
    task = SimpleNamespace(
        task_id=TASK_ID,
        owner_id="alice",
        doc_id="doc-1",
        source_sequence=2,
        source_profile_fingerprint="a" * 64,
        target_profile_name="bge-m3",
        target_profile_fingerprint="b" * 64,
        validation_digest="c" * 64,
        state="validated",
    )
    manifest = SimpleNamespace(
        task_id=TASK_ID,
        owner_id="alice",
        doc_id="different",
        source_sequence=2,
        source_profile_fingerprint="a" * 64,
        target_profile_name="bge-m3",
        target_profile_fingerprint="b" * 64,
        validation_digest="c" * 64,
    )
    monkeypatch.setattr(
        migration_shadow_cli,
        "get_migration_journal",
        lambda: SimpleNamespace(get=lambda task_id: task),
    )
    monkeypatch.setattr(
        migration_shadow_cli,
        "get_migration_shadow_store",
        lambda: SimpleNamespace(
            validate=lambda task_id: manifest,
            remove=lambda task_id: True,
        ),
    )
    assert migration_shadow_cli.main(["validate", TASK_ID]) == 2
    _result, error = output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    assert migration_shadow_cli.main(
        ["remove", TASK_ID, "--confirm-task-id", TASK_ID]
    ) == 2
    _result, error = output(capsys)
    assert error == {"error": "invalid_or_unavailable"}
