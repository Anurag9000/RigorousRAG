from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_execute_cli as cli
from tools import evidence_graph_set_signed_retirement_restore_runtime as runtime
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    export_signed_retirement_snapshot,
)


class Values:
    def __init__(self, values=()):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def cancelled():
    from dataclasses import replace

    value = SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id="1" * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        now=1.0,
    )
    return replace(
        value,
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


def snapshot_and_target(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Values((cancelled(),)),
        output_path=snapshot_path,
        now=10.0,
        limit=100,
    )
    target = SignedPublicationRetirementJournal(tmp_path / "target.sqlite3")
    return snapshot_path, snapshot, target


def test_runtime_rejects_target_path_and_hard_link_aliases(tmp_path, monkeypatch):
    runtime.clear_signed_retirement_restore_journal_cache()
    _snapshot_path, _snapshot, target = snapshot_and_target(tmp_path)
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        str(target.path),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_signed_retirement_restore_journal(
            target_db_path=target.path
        )

    if not hasattr(os, "link"):
        return
    alias = tmp_path / "restore-alias.sqlite3"
    try:
        os.link(target.path, alias)
    except OSError:
        return
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_signed_retirement_restore_journal(
            path=alias,
            target_db_path=target.path,
        )


def test_runtime_uses_distinct_default_and_caches_by_canonical_path(
    tmp_path,
    monkeypatch,
):
    runtime.clear_signed_retirement_restore_journal_cache()
    _snapshot_path, _snapshot, target = snapshot_and_target(tmp_path)
    restore_path = tmp_path / "restore-intents.sqlite3"
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        str(restore_path),
    )
    first = runtime.get_signed_retirement_restore_journal(
        target_db_path=target.path
    )
    second = runtime.get_signed_retirement_restore_journal(
        path=restore_path,
        target_db_path=target.path,
    )
    assert first is second
    assert first.path == restore_path
    assert first.path != target.path


def test_bad_seed_confirmation_does_not_create_restore_journal(
    tmp_path,
    monkeypatch,
    capsys,
):
    runtime.clear_signed_retirement_restore_journal_cache()
    snapshot_path, _snapshot, target = snapshot_and_target(tmp_path)
    restore_path = tmp_path / "restore-intents.sqlite3"
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        str(restore_path),
    )

    assert cli.main(
        [
            "seed",
            "--snapshot",
            str(snapshot_path),
            "--target-db-path",
            str(target.path),
            "--confirm-snapshot-digest",
            "f" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}
    assert not restore_path.exists()


def test_status_and_list_load_only_restore_journal(monkeypatch, capsys):
    value = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=1,
        now=1.0,
    )

    class Journal:
        def get(self, restore_id):
            assert restore_id == value.restore_id
            return value

        def list(self, **kwargs):
            assert kwargs["owner_id"] == "alice"
            return (value,)

    calls = []

    def get_journal(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs == {}
        return Journal()

    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        get_journal,
    )
    assert cli.main(["status", value.restore_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["mutation_performed"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["mutation_performed"] is False
    assert listing["contains_source_text"] is False
    assert len(calls) == 2


def test_recovery_error_output_is_generic_and_secret_free(monkeypatch, capsys):
    from tools.evidence_graph_set_signed_retirement_restore_reconcile import (
        SignedRetirementRestoreRecoveryError,
    )

    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda **kwargs: object(),
    )

    def fail(*args, **kwargs):
        raise SignedRetirementRestoreRecoveryError(
            "private target path and failure details",
            restore_id="a" * 64,
            state="failed",
            phase="target_committed",
        )

    monkeypatch.setattr(cli, "execute_signed_retirement_restore", fail)
    assert cli.main(
        [
            "execute",
            "a" * 64,
            "--snapshot",
            "snapshot.json",
            "--target-db-path",
            "target.sqlite3",
            "--worker-id",
            "worker",
        ]
    ) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["error"] == "restore_failed"
    assert payload["phase"] == "target_committed"
    assert payload["overwrite_performed"] is False
    assert payload["merge_performed"] is False
    assert "private" not in captured.err.lower()
