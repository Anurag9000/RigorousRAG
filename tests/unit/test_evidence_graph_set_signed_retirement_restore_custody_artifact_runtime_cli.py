from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_cli as cli,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_runtime as runtime,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_contracts import (
    RestoreCustodyArtifactAttempt,
)


def attempt(*, state="planned", phase="planned"):
    value = RestoreCustodyArtifactAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        backup_path_digest="3" * 64,
        receipt_path_digest="4" * 64,
        now=1.0,
    )
    if state == "planned":
        return value
    if state == "failed":
        return SimpleNamespace(
            **{
                **value.__dict__,
                "state": "failed",
                "phase": phase,
                "failure_type": "Failure",
                "attempt_count": 1,
                "updated_at": 2.0,
            }
        )
    return value


class Journal:
    def __init__(self, value=None):
        self.value = value or attempt()

    def get(self, artifact_id):
        assert artifact_id == self.value.artifact_id
        return self.value

    def list(self, **kwargs):
        self.list_kwargs = kwargs
        return (self.value,)

    def retry(self, artifact_id, **kwargs):
        self.retry_kwargs = kwargs
        return self.value

    def cancel(self, artifact_id, **kwargs):
        self.cancel_kwargs = kwargs
        return self.value


def test_runtime_rejects_canonical_and_hard_link_aliases(tmp_path, monkeypatch):
    runtime.clear_restore_custody_artifact_journal_cache()
    protected = tmp_path / "protected.sqlite3"
    protected.write_bytes(b"x")
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
        str(protected),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_restore_custody_artifact_journal(
            protected_paths=(protected,)
        )

    journal_path = tmp_path / "journal.sqlite3"
    runtime.clear_restore_custody_artifact_journal_cache()
    journal = runtime.get_restore_custody_artifact_journal(path=journal_path)
    assert journal.path == journal_path
    if hasattr(os, "link"):
        alias = tmp_path / "journal-alias.sqlite3"
        try:
            os.link(journal_path, alias)
        except OSError:
            return
        runtime.clear_restore_custody_artifact_journal_cache()
        with pytest.raises(RuntimeError, match="must not alias"):
            runtime.get_restore_custody_artifact_journal(
                path=alias,
                protected_paths=(journal_path,),
            )


def test_seed_confirmation_fails_before_journal_resolution(monkeypatch, capsys):
    snapshot = SimpleNamespace(snapshot_digest="1" * 64)
    monkeypatch.setattr(
        cli,
        "verify_signed_retirement_snapshot",
        lambda _path: snapshot,
    )
    monkeypatch.setattr(
        cli,
        "get_restore_custody_artifact_journal",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("journal must not be opened")
        ),
    )

    assert cli.main(
        [
            "seed",
            "--snapshot",
            "snapshot.json",
            "--target-db-path",
            "target.sqlite3",
            "--backup-output",
            "backup.sqlite3",
            "--receipt-output",
            "receipt.json",
            "--confirm-snapshot-digest",
            "f" * 64,
        ]
    ) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}


def test_execute_confirmation_fails_before_actor_and_journal(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: (_ for _ in ()).throw(
            AssertionError("actor must not be loaded")
        ),
    )
    monkeypatch.setattr(
        cli,
        "get_restore_custody_artifact_journal",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("journal must not be opened")
        ),
    )
    assert cli.main(
        [
            "execute",
            "a" * 64,
            "--confirm-artifact-id",
            "b" * 64,
            "--snapshot",
            "snapshot.json",
            "--target-db-path",
            "target.sqlite3",
            "--backup-output",
            "backup.sqlite3",
            "--receipt-output",
            "receipt.json",
            "--worker-id",
            "worker",
        ]
    ) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"error": "invalid_or_unavailable"}


def test_status_list_retry_cancel_are_path_free_and_exact(monkeypatch, capsys):
    journal = Journal()
    monkeypatch.setattr(
        cli,
        "get_restore_custody_artifact_journal",
        lambda **kwargs: journal,
    )

    assert cli.main(["status", journal.value.artifact_id]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["raw_paths_returned"] is False
    assert "receipt_actor_id" not in payload
    assert payload["journal_mutation_performed"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["raw_paths_returned"] is False
    assert "receipt_actor_id" not in payload["items"][0]

    assert cli.main(
        [
            "retry",
            journal.value.artifact_id,
            "--owner-id",
            "alice",
            "--confirm-artifact-id",
            journal.value.artifact_id,
        ]
    ) == 0
    capsys.readouterr()
    assert journal.retry_kwargs["confirm_artifact_id"] == journal.value.artifact_id

    assert cli.main(
        [
            "cancel",
            journal.value.artifact_id,
            "--owner-id",
            "alice",
            "--confirm-artifact-id",
            journal.value.artifact_id,
        ]
    ) == 0
    capsys.readouterr()
    assert journal.cancel_kwargs["confirm_artifact_id"] == journal.value.artifact_id


def test_recovery_error_is_generic_and_reports_no_destructive_action(
    monkeypatch,
    capsys,
):
    from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_boundary import (
        RestoreCustodyArtifactRecoveryError,
    )

    monkeypatch.setattr(cli, "load_relation_review_actor", lambda: object())
    monkeypatch.setattr(
        cli,
        "require_relation_review_actor",
        lambda requested, binding: binding,
    )
    monkeypatch.setattr(
        cli,
        "get_restore_custody_artifact_journal",
        lambda **kwargs: object(),
    )

    def fail(*args, **kwargs):
        raise RestoreCustodyArtifactRecoveryError(
            "private paths and evidence",
            artifact_id="a" * 64,
            state="failed",
            phase="publication_intent",
        )

    monkeypatch.setattr(cli, "execute_restore_custody_artifact_attempt", fail)
    assert cli.main(
        [
            "execute",
            "a" * 64,
            "--confirm-artifact-id",
            "a" * 64,
            "--snapshot",
            "snapshot.json",
            "--target-db-path",
            "target.sqlite3",
            "--backup-output",
            "backup.sqlite3",
            "--receipt-output",
            "receipt.json",
            "--worker-id",
            "worker",
        ]
    ) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["error"] == "artifact_publication_failed"
    assert payload["phase"] == "publication_intent"
    assert payload["artifact_deletion_performed"] is False
    assert payload["artifact_overwrite_performed"] is False
    assert "private" not in captured.err.lower()
