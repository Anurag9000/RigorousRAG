from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_execute_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_reconcile import (
    SignedRetirementRestoreDeletionExecution,
    SignedRetirementRestoreDeletionRecoveryError,
)


def attempt():
    return SimpleNamespace(
        deletion_id="1" * 64,
        authorization_id="2" * 64,
        authorization_digest="3" * 64,
        owner_id="alice",
        restore_id="4" * 64,
        snapshot_digest="5" * 64,
        target_path_digest="6" * 64,
        restore_state="cancelled",
        restore_phase="planned",
        restore_record_digest="7" * 64,
        custody_id=None,
        custody_manifest_digest=None,
        state="planned",
        phase="planned",
        attempt_count=0,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        marker_digest=None,
        tombstone_digest=None,
        failure_type=None,
        created_at=1.0,
        updated_at=1.0,
        completed_at=None,
    )


class Journal:
    def __init__(self):
        self.value = attempt()

    def get(self, deletion_id):
        return self.value

    def list(self, **kwargs):
        self.list_kwargs = kwargs
        return (self.value,)

    def retry(self, deletion_id, **kwargs):
        self.retry_kwargs = kwargs
        return self.value

    def cancel(self, deletion_id, **kwargs):
        self.cancel_kwargs = kwargs
        return self.value


def read(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def test_bad_confirmation_precedes_any_store_creation(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_deletion_journal",
        lambda: (_ for _ in ()).throw(
            AssertionError("store opened before confirmation")
        ),
    )
    assert cli.main(
        [
            "seed",
            "1" * 64,
            "--restore-id",
            "2" * 64,
            "--confirm-authorization-id",
            "3" * 64,
            "--confirm-restore-id",
            "2" * 64,
        ]
    ) == 2
    output, error = read(capsys)
    assert output is None
    assert error == {"error": "invalid_or_unavailable"}


def test_status_list_retry_and_cancel_use_only_deletion_journal(
    monkeypatch, capsys
):
    journal = Journal()
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_deletion_journal",
        lambda: journal,
    )
    monkeypatch.setattr(
        cli,
        "_dependencies",
        lambda: (_ for _ in ()).throw(
            AssertionError("full dependencies loaded")
        ),
    )
    assert cli.main(["status", "1" * 64]) == 0
    assert read(capsys)[0]["mutation_performed"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    assert read(capsys)[0]["raw_paths_returned"] is False

    assert cli.main(
        [
            "retry",
            "1" * 64,
            "--owner-id",
            "alice",
            "--confirm-deletion-id",
            "1" * 64,
        ]
    ) == 0
    read(capsys)
    assert journal.retry_kwargs["confirm_deletion_id"] == "1" * 64

    assert cli.main(
        [
            "cancel",
            "1" * 64,
            "--owner-id",
            "alice",
            "--confirm-deletion-id",
            "1" * 64,
        ]
    ) == 0
    read(capsys)
    assert journal.cancel_kwargs["confirm_deletion_id"] == "1" * 64


def test_execute_reports_preserved_custody_and_holds(monkeypatch, capsys):
    result = SignedRetirementRestoreDeletionExecution(
        deletion_id="1" * 64,
        authorization_id="2" * 64,
        restore_id="3" * 64,
        state="completed",
        phase="verified",
        marker_digest="4" * 64,
        tombstone_digest="5" * 64,
        attempt_count=1,
        restore_row_deleted=True,
        authorization_consumed=True,
    )
    monkeypatch.setattr(cli, "_dependencies", lambda: {})
    monkeypatch.setattr(
        cli,
        "execute_signed_retirement_restore_deletion",
        lambda *args, **kwargs: result,
    )
    assert cli.main(
        [
            "execute",
            "1" * 64,
            "--worker-id",
            "worker",
        ]
    ) == 0
    output, error = read(capsys)
    assert error is None
    assert output["restore_row_deleted"] is True
    assert output["custody_preserved"] is True
    assert output["custody_deleted"] is False
    assert output["holds_deleted"] is False


def test_recovery_error_is_generic_and_path_free(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_dependencies", lambda: {})

    def fail(*args, **kwargs):
        raise SignedRetirementRestoreDeletionRecoveryError(
            "private path and failure details",
            deletion_id="1" * 64,
            state="failed",
            phase="marker_active",
        )

    monkeypatch.setattr(
        cli,
        "execute_signed_retirement_restore_deletion",
        fail,
    )
    assert cli.main(
        [
            "execute",
            "1" * 64,
            "--worker-id",
            "worker",
        ]
    ) == 1
    output, error = read(capsys)
    assert output is None
    assert error["error"] == "deletion_failed"
    assert error["phase"] == "marker_active"
    assert error["custody_deleted"] is False
    assert "private" not in json.dumps(error).lower()
