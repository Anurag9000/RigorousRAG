from __future__ import annotations

import json

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_cli_boundary as boundary,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_contracts import (
    RestoreCustodyArtifactAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_journal_boundary import (
    GovernedRestoreCustodyArtifactJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_readonly import (
    ReadOnlyRestoreCustodyArtifactJournal,
)


def attempt():
    return RestoreCustodyArtifactAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        backup_path_digest="3" * 64,
        receipt_path_digest="4" * 64,
        now=1.0,
    )


def test_read_only_artifact_view_requires_schema_and_rejects_writes(tmp_path):
    uninitialized = tmp_path / "uninitialized.sqlite3"
    uninitialized.write_bytes(b"")
    with pytest.raises(RuntimeError, match="not initialized"):
        ReadOnlyRestoreCustodyArtifactJournal(uninitialized)

    writable = GovernedRestoreCustodyArtifactJournal(tmp_path / "artifacts.sqlite3")
    value = writable.seed(attempt())
    read_only = ReadOnlyRestoreCustodyArtifactJournal(writable.path)
    assert read_only.get(value.artifact_id) == value
    assert read_only.list(owner_id="alice", limit=10) == (value,)
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_restore_custody_artifacts"
            )


def test_canonical_status_and_list_use_only_read_only_view(monkeypatch, capsys):
    value = attempt()

    class ReadOnly:
        def get(self, artifact_id):
            assert artifact_id == value.artifact_id
            return value

        def list(self, **kwargs):
            assert kwargs["owner_id"] == "alice"
            return (value,)

    monkeypatch.setattr(boundary, "_read_only_journal", lambda: ReadOnly())
    monkeypatch.setattr(
        boundary._base,
        "get_restore_custody_artifact_journal",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("writable runtime must not be loaded")
        ),
    )

    assert boundary.main(["status", value.artifact_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["journal_mutation_performed"] is False
    assert status["raw_paths_returned"] is False

    assert boundary.main(["list", "--owner-id", "alice"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["journal_mutation_performed"] is False
    assert listing["count"] == 1
