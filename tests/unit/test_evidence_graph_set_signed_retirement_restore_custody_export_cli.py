from __future__ import annotations

import json
from types import SimpleNamespace

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_export_cli as cli,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_export_cli_boundary as boundary,
)


def manifest():
    return SimpleNamespace(
        owner_id="alice",
        restore_id="1" * 64,
        snapshot_digest="2" * 64,
        target_path_digest="3" * 64,
        custody_id="4" * 64,
        chain_digest="5" * 64,
        artifacts=(object(),),
        legal_hold_status="active",
        generated_at=10.0,
    )


def test_offline_verify_does_not_load_live_stores(monkeypatch, capsys):
    value = manifest()
    monkeypatch.setattr(cli, "verify_restore_chain_of_custody", lambda _path: value)
    for name in (
        "ReadOnlySignedRetirementRestoreIntentJournal",
        "ReadOnlySignedRetirementRestoreCustodyStore",
        "ReadOnlyRestoreCustodyArtifactJournal",
        "ReadOnlySignedRetirementRestoreHoldStore",
    ):
        monkeypatch.setattr(
            cli,
            name,
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(
                AssertionError(f"{_name} must not be loaded")
            ),
        )

    assert boundary.main(["verify", "chain.json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["chain_digest"] == "5" * 64
    assert payload["authenticated"] is False
    assert payload["contains_raw_paths"] is False
    assert payload["mutation_performed"] is False
    assert payload["import_performed"] is False


def test_export_summary_is_path_actor_and_secret_free(monkeypatch, capsys):
    value = manifest()
    monkeypatch.setattr(
        cli,
        "ReadOnlySignedRetirementRestoreIntentJournal",
        lambda _path: "restore-store",
    )
    monkeypatch.setattr(
        cli,
        "ReadOnlySignedRetirementRestoreCustodyStore",
        lambda _path: "custody-store",
    )
    monkeypatch.setattr(
        cli,
        "ReadOnlyRestoreCustodyArtifactJournal",
        lambda _path: "artifact-store",
    )
    monkeypatch.setattr(cli, "export_restore_chain_of_custody", lambda **kwargs: value)

    assert boundary.main(
        [
            "export",
            "--restore-id",
            "1" * 64,
            "--snapshot",
            "/private/snapshot.json",
            "--target-db-path",
            "/private/target.sqlite3",
            "--backup-path",
            "/private/backup.sqlite3",
            "--pre-receipt-path",
            "/private/pre.json",
            "--post-receipt-path",
            "/private/post.json",
            "--output",
            "/private/export.json",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    rendered = captured.out.lower()
    assert captured.err == ""
    assert payload["output_created"] is True
    assert payload["artifact_pair_count"] == 1
    assert payload["contains_assertion_secrets"] is False
    assert payload["contains_raw_paths"] is False
    assert "/private" not in rendered
    assert "actor" not in rendered
    assert "key_material" not in rendered


def test_authenticated_verify_reports_key_id_but_no_key_material(monkeypatch, capsys):
    envelope = SimpleNamespace(
        manifest=manifest(),
        algorithm="hmac-sha256",
        key_id="key-1",
    )
    monkeypatch.setattr(
        cli,
        "verify_authenticated_restore_chain_of_custody",
        lambda **kwargs: envelope,
    )
    assert boundary.main(
        [
            "verify-authenticated",
            "chain.auth.json",
            "--key-path",
            "key.bin",
            "--expected-key-id",
            "key-1",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["authenticated"] is True
    assert payload["algorithm"] == "hmac-sha256"
    assert payload["key_id"] == "key-1"
    assert payload["contains_assertion_secrets"] is False
    assert "key.bin" not in captured.out
