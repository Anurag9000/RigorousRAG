from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_cli as cli,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_runtime as runtime,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    CustodyTimestampIssuanceAttempt,
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_journal import (
    CustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_readonly import (
    ReadOnlyCustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile import (
    CustodyTimestampIssuanceExecution,
)


def attempt():
    return CustodyTimestampIssuanceAttempt.create(
        owner_id="alice",
        authority_id="tsa-1",
        key_id="timestamp-key-1",
        serial="1" * 64,
        attestation_digest="2" * 64,
        output_path_digest="3" * 64,
        max_attempts=3,
        now=1.0,
    )


def test_issuance_read_only_requires_initialized_and_refuses_writes(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        ReadOnlyCustodyTimestampIssuanceJournal(tmp_path / "missing.sqlite3")
    journal = CustodyTimestampIssuanceJournal(tmp_path / "issuances.sqlite3")
    read_only = ReadOnlyCustodyTimestampIssuanceJournal(journal.path)
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_restore_custody_timestamp_issuances"
            )


def test_issuance_runtime_rejects_alias_and_caches(tmp_path, monkeypatch):
    runtime.clear_custody_timestamp_issuance_journal_cache()
    protected = tmp_path / "authorities.sqlite3"
    protected.write_bytes(b"not an issuance journal")
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH",
        str(protected),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_custody_timestamp_issuance_journal(protected)

    monkeypatch.delenv("EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH")
    selected = tmp_path / "issuances.sqlite3"
    first = runtime.get_custody_timestamp_issuance_journal(selected)
    second = runtime.get_custody_timestamp_issuance_journal(str(selected))
    assert first is second


def test_bad_seed_confirmation_precedes_store_resolution(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "timestamp.json"
    calls = []
    monkeypatch.setattr(
        cli,
        "ReadOnlyCustodyTimestampAuthorityRegistry",
        lambda path: calls.append("authority"),
    )
    monkeypatch.setattr(
        cli,
        "get_custody_timestamp_issuance_journal",
        lambda path: calls.append("issuance"),
    )
    assert cli.main(
        [
            "seed",
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--authority-private-key-path",
            "private.pem",
            "--signed-envelope-path",
            "envelope.json",
            "--custody-signer-public-key-path",
            "custody.pem",
            "--output-path",
            str(output),
            "--confirm-output-path-digest",
            "f" * 64,
        ]
    ) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().err) == {"error": "invalid_or_unavailable"}


def test_issuance_cli_seed_execute_status_and_list_are_path_free(
    tmp_path,
    monkeypatch,
    capsys,
):
    value = attempt()
    output = tmp_path / "timestamp.json"
    output_digest = timestamp_output_path_digest(output)
    authority = object()
    journal = object()
    monkeypatch.setattr(
        cli,
        "ReadOnlyCustodyTimestampAuthorityRegistry",
        lambda path: authority,
    )
    monkeypatch.setattr(
        cli,
        "get_custody_timestamp_issuance_journal",
        lambda path: journal,
    )
    observed = {}

    def seed(**kwargs):
        observed["seed"] = kwargs
        return value, SimpleNamespace(serial=value.serial)

    monkeypatch.setattr(cli, "seed_custody_timestamp_issuance", seed)
    assert cli.main(
        [
            "seed",
            "--owner-id",
            "alice",
            "--authority-id",
            "tsa-1",
            "--key-id",
            "timestamp-key-1",
            "--authority-private-key-path",
            "/private/authority.pem",
            "--signed-envelope-path",
            "/private/envelope.json",
            "--custody-signer-public-key-path",
            "/private/custody.pem",
            "--output-path",
            str(output),
            "--confirm-output-path-digest",
            output_digest,
        ]
    ) == 0
    seeded = json.loads(capsys.readouterr().out)
    assert seeded["state"] == "planned"
    assert seeded["attestation_output_created"] is False
    assert seeded["private_key_material_stored"] is False
    assert "/private/" not in json.dumps(seeded)
    assert observed["seed"]["journal"] is journal
    assert observed["seed"]["registry"] is authority

    result = CustodyTimestampIssuanceExecution(
        issuance_id=value.issuance_id,
        serial=value.serial,
        state="completed",
        phase="verified",
        attestation_digest=value.attestation_digest,
        output_path_digest=value.output_path_digest,
        verification_digest="4" * 64,
        attempt_count=1,
        output_created=True,
        existing_exact_output_reused=False,
    )
    monkeypatch.setattr(
        cli,
        "execute_custody_timestamp_issuance",
        lambda *args, **kwargs: result,
    )
    assert cli.main(
        [
            "execute",
            value.issuance_id,
            "--output-path",
            str(output),
            "--worker-id",
            "worker",
        ]
    ) == 0
    executed = json.loads(capsys.readouterr().out)
    assert executed["state"] == "completed"
    assert executed["contains_raw_paths"] is False
    assert str(output) not in json.dumps(executed)

    class ReadOnly:
        def __init__(self, path):
            self.path = path

        def get(self, issuance_id):
            assert issuance_id == value.issuance_id
            return value

        def list(self, **kwargs):
            assert kwargs["owner_id"] == "alice"
            return (value,)

    monkeypatch.setattr(cli, "ReadOnlyCustodyTimestampIssuanceJournal", ReadOnly)
    assert cli.main(["status", value.issuance_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["issuance_journal_mutation_performed"] is False
    assert status["contains_attestation_signature"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["count"] == 1
    assert listing["issuance_journal_mutation_performed"] is False
    assert listing["contains_attestation_signatures"] is False
