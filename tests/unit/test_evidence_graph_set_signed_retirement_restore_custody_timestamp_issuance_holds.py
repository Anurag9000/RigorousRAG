from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_cli as cli,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_runtime as runtime,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_operations_cli as operations_cli,
)
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds import (
    CustodyTimestampIssuanceHoldStore,
    deterministic_timestamp_issuance_hold_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_readonly import (
    ReadOnlyCustodyTimestampIssuanceHoldStore,
)


class Issuances:
    def __init__(self, owner="alice"):
        self.owner = owner

    def get(self, issuance_id):
        return SimpleNamespace(issuance_id=issuance_id, owner_id=self.owner)


def actor(name="officer", *, loaded=1.0, expires=None):
    return ReviewActorBinding.create(
        actor_id=name,
        binding_method="process_environment",
        loaded_at=loaded,
        expires_at=expires,
    )


def test_issuance_hold_identity_place_replay_release_and_scope(tmp_path):
    store = CustodyTimestampIssuanceHoldStore(tmp_path / "holds.sqlite3")
    issuance_id = "1" * 64
    placed = store.place(
        owner_id="alice",
        issuance_id=issuance_id,
        hold_key="litigation-1",
        reason_code="legal_matter",
        actor=actor(),
        issuance_journal=Issuances(),
        now=10.0,
    )
    assert placed.hold_id == deterministic_timestamp_issuance_hold_id(
        owner_id="alice",
        issuance_id=issuance_id,
        hold_key="litigation-1",
    )
    replay = store.place(
        owner_id="alice",
        issuance_id=issuance_id,
        hold_key="litigation-1",
        reason_code="legal_matter",
        actor=actor(),
        issuance_journal=Issuances(),
        now=99.0,
    )
    assert replay == placed
    with pytest.raises(RuntimeError, match="owner scope"):
        store.place(
            owner_id="alice",
            issuance_id="2" * 64,
            hold_key="wrong",
            reason_code="legal_matter",
            actor=actor(),
            issuance_journal=Issuances(owner="bob"),
            now=10.0,
        )
    with pytest.raises(ValueError, match="confirmation"):
        store.release(
            placed.hold_id,
            owner_id="alice",
            confirm_hold_id="f" * 64,
            actor=actor("release"),
            now=20.0,
        )
    released = store.release(
        placed.hold_id,
        owner_id="alice",
        confirm_hold_id=placed.hold_id,
        actor=actor("release"),
        now=20.0,
    )
    assert released.status == "released"
    assert released.created_at == 10.0
    assert released.released_at == 20.0
    assert store.release(
        placed.hold_id,
        owner_id="alice",
        confirm_hold_id=placed.hold_id,
        actor=actor("other"),
        now=30.0,
    ) == released


def test_issuance_hold_expiry_integrity_and_identity_fail_closed(tmp_path):
    path = tmp_path / "holds.sqlite3"
    store = CustodyTimestampIssuanceHoldStore(path)
    with pytest.raises(PermissionError, match="expired"):
        store.place(
            owner_id="alice",
            issuance_id="1" * 64,
            hold_key="expired",
            reason_code="legal_matter",
            actor=actor(expires=5.0),
            issuance_journal=Issuances(),
            now=10.0,
        )
    placed = store.place(
        owner_id="alice",
        issuance_id="1" * 64,
        hold_key="hold",
        reason_code="legal_matter",
        actor=actor(),
        issuance_journal=Issuances(),
        now=10.0,
    )
    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_restore_custody_timestamp_issuance_holds "
            "SET reason_code='other' WHERE hold_id=?",
            (placed.hold_id,),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(placed.hold_id)

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        store.list(owner_id="alice")


def test_issuance_hold_readonly_runtime_and_cli_boundaries(
    tmp_path,
    monkeypatch,
    capsys,
):
    path = tmp_path / "holds.sqlite3"
    store = CustodyTimestampIssuanceHoldStore(path)
    placed = store.place(
        owner_id="alice",
        issuance_id="1" * 64,
        hold_key="hold",
        reason_code="legal_matter",
        actor=actor(),
        issuance_journal=Issuances(),
        now=10.0,
    )
    read_only = ReadOnlyCustodyTimestampIssuanceHoldStore(path)
    assert read_only.get(placed.hold_id) == placed
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_restore_custody_timestamp_issuance_holds"
            )

    runtime.clear_custody_timestamp_issuance_hold_store_cache()
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH",
        str(path),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_custody_timestamp_issuance_hold_store(path)
    monkeypatch.delenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH"
    )

    calls = []
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: calls.append("actor"),
    )
    monkeypatch.setattr(
        cli,
        "get_custody_timestamp_issuance_hold_store",
        lambda path: calls.append("store"),
    )
    assert cli.main(
        [
            "release",
            placed.hold_id,
            "--owner-id",
            "alice",
            "--confirm-hold-id",
            "f" * 64,
        ]
    ) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }


def test_durable_issuance_holds_integrate_with_retention_and_release(
    tmp_path,
    monkeypatch,
    capsys,
):
    hold_path = tmp_path / "holds.sqlite3"
    store = CustodyTimestampIssuanceHoldStore(hold_path)
    issuance_id = "1" * 64
    hold = store.place(
        owner_id="alice",
        issuance_id=issuance_id,
        hold_key="hold",
        reason_code="legal_matter",
        actor=actor(),
        issuance_journal=Issuances(),
        now=10.0,
    )
    issuance = SimpleNamespace(
        issuance_id=issuance_id,
        authority_id="tsa",
        key_id="key",
        serial="2" * 64,
        attestation_digest="3" * 64,
        output_path_digest="4" * 64,
        state="cancelled",
        phase="planned",
        attempt_count=0,
        max_attempts=3,
        lease_owner=None,
        lease_expires_at=None,
        failure_type=None,
        updated_at=1.0,
        completed_at=None,
    )

    class Journal:
        def list(self, **kwargs):
            return (issuance,)

    monkeypatch.setattr(
        operations_cli,
        "ReadOnlyCustodyTimestampIssuanceJournal",
        lambda path: Journal(),
    )
    assert operations_cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--issuance-db-path",
            "/private/issuances.sqlite3",
            "--hold-db-path",
            str(hold_path),
            "--minimum-age-seconds",
            "0",
            "--include-completed",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    item = payload["items"][0]
    assert item["held"] is True
    assert item["retention_candidate"] is False
    assert payload["durable_active_hold_count"] == 1
    assert str(hold_path) not in json.dumps(payload)

    store.release(
        hold.hold_id,
        owner_id="alice",
        confirm_hold_id=hold.hold_id,
        actor=actor("release"),
        now=20.0,
    )
    assert operations_cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--issuance-db-path",
            "/private/issuances.sqlite3",
            "--hold-db-path",
            str(hold_path),
            "--minimum-age-seconds",
            "0",
            "--include-completed",
        ]
    ) == 0
    released_payload = json.loads(capsys.readouterr().out)
    assert released_payload["durable_active_hold_count"] == 0
