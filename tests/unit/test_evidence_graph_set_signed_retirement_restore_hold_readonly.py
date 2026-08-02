from __future__ import annotations

import json

import pytest

from tools import evidence_graph_set_signed_retirement_restore_operations_cli as cli
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_integrity import (
    IntegritySignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_readonly import (
    ReadOnlySignedRetirementRestoreHoldStore,
)


def actor():
    return ReviewActorBinding.create(
        actor_id="operator",
        binding_method="process_environment",
        loaded_at=1.0,
    )


class RestoreJournal:
    def get(self, restore_id):
        return type("Restore", (), {"owner_id": "alice"})()


class OperationsJournal:
    def __init__(self, value):
        self.value = value

    def list(self, **kwargs):
        return (self.value,)


def restore_value():
    value = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=1,
        now=1.0,
    )
    from dataclasses import replace

    return replace(
        value,
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


def test_read_only_hold_view_returns_verified_active_ids(tmp_path):
    store = IntegritySignedRetirementRestoreHoldStore(tmp_path / "holds.sqlite3")
    first = store.place(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="one",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=2.0,
    )
    second = store.place(
        owner_id="alice",
        restore_id="2" * 64,
        hold_key="two",
        reason_code="regulatory",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=3.0,
    )
    store.release(
        second.hold_id,
        owner_id="alice",
        confirm_hold_id=second.hold_id,
        actor=actor(),
        now=4.0,
    )

    readonly = ReadOnlySignedRetirementRestoreHoldStore(store.path)
    assert readonly.active_restore_ids(
        owner_id="alice", limit=100
    ) == frozenset({first.restore_id})
    with readonly._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_set_signed_restore_holds"
            )


def test_read_only_hold_view_fails_on_missing_or_tampered_integrity(tmp_path):
    store = IntegritySignedRetirementRestoreHoldStore(tmp_path / "holds.sqlite3")
    value = store.place(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="one",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=2.0,
    )
    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_restore_hold_integrity "
            "SET hold_digest=? WHERE hold_id=?",
            ("f" * 64, value.hold_id),
        )
    readonly = ReadOnlySignedRetirementRestoreHoldStore(store.path)
    with pytest.raises(RuntimeError, match="integrity differs"):
        readonly.active_restore_ids(owner_id="alice", limit=100)


def test_retention_cli_unions_explicit_and_durable_holds_without_mutation(
    tmp_path,
    monkeypatch,
    capsys,
):
    value = restore_value()
    hold_store = IntegritySignedRetirementRestoreHoldStore(
        tmp_path / "holds.sqlite3"
    )
    hold_store.place(
        owner_id="alice",
        restore_id=value.restore_id,
        hold_key="case",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=3.0,
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: OperationsJournal(value),
    )

    before = hold_store.path.read_bytes()
    assert cli.main([
        "retention-plan",
        "--owner-id",
        "alice",
        "--minimum-age-seconds",
        "0",
        "--durable-hold-db-path",
        str(hold_store.path),
        "--hold-restore-id",
        "3" * 64,
        "--limit",
        "100",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["durable_hold_count"] == 1
    assert payload["explicit_hold_count"] == 1
    item = next(
        entry
        for entry in payload["items"]
        if entry["restore_id"] == value.restore_id
    )
    assert item["held"] is True
    assert item["reason"] == "legal_hold"
    assert payload["hold_store_mutation_performed"] is False
    assert payload["deletion_performed"] is False
    assert payload["raw_paths_returned"] is False
    assert hold_store.path.read_bytes() == before
