from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_hold_cli as cli
from tools import evidence_graph_set_signed_retirement_restore_hold_runtime as runtime
from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_hold_integrity import (
    IntegritySignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    SignedRetirementRestoreHold,
    deterministic_restore_hold_id,
)


def actor(actor_id="operator-1", *, loaded_at=1.0):
    return ReviewActorBinding.create(
        actor_id=actor_id,
        binding_method="process_environment",
        loaded_at=loaded_at,
    )


class RestoreJournal:
    def __init__(self, *, owner_id="alice"):
        self.owner_id = owner_id
        self.calls = []

    def get(self, restore_id):
        self.calls.append(restore_id)
        return SimpleNamespace(owner_id=self.owner_id, restore_id=restore_id)


def test_hold_identity_place_and_collision_are_deterministic(tmp_path):
    store = IntegritySignedRetirementRestoreHoldStore(tmp_path / "holds.sqlite3")
    restore_id = "1" * 64
    binding = actor()
    value = store.place(
        owner_id="alice",
        restore_id=restore_id,
        hold_key="case-2026-001",
        reason_code="litigation",
        actor=binding,
        restore_journal=RestoreJournal(),
        now=2.0,
    )

    assert value.hold_id == deterministic_restore_hold_id(
        owner_id="alice",
        restore_id=restore_id,
        hold_key="case-2026-001",
    )
    assert value.status == "active"
    assert value.created_actor_id == "operator-1"
    assert len(value.hold_digest) == 64
    assert store.place(
        owner_id="alice",
        restore_id=restore_id,
        hold_key="case-2026-001",
        reason_code="litigation",
        actor=binding,
        restore_journal=RestoreJournal(),
        now=9.0,
    ) == value

    with pytest.raises(RuntimeError, match="collision"):
        store.place(
            owner_id="alice",
            restore_id=restore_id,
            hold_key="case-2026-001",
            reason_code="regulatory",
            actor=binding,
            restore_journal=RestoreJournal(),
            now=3.0,
        )


def test_hold_requires_restore_owner_scope_and_valid_actor(tmp_path):
    store = IntegritySignedRetirementRestoreHoldStore(tmp_path / "holds.sqlite3")
    with pytest.raises(RuntimeError, match="owner scope"):
        store.place(
            owner_id="alice",
            restore_id="1" * 64,
            hold_key="case",
            reason_code="litigation",
            actor=actor(),
            restore_journal=RestoreJournal(owner_id="bob"),
            now=2.0,
        )
    with pytest.raises(ValueError, match="ReviewActorBinding"):
        store.place(
            owner_id="alice",
            restore_id="1" * 64,
            hold_key="case",
            reason_code="litigation",
            actor=object(),
            restore_journal=RestoreJournal(),
            now=2.0,
        )


def test_release_is_exact_monotonic_and_never_reactivates(tmp_path):
    store = IntegritySignedRetirementRestoreHoldStore(tmp_path / "holds.sqlite3")
    value = store.place(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="case",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=2.0,
    )
    with pytest.raises(ValueError, match="confirmation"):
        store.release(
            value.hold_id,
            owner_id="alice",
            confirm_hold_id="f" * 64,
            actor=actor("operator-2"),
            now=3.0,
        )
    released = store.release(
        value.hold_id,
        owner_id="alice",
        confirm_hold_id=value.hold_id,
        actor=actor("operator-2"),
        now=3.0,
    )
    assert released.status == "released"
    assert released.released_actor_id == "operator-2"
    assert released.released_at == 3.0
    assert store.release(
        value.hold_id,
        owner_id="alice",
        confirm_hold_id=value.hold_id,
        actor=actor("operator-3"),
        now=4.0,
    ) == released
    replayed_place = store.place(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="case",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=5.0,
    )
    assert replayed_place.status == "released"


def test_active_ids_listing_and_complete_row_tamper_detection(tmp_path):
    path = tmp_path / "holds.sqlite3"
    store = IntegritySignedRetirementRestoreHoldStore(path)
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
        actor=actor("operator-2"),
        now=4.0,
    )
    assert store.active_restore_ids(owner_id="alice", limit=100) == frozenset(
        {first.restore_id}
    )
    assert tuple(value.hold_id for value in store.list(
        owner_id="alice", status="released", limit=100
    )) == (second.hold_id,)

    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_restore_holds "
            "SET reason_code=? WHERE hold_id=?",
            ("tampered", second.hold_id),
        )
    with pytest.raises(RuntimeError, match="integrity differs"):
        store.get(second.hold_id)

    with store._lock, store._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_restore_holds "
            "SET restore_id=? WHERE hold_id=?",
            ("f" * 64, first.hold_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(first.hold_id)


def test_missing_integrity_and_database_identity_fail_closed(tmp_path):
    path = tmp_path / "holds.sqlite3"
    store = IntegritySignedRetirementRestoreHoldStore(path)
    value = store.place(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="case",
        reason_code="litigation",
        actor=actor(),
        restore_journal=RestoreJournal(),
        now=2.0,
    )
    with store._lock, store._connect() as connection:
        connection.execute(
            "DELETE FROM evidence_graph_set_signed_restore_hold_integrity "
            "WHERE hold_id=?",
            (value.hold_id,),
        )
    with pytest.raises(RuntimeError, match="integrity record is missing"):
        store.get(value.hold_id)

    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        store.list(owner_id="alice")


def test_hold_runtime_aliases_fail_closed(tmp_path, monkeypatch):
    runtime.clear_signed_retirement_restore_hold_store_cache()
    protected = tmp_path / "restore.sqlite3"
    protected.write_bytes(b"database")
    monkeypatch.setenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        str(protected),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        runtime.get_signed_retirement_restore_hold_store(path=protected)


def test_hold_cli_confirmation_actor_and_read_boundaries(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cli,
        "load_relation_review_actor",
        lambda: calls.append("actor") or actor(),
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_hold_store",
        lambda: calls.append("store") or object(),
    )
    assert cli.main([
        "place",
        "1" * 64,
        "--owner-id",
        "alice",
        "--confirm-restore-id",
        "2" * 64,
        "--hold-key",
        "case",
        "--reason-code",
        "litigation",
    ]) == 2
    assert calls == []
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }

    value = SignedRetirementRestoreHold.create(
        owner_id="alice",
        restore_id="1" * 64,
        hold_key="case",
        reason_code="litigation",
        actor=actor(),
        now=2.0,
    )

    class Store:
        def get(self, hold_id):
            return value

        def list(self, **kwargs):
            return (value,)

        def active_restore_ids(self, **kwargs):
            return frozenset({value.restore_id})

    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_hold_store",
        lambda: Store(),
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: (_ for _ in ()).throw(
            AssertionError("read commands must not load restore journal")
        ),
    )
    assert cli.main(["status", value.hold_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["mutation_performed"] is False
    assert status["raw_paths_returned"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["mutation_performed"] is False

    assert cli.main(["active-restore-ids", "--owner-id", "alice"]) == 0
    active = json.loads(capsys.readouterr().out)
    assert active["restore_ids"] == [value.restore_id]
    assert active["mutation_performed"] is False
