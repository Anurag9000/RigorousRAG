from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_hold_permit_recovery_operations as operations,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_hold_permit_recovery_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery_contracts import (
    RestoreHoldPermitRecoveryReceipt,
)


class RestoreJournal:
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE signed_retirement_restore_hold_placement_permits ("
                "hold_id TEXT PRIMARY KEY, owner_id TEXT, restore_id TEXT, "
                "state TEXT, permit_digest TEXT)"
            )

    def _connect(self):
        connection = sqlite3.connect(self.path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def permit(self, receipt, *, digest=None, state="released"):
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO signed_retirement_restore_hold_placement_permits "
                "VALUES (?,?,?,?,?)",
                (
                    receipt.hold_id,
                    receipt.owner_id,
                    receipt.restore_id,
                    state,
                    receipt.released_permit_digest if digest is None else digest,
                ),
            )


class HoldStore:
    def __init__(self, values=()):
        self.values = {value.hold_id: value for value in values}

    def get(self, hold_id):
        try:
            return self.values[hold_id]
        except KeyError as exc:
            raise KeyError(hold_id) from exc


def receipt(
    digit: str,
    *,
    restore_digit: str,
    recovered_at: float,
    quarantine_status: str | None,
):
    quarantine_id = None if quarantine_status is None else (digit.upper() * 64).lower()
    quarantine_digest = None if quarantine_status is None else ((int(digit, 16) + 1) % 16).__format__("x") * 64
    value = RestoreHoldPermitRecoveryReceipt.create(
        owner_id="alice",
        restore_id=restore_digit * 64,
        hold_id=digit * 64,
        original_permit_digest="a" * 64,
        released_permit_digest="b" * 64,
        classification=(
            "released_hold_cleanup"
            if quarantine_status is None
            else "abandoned_without_hold_quarantined"
        ),
        quarantine_hold_id=quarantine_id,
        quarantine_hold_digest=quarantine_digest,
        actor_id="operator",
        actor_binding_method="process_environment",
        actor_binding_digest="c" * 64,
        recovered_at=recovered_at,
    )
    hold = None
    if quarantine_status is not None:
        hold = SimpleNamespace(
            hold_id=value.quarantine_hold_id,
            owner_id="alice",
            restore_id=value.restore_id,
            hold_digest=value.quarantine_hold_digest,
            status=quarantine_status,
        )
    return value, hold


def install_receipts(monkeypatch, values):
    monkeypatch.setattr(
        operations,
        "list_hold_permit_recoveries",
        lambda journal, owner_id, limit: tuple(values),
    )


def test_audit_classifies_active_released_and_cleanup_receipts(tmp_path, monkeypatch):
    active, active_hold = receipt(
        "1", restore_digit="d", recovered_at=10.0, quarantine_status="active"
    )
    released, released_hold = receipt(
        "2", restore_digit="e", recovered_at=20.0, quarantine_status="released"
    )
    cleanup, _ = receipt(
        "3", restore_digit="f", recovered_at=30.0, quarantine_status=None
    )
    values = (active, released, cleanup)
    install_receipts(monkeypatch, values)
    journal = RestoreJournal(tmp_path / "restore.sqlite3")
    for value in values:
        journal.permit(value)
    store = HoldStore((active_hold, released_hold))

    report = operations.audit_hold_permit_recoveries(
        owner_id="alice",
        restore_journal=journal,
        hold_store=store,
        now=100.0,
        limit=100,
    )

    assert report.classification_counts["quarantine_active"] == 1
    assert report.classification_counts["quarantine_released"] == 1
    assert report.classification_counts["released_hold_cleanup"] == 1
    assert report.mutation_performed is False
    assert report.raw_paths_returned is False
    with pytest.raises(ValueError, match="digest differs"):
        replace(report, report_digest="f" * 64)


def test_audit_refuses_permit_and_quarantine_drift(tmp_path, monkeypatch):
    value, hold = receipt(
        "1", restore_digit="d", recovered_at=10.0, quarantine_status="active"
    )
    install_receipts(monkeypatch, (value,))
    journal = RestoreJournal(tmp_path / "restore.sqlite3")
    journal.permit(value, digest="f" * 64)
    with pytest.raises(RuntimeError, match="permit differs"):
        operations.audit_hold_permit_recoveries(
            owner_id="alice",
            restore_journal=journal,
            hold_store=HoldStore((hold,)),
            now=100.0,
        )

    journal = RestoreJournal(tmp_path / "restore-2.sqlite3")
    journal.permit(value)
    drifted = SimpleNamespace(**hold.__dict__)
    drifted.hold_digest = "f" * 64
    with pytest.raises(RuntimeError, match="quarantine hold differs"):
        operations.audit_hold_permit_recoveries(
            owner_id="alice",
            restore_journal=journal,
            hold_store=HoldStore((drifted,)),
            now=100.0,
        )


def test_retention_protects_active_quarantine_latest_and_explicit_holds(
    tmp_path,
    monkeypatch,
):
    old, _ = receipt(
        "1", restore_digit="d", recovered_at=100.0, quarantine_status=None
    )
    newest, _ = receipt(
        "2", restore_digit="d", recovered_at=900.0, quarantine_status=None
    )
    active, active_hold = receipt(
        "3", restore_digit="e", recovered_at=100.0, quarantine_status="active"
    )
    values = (old, newest, active)
    install_receipts(monkeypatch, values)
    journal = RestoreJournal(tmp_path / "restore.sqlite3")
    for value in values:
        journal.permit(value)
    store = HoldStore((active_hold,))

    plan = operations.plan_hold_permit_recovery_retention(
        owner_id="alice",
        restore_journal=journal,
        hold_store=store,
        now=1_000.0,
        minimum_age_seconds=50.0,
        retain_latest_per_restore=1,
        limit=100,
    )
    by_id = {item.recovery_id: item for item in plan.items}
    assert by_id[old.recovery_id].retention_candidate is True
    assert by_id[newest.recovery_id].reason == "latest_recovery_for_restore"
    assert by_id[active.recovery_id].reason == "active_quarantine_hold"
    assert plan.candidate_count == 1
    assert plan.deletion_performed is False

    held = operations.plan_hold_permit_recovery_retention(
        owner_id="alice",
        restore_journal=journal,
        hold_store=store,
        now=1_000.0,
        minimum_age_seconds=50.0,
        retain_latest_per_restore=1,
        held_recovery_ids=(old.recovery_id,),
        limit=100,
    )
    assert held.candidate_count == 0


@dataclass(frozen=True)
class Result:
    value: str
    mutation_performed: bool = False
    deletion_performed: bool = False
    source_text_returned: bool = False
    raw_paths_returned: bool = False


def test_cli_is_read_only_and_has_no_deletion_command(monkeypatch, capsys):
    journal = object()
    store = object()
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: journal,
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_hold_store",
        lambda: store,
    )
    monkeypatch.setattr(
        cli,
        "audit_hold_permit_recoveries",
        lambda **kwargs: Result("audit"),
    )
    monkeypatch.setattr(
        cli,
        "plan_hold_permit_recovery_retention",
        lambda **kwargs: Result("plan"),
    )

    assert cli.main(["audit", "--owner-id", "alice"]) == 0
    assert json.loads(capsys.readouterr().out)["mutation_performed"] is False
    assert cli.main(["retention-plan", "--owner-id", "alice"]) == 0
    assert json.loads(capsys.readouterr().out)["deletion_performed"] is False
    parser = cli.build_parser()
    command = next(
        action for action in parser._actions if action.dest == "command"
    )
    assert "delete" not in command.choices
    assert "release" not in command.choices
