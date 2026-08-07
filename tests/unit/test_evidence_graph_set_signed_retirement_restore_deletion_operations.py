from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_operations as operations,
)
from tools import (
    evidence_graph_set_signed_retirement_restore_deletion_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_audit import (
    audit_restore_hold_placement_permits,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    _ensure_table,
    _permit_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


def deletion(
    digit: str,
    *,
    state: str,
    phase: str = "planned",
    attempts: int = 0,
    maximum: int = 3,
    lease: float | None = None,
    completed: float | None = None,
    restore_digit: str | None = None,
):
    return SimpleNamespace(
        deletion_id=digit * 64,
        authorization_id=((int(digit, 16) + 1) % 16).__format__("x") * 64,
        restore_id=(restore_digit or digit) * 64,
        snapshot_digest="a" * 64,
        target_path_digest="b" * 64,
        state=state,
        phase=phase,
        attempt_count=attempts,
        max_attempts=maximum,
        lease_owner="worker" if state == "running" else None,
        lease_expires_at=lease,
        marker_digest="c" * 64 if phase != "planned" else None,
        tombstone_digest="d" * 64 if phase in {"restore_deleted", "verified"} else None,
        custody_manifest_digest="e" * 64,
        failure_type="Stopped" if state == "failed" else None,
        updated_at=5.0,
        completed_at=completed,
    )


class Journal:
    def __init__(self, values):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def test_audit_classifies_all_states_and_is_digest_bound():
    values = (
        deletion("1", state="planned"),
        deletion("2", state="running", attempts=1, lease=20.0),
        deletion("3", state="running", attempts=1, lease=5.0),
        deletion("4", state="running", attempts=3, lease=5.0),
        deletion("5", state="failed", attempts=1),
        deletion("6", state="failed", attempts=3),
        deletion("7", state="completed", phase="verified", completed=2.0),
        deletion("8", state="cancelled", completed=2.0),
    )
    report = operations.audit_restore_deletion_operations(
        owner_id="alice",
        journal=Journal(values),
        now=10.0,
        limit=100,
    )
    assert report.item_count == 8
    assert all(value == 1 for value in report.classification_counts.values())
    assert report.mutation_performed is False
    assert report.restore_row_deleted is False
    assert len(report.report_digest) == 64


def test_audit_and_retention_refuse_duplicates_and_bounded_truncation():
    value = deletion("1", state="planned")
    with pytest.raises(RuntimeError, match="bounded"):
        operations.audit_restore_deletion_operations(
            owner_id="alice",
            journal=Journal((value,)),
            now=10.0,
            limit=1,
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        operations.audit_restore_deletion_operations(
            owner_id="alice",
            journal=Journal((value, value)),
            now=10.0,
            limit=10,
        )


def test_retention_protects_holds_latest_and_completed_by_default():
    old_cancelled = deletion(
        "1", state="cancelled", completed=1.0, restore_digit="f"
    )
    latest_cancelled = deletion(
        "2", state="cancelled", completed=2.0, restore_digit="f"
    )
    completed = deletion(
        "3",
        state="completed",
        phase="verified",
        completed=1.0,
        restore_digit="e",
    )
    plan = operations.plan_restore_deletion_retention(
        owner_id="alice",
        journal=Journal((old_cancelled, latest_cancelled, completed)),
        now=100.0,
        minimum_age_seconds=10.0,
        retain_latest_per_restore=1,
        include_completed=False,
        limit=100,
    )
    by_id = {item.deletion_id: item for item in plan.items}
    assert by_id[old_cancelled.deletion_id].retention_candidate is True
    assert by_id[latest_cancelled.deletion_id].reason == (
        "latest_terminal_for_restore"
    )
    assert by_id[completed.deletion_id].reason == (
        "latest_terminal_for_restore"
    )

    held = operations.plan_restore_deletion_retention(
        owner_id="alice",
        journal=Journal((old_cancelled, latest_cancelled)),
        now=100.0,
        minimum_age_seconds=10.0,
        retain_latest_per_restore=1,
        held_deletion_ids=(old_cancelled.deletion_id,),
        limit=100,
    )
    assert next(
        item for item in held.items if item.deletion_id == old_cancelled.deletion_id
    ).reason == "operator_hold"
    assert held.deletion_performed is False
    assert held.compaction_performed is False


class HoldStore:
    def __init__(self, values):
        self.values = values

    def get(self, hold_id):
        if hold_id not in self.values:
            raise KeyError(hold_id)
        return self.values[hold_id]


def test_permit_audit_distinguishes_replay_and_stale_cases(tmp_path):
    restore = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=1,
        now=1.0,
    )
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    journal.seed(restore)
    hold_ids = [character * 64 for character in "3456"]
    states = ["active", "active", "active", "released"]
    with journal._lock, journal._connect() as connection:
        _ensure_table(connection)
        for index, (hold_id, state) in enumerate(zip(hold_ids, states), 1):
            created = float(index)
            released = created + 1.0 if state == "released" else None
            updated = released or created
            digest = _permit_digest(
                hold_id=hold_id,
                owner_id="alice",
                restore_id=restore.restore_id,
                state=state,
                created_at=created,
                updated_at=updated,
                released_at=released,
            )
            connection.execute(
                "INSERT INTO signed_retirement_restore_hold_placement_permits "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    hold_id,
                    "alice",
                    restore.restore_id,
                    state,
                    digest,
                    created,
                    updated,
                    released,
                ),
            )
    holds = HoldStore(
        {
            hold_ids[0]: SimpleNamespace(
                owner_id="alice",
                restore_id=restore.restore_id,
                status="active",
            ),
            hold_ids[1]: SimpleNamespace(
                owner_id="alice",
                restore_id=restore.restore_id,
                status="released",
            ),
            hold_ids[3]: SimpleNamespace(
                owner_id="alice",
                restore_id=restore.restore_id,
                status="released",
            ),
        }
    )
    report = audit_restore_hold_placement_permits(
        owner_id="alice",
        restore_journal=journal,
        hold_store=holds,
        now=10.0,
        limit=100,
    )
    assert report.classification_counts == {
        "active_permit_with_active_hold": 1,
        "active_permit_with_released_hold": 1,
        "active_permit_without_hold_record": 1,
        "released_permit_history": 1,
    }
    replay = next(
        item
        for item in report.items
        if item.classification == "active_permit_with_active_hold"
    )
    assert replay.exact_hold_replay_recommended is True
    assert report.mutation_performed is False
    assert report.permit_released is False


def test_cli_has_only_read_only_commands(monkeypatch, capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["release-permit"])
    with pytest.raises(SystemExit):
        parser.parse_args(["delete"])

    report = operations.RestoreDeletionOperationalReport(
        owner_id="alice",
        generated_at=1.0,
        item_count=0,
        classification_counts={},
        items=(),
        report_digest="1" * 64,
        mutation_performed=False,
        restore_row_deleted=False,
        source_text_returned=False,
        raw_paths_returned=False,
    )
    monkeypatch.setattr(
        cli,
        "audit_restore_deletion_operations",
        lambda **kwargs: report,
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_deletion_journal",
        lambda: object(),
    )
    assert cli.main(["audit", "--owner-id", "alice"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mutation_performed"] is False
    assert payload["raw_paths_returned"] is False
