from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_operations_cli as cli
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_operations import (
    audit_signed_retirement_restore_operations,
    plan_signed_retirement_restore_retention,
)


def base(
    snapshot_digit: str,
    *,
    target_digit: str = "a",
    now: float = 1.0,
    max_attempts: int = 3,
):
    return SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest=snapshot_digit * 64,
        target_path_digest=target_digit * 64,
        snapshot_record_count=1,
        max_attempts=max_attempts,
        now=now,
    )


def running(value, *, attempts: int, lease_expires: float):
    return replace(
        value,
        state="running",
        attempt_count=attempts,
        lease_owner="worker",
        lease_expires_at=lease_expires,
        updated_at=2.0,
    )


def failed(value, *, attempts: int):
    return replace(
        value,
        state="failed",
        attempt_count=attempts,
        failure_type="GenericFailure",
        updated_at=2.0,
    )


def cancelled(value, *, completed_at: float):
    return replace(
        value,
        state="cancelled",
        updated_at=completed_at,
        completed_at=completed_at,
    )


def completed(value, *, completed_at: float):
    return replace(
        value,
        state="completed",
        phase="verified",
        target_verification_digest="f" * 64,
        updated_at=completed_at,
        completed_at=completed_at,
    )


class Journal:
    def __init__(self, values=()):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        values = self.values
        state = kwargs.get("state")
        if state is not None:
            values = tuple(value for value in values if value.state == state)
        return values[: kwargs["limit"]]


def test_restore_operational_audit_classifies_every_state_and_lease():
    values = (
        base("1"),
        running(base("2"), attempts=1, lease_expires=20.0),
        running(base("3"), attempts=1, lease_expires=9.0),
        running(base("4", max_attempts=1), attempts=1, lease_expires=9.0),
        failed(base("5"), attempts=1),
        failed(base("6", max_attempts=1), attempts=1),
        completed(base("7"), completed_at=5.0),
        cancelled(base("8"), completed_at=6.0),
    )
    report = audit_signed_retirement_restore_operations(
        owner_id="alice",
        journal=Journal(values),
        now=10.0,
        limit=100,
    )

    assert report.item_count == 8
    assert report.classification_counts == {
        "cancelled": 1,
        "completed": 1,
        "failed_exhausted": 1,
        "failed_retryable": 1,
        "planned_ready": 1,
        "running_active": 1,
        "running_expired_exhausted": 1,
        "running_expired_reclaimable": 1,
    }
    active = next(
        value for value in report.items if value.classification == "running_active"
    )
    expired = next(
        value
        for value in report.items
        if value.classification == "running_expired_reclaimable"
    )
    assert active.lease_owner_present is True
    assert active.lease_active is True
    assert expired.lease_expired is True
    assert report.mutation_performed is False
    assert report.source_text_returned is False
    assert report.raw_paths_returned is False
    assert len(report.report_digest) == 64


def test_restore_audit_filters_by_digest_and_fails_closed_at_limit():
    first = base("1", target_digit="a")
    second = base("2", target_digit="b")
    journal = Journal((first, second))
    report = audit_signed_retirement_restore_operations(
        owner_id="alice",
        journal=journal,
        snapshot_digest=first.snapshot_digest,
        target_path_digest=first.target_path_digest,
        now=10.0,
        limit=100,
    )
    assert tuple(value.restore_id for value in report.items) == (first.restore_id,)

    with pytest.raises(RuntimeError, match="bounded result limit"):
        audit_signed_retirement_restore_operations(
            owner_id="alice",
            journal=Journal((first,)),
            now=10.0,
            limit=1,
        )


def test_restore_audit_and_retention_refuse_duplicate_ids():
    value = base("1")
    duplicate = Journal((value, value))
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_signed_retirement_restore_operations(
            owner_id="alice",
            journal=duplicate,
            now=10.0,
            limit=100,
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        plan_signed_retirement_restore_retention(
            owner_id="alice",
            journal=duplicate,
            now=100.0,
            minimum_age_seconds=1.0,
            limit=100,
        )


def test_restore_retention_protects_holds_latest_failures_and_completed_defaults():
    old_cancelled = cancelled(base("1", target_digit="a"), completed_at=10.0)
    latest_cancelled = cancelled(base("2", target_digit="a"), completed_at=20.0)
    held_cancelled = cancelled(base("3", target_digit="b"), completed_at=10.0)
    latest_other_target = cancelled(base("4", target_digit="b"), completed_at=20.0)
    old_completed = completed(base("5", target_digit="c"), completed_at=10.0)
    latest_completed = completed(base("6", target_digit="c"), completed_at=20.0)
    retryable = failed(base("7", target_digit="d"), attempts=1)

    plan = plan_signed_retirement_restore_retention(
        owner_id="alice",
        journal=Journal(
            (
                old_cancelled,
                latest_cancelled,
                held_cancelled,
                latest_other_target,
                old_completed,
                latest_completed,
                retryable,
            )
        ),
        now=100.0,
        minimum_age_seconds=30.0,
        held_restore_ids=(held_cancelled.restore_id,),
        limit=100,
    )
    by_id = {value.restore_id: value for value in plan.items}

    assert by_id[old_cancelled.restore_id].retention_candidate is True
    assert by_id[latest_cancelled.restore_id].reason == "latest_terminal_for_target"
    assert by_id[held_cancelled.restore_id].reason == "legal_hold"
    assert by_id[latest_other_target.restore_id].reason == "latest_terminal_for_target"
    assert (
        by_id[old_completed.restore_id].reason
        == "completed_restores_retained_by_default"
    )
    assert by_id[latest_completed.restore_id].reason == "latest_terminal_for_target"
    assert retryable.restore_id not in by_id
    assert plan.candidate_count == 1
    assert plan.deletion_performed is False
    assert plan.journal_mutation_performed is False

    include_completed = plan_signed_retirement_restore_retention(
        owner_id="alice",
        journal=Journal((old_completed, latest_completed)),
        now=100.0,
        minimum_age_seconds=30.0,
        include_completed=True,
        limit=100,
    )
    include_by_id = {value.restore_id: value for value in include_completed.items}
    assert include_by_id[old_completed.restore_id].retention_candidate is True
    assert include_by_id[latest_completed.restore_id].retention_candidate is False


def test_restore_operations_cli_is_read_only_path_free_and_has_no_delete(
    monkeypatch,
    capsys,
):
    value = cancelled(base("1"), completed_at=2.0)
    journal = Journal((value,))
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: journal,
    )

    assert cli.main(["audit", "--owner-id", "alice", "--limit", "100"]) == 0
    audit_payload = json.loads(capsys.readouterr().out)
    assert audit_payload["journal_mutation_performed"] is False
    assert audit_payload["target_mutation_performed"] is False
    assert audit_payload["deletion_performed"] is False
    assert audit_payload["raw_paths_returned"] is False

    assert cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--minimum-age-seconds",
            "0",
            "--limit",
            "100",
        ]
    ) == 0
    retention_payload = json.loads(capsys.readouterr().out)
    assert retention_payload["journal_mutation_performed"] is False
    assert retention_payload["target_mutation_performed"] is False
    assert retention_payload["deletion_performed"] is False
    assert retention_payload["raw_paths_returned"] is False

    rendered = json.dumps((audit_payload, retention_payload)).lower()
    assert "/tmp/" not in rendered
    assert "target.sqlite3" not in rendered
    commands = cli.build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {"audit", "retention-plan"}
