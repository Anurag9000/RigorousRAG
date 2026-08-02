from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_retirement_operations_cli as cli
from tools.evidence_graph_set_signed_retirement_operations import (
    audit_signed_retirement_operations,
    plan_signed_retirement_retention,
)


def value(
    digit: str,
    *,
    operation_digit: str | None = None,
    state: str,
    phase: str,
    attempt_count: int = 1,
    max_attempts: int = 3,
    lease_owner=None,
    lease_expires_at=None,
    updated_at: float = 10.0,
    completed_at=None,
    failure_type=None,
):
    return SimpleNamespace(
        retirement_id=digit * 64,
        publication_operation_id=(operation_digit or digit) * 64,
        graph_set_key="review",
        state=state,
        phase=phase,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        lease_owner=lease_owner,
        lease_expires_at=lease_expires_at,
        updated_at=updated_at,
        completed_at=completed_at,
        failure_type=failure_type,
    )


class Journal:
    def __init__(self, values=()):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self.values[: kwargs["limit"]]


def test_operational_audit_classifies_queue_and_failure_states():
    journal = Journal(
        (
            value("1", state="planned", phase="planned", attempt_count=0),
            value(
                "2",
                state="running",
                phase="pointer_restore_intent",
                lease_owner="worker",
                lease_expires_at=20.0,
            ),
            value(
                "3",
                state="running",
                phase="pointer_safe",
                lease_owner="worker",
                lease_expires_at=9.0,
            ),
            value(
                "4",
                state="running",
                phase="pointer_safe",
                attempt_count=3,
                max_attempts=3,
                lease_owner="worker",
                lease_expires_at=9.0,
            ),
            value(
                "5",
                state="failed",
                phase="pointer_safe",
                attempt_count=1,
                failure_type="PointerFailure",
            ),
            value(
                "6",
                state="failed",
                phase="authorization_retired",
                attempt_count=3,
                max_attempts=3,
                failure_type="AuthorityFailure",
            ),
            value(
                "7",
                state="completed",
                phase="verified",
                completed_at=8.0,
            ),
            value(
                "8",
                state="cancelled",
                phase="planned",
                completed_at=8.0,
            ),
        )
    )

    report = audit_signed_retirement_operations(
        owner_id="alice",
        journal=journal,
        now=10.0,
        limit=100,
    )

    classifications = {
        item.retirement_id[0]: item.classification for item in report.items
    }
    assert classifications == {
        "1": "planned_ready",
        "2": "running_active",
        "3": "running_expired_reclaimable",
        "4": "running_expired_exhausted",
        "5": "failed_retryable",
        "6": "failed_exhausted",
        "7": "completed",
        "8": "cancelled",
    }
    assert report.classification_counts["running_active"] == 1
    assert report.classification_counts["failed_exhausted"] == 1
    assert report.item_count == 8
    assert len(report.report_digest) == 64
    assert report.mutation_performed is False
    assert report.source_text_returned is False


def test_operational_audit_fails_closed_at_bound_and_duplicate_ids():
    one = value("1", state="planned", phase="planned")
    with pytest.raises(RuntimeError, match="bounded result limit"):
        audit_signed_retirement_operations(
            owner_id="alice",
            journal=Journal((one,)),
            now=1.0,
            limit=1,
        )
    with pytest.raises(RuntimeError, match="duplicate IDs"):
        audit_signed_retirement_operations(
            owner_id="alice",
            journal=Journal((one, one)),
            now=1.0,
            limit=3,
        )


def test_retention_plan_protects_latest_holds_and_completed_by_default():
    values = (
        value(
            "1",
            operation_digit="a",
            state="cancelled",
            phase="planned",
            updated_at=1.0,
            completed_at=1.0,
        ),
        value(
            "2",
            operation_digit="a",
            state="cancelled",
            phase="planned",
            updated_at=2.0,
            completed_at=2.0,
        ),
        value(
            "3",
            operation_digit="b",
            state="completed",
            phase="verified",
            updated_at=1.0,
            completed_at=1.0,
        ),
        value(
            "4",
            operation_digit="c",
            state="failed",
            phase="pointer_safe",
            failure_type="Failure",
        ),
    )
    plan = plan_signed_retirement_retention(
        owner_id="alice",
        journal=Journal(values),
        now=100.0,
        minimum_age_seconds=10.0,
        held_retirement_ids=("1" * 64,),
        limit=100,
    )
    items = {item.retirement_id[0]: item for item in plan.items}

    assert items["1"].held is True
    assert items["1"].retention_candidate is False
    assert items["2"].protected_as_latest is True
    assert items["2"].retention_candidate is False
    assert items["3"].reason == "latest_terminal_for_operation"
    assert "4" not in items
    assert plan.candidate_count == 0
    assert plan.deletion_performed is False


def test_retention_plan_marks_only_old_unheld_terminal_duplicates():
    values = (
        value(
            "1",
            operation_digit="a",
            state="cancelled",
            phase="planned",
            completed_at=1.0,
        ),
        value(
            "2",
            operation_digit="a",
            state="cancelled",
            phase="planned",
            completed_at=2.0,
        ),
        value(
            "3",
            operation_digit="b",
            state="completed",
            phase="verified",
            completed_at=1.0,
        ),
        value(
            "4",
            operation_digit="b",
            state="completed",
            phase="verified",
            completed_at=2.0,
        ),
    )
    plan = plan_signed_retirement_retention(
        owner_id="alice",
        journal=Journal(values),
        now=100.0,
        minimum_age_seconds=10.0,
        retain_latest_per_operation=1,
        include_completed=True,
        limit=100,
    )
    items = {item.retirement_id[0]: item for item in plan.items}

    assert items["1"].retention_candidate is True
    assert items["1"].reason == "old_terminal_duplicate_candidate"
    assert items["2"].retention_candidate is False
    assert items["3"].retention_candidate is True
    assert items["4"].retention_candidate is False
    assert plan.candidate_count == 2
    assert len(plan.plan_digest) == 64


def test_operations_cli_is_text_free_and_non_mutating(monkeypatch, capsys):
    journal = Journal((value("1", state="planned", phase="planned"),))
    monkeypatch.setattr(
        cli,
        "get_signed_publication_retirement_journal",
        lambda: journal,
    )

    assert cli.main(["audit", "--owner-id", "alice", "--limit", "100"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["journal_mutation_performed"] is False
    assert payload["pointer_mutation_performed"] is False
    assert payload["deletion_performed"] is False
    assert payload["source_text_returned"] is False

    assert cli.main([
        "retention-plan",
        "--owner-id", "alice",
        "--minimum-age-days", "1",
        "--limit", "100",
    ]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["deletion_performed"] is False
    assert payload["journal_mutation_performed"] is False
    assert "private text" not in json.dumps(payload).lower()
