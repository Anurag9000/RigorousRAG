from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import evidence_graph_set_signed_transition_cli as cli
from tools.evidence_graph_set_signed_transition import (
    assess_signed_publication_transition,
)


def attempt(
    digit: str,
    *,
    state: str,
    phase: str = "planned",
    lease_expires_at=None,
    candidate=False,
):
    return SimpleNamespace(
        operation_id=digit * 64,
        owner_id="alice",
        graph_set_key="review",
        state=state,
        phase=phase,
        expected_current_set_id=None,
        candidate_graph_set_id=("f" * 64 if candidate else None),
        lease_expires_at=lease_expires_at,
    )


class Journal:
    def __init__(self, values=()):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return self.values[: kwargs["limit"]]


def test_transition_audit_classifies_authorization_only_attempts():
    common = Journal(
        (
            attempt("1", state="planned"),
            attempt("2", state="failed", phase="candidate_stored", candidate=True),
            attempt(
                "3",
                state="running",
                phase="candidate_stored",
                lease_expires_at=20.0,
                candidate=True,
            ),
            attempt("4", state="completed", phase="verified", candidate=True),
            attempt("5", state="cancelled"),
        )
    )
    signed = Journal()

    report = assess_signed_publication_transition(
        owner_id="alice",
        authorization_journal=common,
        signed_journal=signed,
        graph_set_key="review",
        now=10.0,
        limit=100,
    )

    actions = {item.operation_id[0]: item.action for item in report.items}
    assert actions == {
        "1": "cancel_authorization_only_then_reseed_signed",
        "2": "cancel_authorization_only_then_reseed_signed",
        "3": "wait_for_authorization_only_lease",
        "4": "do_not_claim_signed_provenance_reseed_with_current_pointer_if_needed",
        "5": "no_signed_transition_required",
    }
    active = next(item for item in report.items if item.operation_id == "3" * 64)
    assert active.lease_active is True
    assert report.authorization_attempt_count == 5
    assert report.signed_attempt_count == 0
    assert report.actionable_count == 4
    assert len(report.report_digest) == 64
    assert report.mutation_performed is False
    assert report.source_text_returned is False
    assert common.calls == [
        {"owner_id": "alice", "graph_set_key": "review", "limit": 100}
    ]


def test_transition_audit_detects_signed_attempts_and_expired_leases():
    common = Journal(
        (
            attempt(
                "1",
                state="running",
                phase="candidate_stored",
                lease_expires_at=9.0,
                candidate=True,
            ),
            attempt("2", state="planned"),
            attempt("3", state="planned"),
        )
    )
    signed = Journal(
        (
            attempt("2", state="planned"),
            attempt("3", state="completed", phase="verified", candidate=True),
        )
    )

    report = assess_signed_publication_transition(
        owner_id="alice",
        authorization_journal=common,
        signed_journal=signed,
        now=10.0,
    )
    items = {value.operation_id[0]: value for value in report.items}

    assert items["1"].lease_active is False
    assert (
        items["1"].action
        == "inspect_expired_authorization_only_lease_then_cancel_or_retry"
    )
    assert items["2"].signed_attempt_present is True
    assert items["2"].action == "resolve_duplicate_nonterminal_attempts"
    assert items["3"].signed_state == "completed"
    assert (
        items["3"].action
        == "cancel_authorization_only_duplicate_after_signed_completion"
    )
    assert report.actionable_count == 3


def test_completed_twins_need_no_transition_action():
    common = Journal((attempt("1", state="completed", phase="verified", candidate=True),))
    signed = Journal((attempt("1", state="completed", phase="verified", candidate=True),))

    report = assess_signed_publication_transition(
        owner_id="alice",
        authorization_journal=common,
        signed_journal=signed,
        now=10.0,
    )

    assert report.actionable_count == 0
    assert report.items[0].action == "signed_attempt_already_completed"


def test_transition_report_revalidates_counts_and_digest():
    report = assess_signed_publication_transition(
        owner_id="alice",
        authorization_journal=Journal((attempt("1", state="planned"),)),
        signed_journal=Journal(),
        now=10.0,
    )

    with pytest.raises(ValueError, match="actionable_count"):
        replace(report, actionable_count=0)
    with pytest.raises(ValueError, match="report_digest"):
        replace(report, report_digest="f" * 64)


def test_transition_audit_is_deterministic_for_same_time_and_inputs():
    common = Journal((attempt("1", state="planned"),))
    signed = Journal()
    first = assess_signed_publication_transition(
        owner_id="alice",
        authorization_journal=common,
        signed_journal=signed,
        now=10.0,
    )
    second = assess_signed_publication_transition(
        owner_id="alice",
        authorization_journal=common,
        signed_journal=signed,
        now=10.0,
    )
    assert first == second


def test_transition_audit_fails_closed_at_result_limit():
    common = Journal((attempt("1", state="planned"),))
    with pytest.raises(RuntimeError, match="bounded result limit"):
        assess_signed_publication_transition(
            owner_id="alice",
            authorization_journal=common,
            signed_journal=Journal(),
            now=1.0,
            limit=1,
        )


def test_transition_cli_is_read_only_and_text_free(monkeypatch, capsys):
    common = Journal((attempt("1", state="planned"),))
    signed = Journal()
    monkeypatch.setattr(
        cli, "get_evidence_graph_set_publication_journal", lambda: common
    )
    monkeypatch.setattr(
        cli,
        "get_evidence_graph_set_signed_publication_journal",
        lambda: signed,
    )

    assert cli.main([
        "audit",
        "--owner-id", "alice",
        "--graph-set-key", "review",
        "--limit", "100",
    ]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["actionable_count"] == 1
    assert payload["automatic_migration_performed"] is False
    assert payload["publication_mutation_performed"] is False
    assert payload["journal_mutation_performed"] is False
    assert payload["source_text_returned"] is False
    rendered = json.dumps(payload).lower()
    assert "private text" not in rendered
    assert "source_text" in rendered
