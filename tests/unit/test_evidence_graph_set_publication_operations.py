from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

from tools import evidence_graph_set_publish_operations_cli as cli
from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationAttempt,
)
from tools.evidence_graph_set_publish_operations import (
    audit_publication_attempts,
    plan_publication_retention,
)


def attempt(
    digit,
    *,
    state="planned",
    phase="planned",
    attempts=0,
    maximum=3,
    now=1.0,
):
    value = EvidenceGraphSetPublicationAttempt.create(
        owner_id="alice",
        graph_set_key="review",
        proposal_ids=(digit * 64,),
        expected_current_set_id=None,
        max_attempts=maximum,
        now=now,
    )
    if state == "planned":
        return value
    candidate = digit * 64
    common = dict(
        state=state,
        phase=phase,
        attempt_count=attempts,
        lease_owner=None,
        lease_expires_at=None,
        updated_at=now + 1,
    )
    if phase != "planned":
        common.update(
            previous_graph_set_id=None,
            previous_graph_set_digest=None,
            candidate_graph_set_id=candidate,
            candidate_graph_set_digest=digit * 64,
            member_count=2,
            edge_count=1,
        )
    if state == "running":
        common.update(lease_owner="worker", lease_expires_at=now + 5)
    if state == "completed":
        common.update(
            phase="verified",
            verification_digest="a" * 64,
            completed_at=now + 1,
        )
    elif state == "compensated":
        common.update(
            phase="compensated",
            verification_digest="b" * 64,
            failure_type="Boom",
            completed_at=now + 1,
        )
    elif state == "failed":
        common.update(failure_type="Boom")
    elif state == "cancelled":
        common.update(completed_at=now + 1)
    return replace(value, **common)


class Journal:
    def __init__(self, values):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


class Store:
    def __init__(self, current=None):
        self.value = current

    def current(self, **kwargs):
        return self.value


def test_audit_classifies_expired_retryable_and_exhausted():
    running = replace(
        attempt("1", state="running", attempts=1), lease_expires_at=3.0
    )
    failed = attempt(
        "2", state="failed", phase="candidate_stored", attempts=1
    )
    exhausted = attempt(
        "3", state="failed", phase="candidate_stored", attempts=3
    )
    report = audit_publication_attempts(
        Journal((running, failed, exhausted)),
        owner_id="alice",
        now=10.0,
    )
    assert report.classification_counts == {
        "expired_reclaimable": 1,
        "failed_exhausted": 1,
        "failed_retryable": 1,
    }
    assert len(report.report_digest) == 64


def test_retention_plan_never_selects_current_recent_or_failure():
    old = attempt(
        "1", state="completed", phase="verified", attempts=1, now=1.0
    )
    recent = attempt(
        "2", state="completed", phase="verified", attempts=1, now=95.0
    )
    failed = attempt(
        "3", state="failed", phase="candidate_stored", attempts=1, now=1.0
    )
    current = SimpleNamespace(graph_set_id=recent.candidate_graph_set_id)
    plan = plan_publication_retention(
        Journal((old, recent, failed)),
        set_store=Store(current),
        owner_id="alice",
        minimum_age_seconds=20,
        now=100.0,
    )
    reasons = {item.operation_id: item.reason for item in plan.items}
    assert reasons[old.operation_id] == "old_terminal_noncurrent"
    assert reasons[recent.operation_id] == "recent_terminal"
    assert reasons[failed.operation_id] == "failure_record"
    assert plan.eligible_count == 1
    assert plan.deletion_performed is False


def test_retention_keeps_old_terminal_that_references_current_pointer():
    value = attempt(
        "4", state="compensated", phase="compensated", attempts=1, now=1.0
    )
    current = SimpleNamespace(graph_set_id=value.candidate_graph_set_id)
    plan = plan_publication_retention(
        Journal((value,)),
        set_store=Store(current),
        owner_id="alice",
        minimum_age_seconds=1,
        now=100.0,
    )
    assert plan.items[0].reason == "references_current_pointer"
    assert plan.eligible_count == 0


def test_operations_cli_is_read_only(monkeypatch, capsys):
    value = attempt("1")
    monkeypatch.setattr(
        cli,
        "get_evidence_graph_set_publication_journal",
        lambda: Journal((value,)),
    )
    monkeypatch.setattr(cli, "get_evidence_graph_set_store", lambda: Store())
    assert cli.main(["audit", "--owner-id", "alice"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mutation_performed"] is False
    assert payload["source_text_returned"] is False
    assert cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--minimum-age-seconds",
            "0",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["deletion_performed"] is False
