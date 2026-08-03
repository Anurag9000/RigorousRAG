from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_operations import (
    audit_custody_timestamp_issuances,
    plan_custody_timestamp_issuance_retention,
)


class Journal:
    def __init__(self, values=()):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def value(
    digit: str,
    *,
    state: str,
    phase: str = "planned",
    attempts: int = 0,
    maximum: int = 3,
    lease: float | None = None,
    completed: float | None = None,
    updated: float = 10.0,
):
    return SimpleNamespace(
        issuance_id=digit * 64,
        authority_id="tsa",
        key_id="key",
        serial=(hex((int(digit, 16) + 1) % 16)[2:]) * 64,
        attestation_digest=(hex((int(digit, 16) + 2) % 16)[2:]) * 64,
        output_path_digest=(hex((int(digit, 16) + 3) % 16)[2:]) * 64,
        state=state,
        phase=phase,
        attempt_count=attempts,
        max_attempts=maximum,
        lease_owner="worker" if lease is not None else None,
        lease_expires_at=lease,
        failure_type="Failure" if state == "failed" else None,
        updated_at=updated,
        completed_at=completed,
    )


def test_timestamp_issuance_audit_classifies_all_states():
    values = (
        value("1", state="planned"),
        value("2", state="running", attempts=1, lease=200.0),
        value("3", state="running", attempts=1, lease=50.0),
        value("4", state="running", attempts=3, maximum=3, lease=50.0),
        value("5", state="failed", attempts=1),
        value("6", state="failed", attempts=3, maximum=3),
        value("7", state="completed", phase="verified", completed=80.0),
        value("8", state="cancelled", completed=None),
    )
    report = audit_custody_timestamp_issuances(
        owner_id="alice",
        journal=Journal(values),
        now=100.0,
        limit=100,
    )

    assert set(report.classification_counts) == {
        "planned_ready",
        "running_active",
        "running_expired_reclaimable",
        "running_expired_exhausted",
        "failed_retryable",
        "failed_exhausted",
        "completed",
        "cancelled",
    }
    assert all(count == 1 for count in report.classification_counts.values())
    assert report.mutation_performed is False
    assert report.contains_attestation_signatures is False
    assert len(report.report_digest) == 64


def test_timestamp_issuance_audit_refuses_duplicates_bounds_and_tamper():
    duplicate = value("1", state="planned")
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_custody_timestamp_issuances(
            owner_id="alice",
            journal=Journal((duplicate, duplicate)),
            now=100.0,
        )
    with pytest.raises(RuntimeError, match="bounded"):
        audit_custody_timestamp_issuances(
            owner_id="alice",
            journal=Journal((duplicate,)),
            now=100.0,
            limit=1,
        )
    report = audit_custody_timestamp_issuances(
        owner_id="alice",
        journal=Journal((duplicate,)),
        now=100.0,
    )
    with pytest.raises(ValueError, match="report_digest"):
        replace(report, report_digest="f" * 64)


def test_timestamp_issuance_retention_protects_holds_latest_and_completed():
    old_cancelled = value("1", state="cancelled", updated=1.0)
    newest_cancelled = value("2", state="cancelled", updated=2.0)
    old_completed = value(
        "3",
        state="completed",
        phase="verified",
        completed=1.0,
        updated=1.0,
    )
    active = value("4", state="planned")
    plan = plan_custody_timestamp_issuance_retention(
        owner_id="alice",
        journal=Journal((old_cancelled, newest_cancelled, old_completed, active)),
        now=1_000.0,
        minimum_age_seconds=10.0,
        retain_latest_per_authority_key=1,
        include_completed=False,
        held_issuance_ids=(old_cancelled.issuance_id,),
    )
    by_id = {item.issuance_id: item for item in plan.items}
    assert by_id[old_cancelled.issuance_id].reason == "legal_hold"
    assert by_id[newest_cancelled.issuance_id].protected_as_latest is True
    assert by_id[old_completed.issuance_id].reason == (
        "completed_issuances_retained_by_default"
    )
    assert active.issuance_id not in by_id
    assert plan.candidate_count == 0

    candidate_plan = plan_custody_timestamp_issuance_retention(
        owner_id="alice",
        journal=Journal((old_cancelled, newest_cancelled)),
        now=1_000.0,
        minimum_age_seconds=10.0,
        retain_latest_per_authority_key=1,
    )
    candidates = [item for item in candidate_plan.items if item.retention_candidate]
    assert [item.issuance_id for item in candidates] == [old_cancelled.issuance_id]
    assert candidate_plan.deletion_performed is False


def test_timestamp_issuance_operations_cli_is_query_only(monkeypatch, capsys):
    journal = Journal((value("1", state="planned"),))
    observed = {}

    def read_only(path):
        observed["path"] = path
        return journal

    monkeypatch.setattr(cli, "ReadOnlyCustodyTimestampIssuanceJournal", read_only)
    assert cli.main(
        [
            "audit",
            "--owner-id",
            "alice",
            "--issuance-db-path",
            "/private/issuances.sqlite3",
        ]
    ) == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["mutation_performed"] is False
    assert audit["retry_performed"] is False
    assert audit["cancellation_performed"] is False
    assert audit["attestation_created"] is False
    assert "/private/issuances.sqlite3" not in json.dumps(audit)

    assert cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--issuance-db-path",
            "/private/issuances.sqlite3",
        ]
    ) == 0
    retention = json.loads(capsys.readouterr().out)
    assert retention["mutation_performed"] is False
    assert retention["deletion_performed"] is False
    assert retention["compaction_performed"] is False
    assert observed["path"] == "/private/issuances.sqlite3"
