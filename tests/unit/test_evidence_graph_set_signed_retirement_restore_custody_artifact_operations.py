from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools import (
    evidence_graph_set_signed_retirement_restore_custody_artifact_operations_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    deterministic_signed_retirement_restore_id,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_contracts import (
    RestoreCustodyArtifactAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_operations import (
    audit_restore_custody_artifacts,
    plan_restore_custody_artifact_retention,
)


def base(digit: str, *, target_digit: str = "a", now: float = 1.0, max_attempts=3):
    return RestoreCustodyArtifactAttempt.create(
        owner_id="alice",
        snapshot_digest=digit * 64,
        target_path_digest=target_digit * 64,
        backup_path_digest=(hex((int(digit, 16) + 1) % 16)[2:]) * 64,
        receipt_path_digest=(hex((int(digit, 16) + 2) % 16)[2:]) * 64,
        max_attempts=max_attempts,
        now=now,
    )


def running(value, *, attempts=1, expires=100.0):
    return replace(
        value,
        state="running",
        attempt_count=attempts,
        lease_owner="worker",
        lease_expires_at=expires,
        updated_at=2.0,
    )


def failed(value, *, attempts=1):
    return replace(
        value,
        state="failed",
        attempt_count=attempts,
        failure_type="Failure",
        updated_at=2.0,
    )


def completed(value, *, completed_at=2.0):
    return replace(
        value,
        state="completed",
        phase="verified",
        attempt_count=1,
        backup_sha256="b" * 64,
        backup_size_bytes=100,
        receipt_digest="c" * 64,
        receipt_actor_id="actor",
        receipt_binding_method="process_environment",
        receipt_binding_digest="d" * 64,
        disposition="paired",
        updated_at=completed_at,
        completed_at=completed_at,
    )


def orphan(value, disposition, *, completed_at=2.0):
    return replace(
        value,
        state="orphaned",
        phase="observed",
        attempt_count=1,
        disposition=disposition,
        updated_at=completed_at,
        completed_at=completed_at,
    )


def cancelled(value, *, completed_at=2.0):
    return replace(
        value,
        state="cancelled",
        updated_at=completed_at,
        completed_at=completed_at,
    )


class Journal:
    def __init__(self, values):
        self.values = tuple(values)
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        values = self.values
        if kwargs.get("state") is not None:
            values = tuple(value for value in values if value.state == kwargs["state"])
        return values[: kwargs["limit"]]


def test_artifact_audit_classifies_every_state_and_revalidates_digest():
    values = (
        base("1"),
        running(base("2"), attempts=1, expires=200.0),
        running(base("3"), attempts=1, expires=10.0),
        running(base("4", max_attempts=1), attempts=1, expires=10.0),
        failed(base("5"), attempts=1),
        failed(base("6", max_attempts=1), attempts=1),
        completed(base("7")),
        orphan(base("8"), "backup_without_receipt"),
        orphan(base("9"), "receipt_without_backup"),
        orphan(base("a"), "artifact_collision"),
        cancelled(base("b")),
    )
    report = audit_restore_custody_artifacts(
        owner_id="alice",
        journal=Journal(values),
        now=100.0,
        limit=100,
    )
    assert report.item_count == 11
    assert all(count == 1 for count in report.classification_counts.values())
    assert report.mutation_performed is False
    assert report.raw_path_returned is False
    with pytest.raises(ValueError, match="report_digest"):
        replace(report, report_digest="0" * 64)

    selected = deterministic_signed_retirement_restore_id(
        owner_id="alice",
        snapshot_digest="7" * 64,
        target_path_digest="a" * 64,
    )
    filtered = audit_restore_custody_artifacts(
        owner_id="alice",
        journal=Journal(values),
        restore_id=selected,
        now=100.0,
        limit=100,
    )
    assert filtered.item_count == 1
    assert filtered.items[0].classification == "completed_pair"


def test_artifact_audit_refuses_truncation_duplicates_and_bad_state():
    value = base("1")
    with pytest.raises(RuntimeError, match="bounded"):
        audit_restore_custody_artifacts(
            owner_id="alice",
            journal=Journal((value,)),
            now=10.0,
            limit=1,
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_restore_custody_artifacts(
            owner_id="alice",
            journal=Journal((value, value)),
            now=10.0,
            limit=10,
        )
    with pytest.raises(ValueError, match="unsupported"):
        audit_restore_custody_artifacts(
            owner_id="alice",
            journal=Journal(()),
            state="unknown",
            now=10.0,
            limit=10,
        )


def test_artifact_retention_never_selects_orphans_and_honors_holds_latest_defaults():
    old_completed = completed(base("1", target_digit="a"), completed_at=2.0)
    new_completed = completed(base("2", target_digit="a"), completed_at=3.0)
    held_cancelled = cancelled(base("3", target_digit="b"), completed_at=2.0)
    old_cancelled = cancelled(base("4", target_digit="c"), completed_at=2.0)
    new_cancelled = cancelled(base("6", target_digit="c"), completed_at=3.0)
    orphaned = orphan(
        base("5", target_digit="d"),
        "backup_without_receipt",
        completed_at=2.0,
    )
    held_restore = deterministic_signed_retirement_restore_id(
        owner_id="alice",
        snapshot_digest="3" * 64,
        target_path_digest="b" * 64,
    )
    journal = Journal(
        (
            old_completed,
            new_completed,
            held_cancelled,
            old_cancelled,
            new_cancelled,
            orphaned,
        )
    )

    default = plan_restore_custody_artifact_retention(
        owner_id="alice",
        journal=journal,
        now=1000.0,
        minimum_age_seconds=10.0,
        held_restore_ids=(held_restore,),
        limit=100,
    )
    reasons = {item.artifact_id: item.reason for item in default.items}
    assert default.candidate_count == 1
    assert reasons[old_completed.artifact_id] == "completed_pairs_retained_by_default"
    assert reasons[new_completed.artifact_id] == "latest_terminal_for_target"
    assert reasons[held_cancelled.artifact_id] == "legal_hold"
    assert reasons[old_cancelled.artifact_id] == "old_terminal_duplicate_candidate"
    assert reasons[new_cancelled.artifact_id] == "latest_terminal_for_target"
    assert reasons[orphaned.artifact_id] == "orphan_evidence_never_candidate"

    enabled = plan_restore_custody_artifact_retention(
        owner_id="alice",
        journal=journal,
        now=1000.0,
        minimum_age_seconds=10.0,
        include_completed=True,
        held_restore_ids=(held_restore,),
        limit=100,
    )
    candidates = {
        item.artifact_id for item in enabled.items if item.retention_candidate
    }
    assert candidates == {old_completed.artifact_id, old_cancelled.artifact_id}
    assert orphaned.artifact_id not in candidates
    assert enabled.deletion_performed is False
    with pytest.raises(ValueError, match="plan_digest"):
        replace(enabled, plan_digest="0" * 64)


def test_artifact_operations_cli_integrates_durable_holds_without_mutation(
    monkeypatch,
    capsys,
):
    held = cancelled(base("1", target_digit="a"), completed_at=2.0)
    free = cancelled(base("2", target_digit="b"), completed_at=2.0)
    newer_free = cancelled(base("3", target_digit="b"), completed_at=3.0)
    journal = Journal((held, free, newer_free))
    monkeypatch.setattr(
        cli,
        "ReadOnlyRestoreCustodyArtifactJournal",
        lambda _path: journal,
    )
    held_restore = deterministic_signed_retirement_restore_id(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="a" * 64,
    )

    class Holds:
        def __init__(self, path):
            assert path == "holds.sqlite3"

        def active_restore_ids(self, **kwargs):
            assert kwargs["owner_id"] == "alice"
            return frozenset({held_restore})

    monkeypatch.setattr(cli, "ReadOnlySignedRetirementRestoreHoldStore", Holds)
    assert cli.main(
        [
            "retention-plan",
            "--owner-id",
            "alice",
            "--artifact-db-path",
            "artifacts.sqlite3",
            "--durable-hold-db-path",
            "holds.sqlite3",
            "--minimum-age-seconds",
            "1",
            "--limit",
            "100",
        ]
    ) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["durable_restore_hold_count"] == 1
    assert payload["candidate_count"] == 1
    assert payload["artifact_mutation_performed"] is False
    assert payload["artifact_deletion_performed"] is False
    assert payload["artifact_overwrite_performed"] is False
    assert payload["raw_path_returned"] is False

    with pytest.raises(SystemExit):
        cli.main(["delete"])
