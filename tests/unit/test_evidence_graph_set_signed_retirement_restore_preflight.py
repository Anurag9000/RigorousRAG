from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tools import evidence_graph_set_signed_retirement_restore_cli as cli
from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_readonly import (
    ReadOnlySignedPublicationRetirementJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_preflight import (
    preflight_signed_retirement_snapshot_restore,
)
from tools.evidence_graph_set_signed_retirement_snapshot import (
    build_signed_retirement_snapshot,
    export_signed_retirement_snapshot,
)


def planned(digit: str):
    return SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id=digit * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        now=1.0,
    )


def cancelled(digit: str):
    return replace(
        planned(digit),
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


def completed(digit: str):
    return replace(
        planned(digit),
        state="completed",
        phase="verified",
        final_pointer_set_id="2" * 64,
        verification_digest="6" * 64,
        updated_at=2.0,
        completed_at=2.0,
    )


class Values:
    def __init__(self, values=()):
        self.values = tuple(values)

    def list(self, **kwargs):
        return self.values[: kwargs["limit"]]


def snapshot(values):
    return build_signed_retirement_snapshot(
        owner_id="alice",
        journal=Values(values),
        now=10.0,
        limit=100,
    )


def target(tmp_path, values=()):
    journal = SignedPublicationRetirementJournal(tmp_path / "target.sqlite3")
    for value in values:
        journal.seed(value)
    return journal


def test_empty_target_is_only_restore_eligible_disposition(tmp_path):
    source = snapshot((cancelled("1"),))
    report = preflight_signed_retirement_snapshot_restore(
        snapshot=source,
        target_journal=target(tmp_path),
        now=20.0,
        limit=100,
    )

    assert report.disposition == "empty_target_restore_candidate"
    assert report.eligible_for_future_restore is True
    assert report.missing_count == 1
    assert report.target_mutation_performed is False
    assert report.restore_performed is False
    assert len(report.report_digest) == 64


def test_exact_terminal_target_is_idempotent_and_nonterminal_refuses(tmp_path):
    terminal = cancelled("1")
    exact = preflight_signed_retirement_snapshot_restore(
        snapshot=snapshot((terminal,)),
        target_journal=target(tmp_path / "exact", (terminal,)),
        now=20.0,
        limit=100,
    )
    assert exact.disposition == "already_restored_exactly"
    assert exact.exact_count == 1
    assert exact.eligible_for_future_restore is False

    active = planned("2")
    nonterminal = preflight_signed_retirement_snapshot_restore(
        snapshot=snapshot((active,)),
        target_journal=target(tmp_path / "active", (active,)),
        now=20.0,
        limit=100,
    )
    assert nonterminal.disposition == "target_nonterminal_refusal"
    assert nonterminal.nonterminal_target_count == 1


def test_state_collision_partial_and_additional_history_refuse(tmp_path):
    source_cancelled = cancelled("1")
    target_completed = completed("1")
    collision = preflight_signed_retirement_snapshot_restore(
        snapshot=snapshot((source_cancelled,)),
        target_journal=target(tmp_path / "collision", (target_completed,)),
        now=20.0,
        limit=100,
    )
    assert collision.disposition == "state_collision_refusal"
    assert collision.state_collision_count == 1

    first = cancelled("2")
    second = cancelled("3")
    partial = preflight_signed_retirement_snapshot_restore(
        snapshot=snapshot((first, second)),
        target_journal=target(tmp_path / "partial", (first,)),
        now=20.0,
        limit=100,
    )
    assert partial.disposition == "partial_restore_refusal"
    assert partial.missing_count == 1

    extra = cancelled("4")
    additional = preflight_signed_retirement_snapshot_restore(
        snapshot=snapshot((first,)),
        target_journal=target(tmp_path / "additional", (first, extra)),
        now=20.0,
        limit=100,
    )
    assert additional.disposition == "target_additional_history_refusal"
    assert additional.additional_count == 1


def test_read_only_target_requires_initialized_database_and_refuses_writes(tmp_path):
    uninitialized = tmp_path / "empty.sqlite3"
    uninitialized.write_bytes(b"")
    with pytest.raises(ValueError, match="not initialized"):
        ReadOnlySignedPublicationRetirementJournal(uninitialized)

    initialized = target(tmp_path / "initialized")
    read_only = ReadOnlySignedPublicationRetirementJournal(initialized.path)
    with read_only._connect() as connection:
        with pytest.raises(Exception):
            connection.execute(
                "DELETE FROM evidence_graph_set_signed_retirements"
            )


def test_restore_cli_is_byte_preserving_and_non_mutating(tmp_path, capsys):
    terminal = cancelled("1")
    snapshot_path = tmp_path / "snapshot.json"
    export_signed_retirement_snapshot(
        owner_id="alice",
        journal=Values((terminal,)),
        output_path=snapshot_path,
        now=10.0,
        limit=100,
    )
    target_journal = target(tmp_path / "target", (terminal,))
    before = target_journal.path.read_bytes()

    assert cli.main([
        "preflight",
        str(snapshot_path),
        "--target-db-path", str(target_journal.path),
        "--limit", "100",
    ]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload["disposition"] == "already_restored_exactly"
    assert payload["target_mutation_performed"] is False
    assert payload["restore_performed"] is False
    assert payload["journal_insert_performed"] is False
    assert payload["source_text_returned"] is False
    assert target_journal.path.read_bytes() == before
