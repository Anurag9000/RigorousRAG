from __future__ import annotations

import json

from tools import (
    evidence_graph_set_signed_retirement_restore_hold_permit_recovery_cli as cli,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery_contracts import (
    RestoreHoldPermitRecoveryReceipt,
)


def receipt():
    return RestoreHoldPermitRecoveryReceipt.create(
        owner_id="alice",
        restore_id="1" * 64,
        hold_id="2" * 64,
        original_permit_digest="3" * 64,
        released_permit_digest="4" * 64,
        classification="abandoned_without_hold_quarantined",
        quarantine_hold_id="5" * 64,
        quarantine_hold_digest="6" * 64,
        actor_id="operator",
        actor_binding_method="process_environment",
        actor_binding_digest="7" * 64,
        recovered_at=10.0,
    )


def test_bad_confirmation_is_rejected_before_store_loading(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: (_ for _ in ()).throw(
            AssertionError("stores must not load")
        ),
    )
    assert cli.main(
        [
            "recover",
            "2" * 64,
            "--owner-id",
            "alice",
            "--confirm-hold-id",
            "8" * 64,
            "--confirm-permit-digest",
            "3" * 64,
        ]
    ) == 2
    assert json.loads(capsys.readouterr().err) == {
        "error": "invalid_or_unavailable"
    }


def test_recover_is_actor_bound_and_reports_quarantine(monkeypatch, capsys):
    value = receipt()
    journal = object()
    hold_store = object()
    actor = object()
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: journal,
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_hold_store",
        lambda: hold_store,
    )
    monkeypatch.setattr(cli, "load_relation_review_actor", lambda: actor)
    monkeypatch.setattr(
        cli,
        "require_relation_review_actor",
        lambda requested, binding: binding,
    )
    observed = {}

    def recover(**kwargs):
        observed.update(kwargs)
        return value, True

    monkeypatch.setattr(
        cli,
        "recover_abandoned_hold_placement_permit",
        recover,
    )
    assert cli.main(
        [
            "recover",
            value.hold_id,
            "--owner-id",
            "alice",
            "--confirm-hold-id",
            value.hold_id,
            "--confirm-permit-digest",
            value.original_permit_digest,
            "--actor-id",
            "operator",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert observed["restore_journal"] is journal
    assert observed["hold_store"] is hold_store
    assert observed["actor"] is actor
    assert payload["permit_mutation_performed"] is True
    assert payload["quarantine_hold_mutation_performed"] is True
    assert payload["restore_record_mutation_performed"] is False
    assert payload["deletion_performed"] is False
    assert payload["source_text_returned"] is False
    assert payload["raw_paths_returned"] is False


def test_status_and_list_are_read_only_and_do_not_load_hold_store(
    monkeypatch,
    capsys,
):
    value = receipt()
    journal = object()
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_journal",
        lambda: journal,
    )
    monkeypatch.setattr(
        cli,
        "get_signed_retirement_restore_hold_store",
        lambda: (_ for _ in ()).throw(
            AssertionError("read-only commands must not load hold store")
        ),
    )
    monkeypatch.setattr(
        cli,
        "get_hold_permit_recovery",
        lambda selected, recovery_id: value,
    )
    monkeypatch.setattr(
        cli,
        "list_hold_permit_recoveries",
        lambda selected, owner_id, limit: (value,),
    )

    assert cli.main(["status", value.recovery_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["mutation_performed"] is False

    assert cli.main(["list", "--owner-id", "alice"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["count"] == 1
    assert listing["mutation_performed"] is False
    assert listing["raw_paths_returned"] is False
