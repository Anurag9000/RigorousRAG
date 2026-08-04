from __future__ import annotations

from dataclasses import replace

import pytest

from tools.evidence_graph_relation_actor import ReviewActorBinding
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_boundary import (
    GovernedSignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_integrity import (
    IntegritySignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery import (
    get_hold_permit_recovery,
    list_hold_permit_recoveries,
    recover_abandoned_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_permits import (
    acquire_hold_placement_permit,
    active_hold_placement_permit,
)
from tools.evidence_graph_set_signed_retirement_restore_holds import (
    deterministic_restore_hold_id,
)
from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)


def actor():
    return ReviewActorBinding.create(
        actor_id="operator",
        binding_method="process_environment",
        loaded_at=1.0,
    )


def cancelled_restore():
    value = SignedRetirementRestoreAttempt.create(
        owner_id="alice",
        snapshot_digest="1" * 64,
        target_path_digest="2" * 64,
        snapshot_record_count=1,
        now=1.0,
    )
    return replace(
        value,
        state="cancelled",
        updated_at=2.0,
        completed_at=2.0,
    )


def setup(tmp_path):
    restore = cancelled_restore()
    journal = SignedRetirementRestoreJournal(tmp_path / "restores.sqlite3")
    journal.seed(restore)
    holds = GovernedSignedRetirementRestoreHoldStore(
        tmp_path / "holds.sqlite3"
    )
    return restore, journal, holds


def stale_permit(journal, restore, hold_key):
    hold_id = deterministic_restore_hold_id(
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key=hold_key,
    )
    permit_digest = acquire_hold_placement_permit(
        journal,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_id=hold_id,
        now=4.0,
    )
    return hold_id, permit_digest


def recover(journal, holds, hold_id, permit_digest, *, now=100.0):
    return recover_abandoned_hold_placement_permit(
        restore_journal=journal,
        hold_store=holds,
        owner_id="alice",
        hold_id=hold_id,
        confirm_hold_id=hold_id,
        confirm_permit_digest=permit_digest,
        actor=actor(),
        minimum_age_seconds=60,
        now=now,
    )


def test_missing_hold_is_quarantined_before_permit_release_and_replays(tmp_path):
    restore, journal, holds = setup(tmp_path)
    hold_id, permit_digest = stale_permit(journal, restore, "case-a")

    receipt, changed = recover(
        journal,
        holds,
        hold_id,
        permit_digest,
    )

    assert changed is True
    assert receipt.classification == "abandoned_without_hold_quarantined"
    assert receipt.quarantine_hold_id is not None
    quarantine = holds.get(receipt.quarantine_hold_id)
    assert quarantine.status == "active"
    assert quarantine.restore_id == restore.restore_id
    assert quarantine.reason_code == "abandoned_hold_placement_permit"
    with journal._lock, journal._connect() as connection:
        assert active_hold_placement_permit(
            connection,
            restore.restore_id,
        ) is None

    replay, changed = recover(
        journal,
        holds,
        hold_id,
        permit_digest,
        now=101.0,
    )
    assert changed is False
    assert replay == receipt
    assert get_hold_permit_recovery(journal, receipt.recovery_id) == receipt
    assert list_hold_permit_recoveries(
        journal,
        owner_id="alice",
    ) == (receipt,)


def test_active_original_hold_requires_exact_hold_replay(tmp_path):
    restore, journal, holds = setup(tmp_path)
    hold_id, permit_digest = stale_permit(journal, restore, "case-a")
    IntegritySignedRetirementRestoreHoldStore.place(
        holds,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-a",
        reason_code="litigation",
        actor=actor(),
        restore_journal=journal,
        now=5.0,
    )

    with pytest.raises(RuntimeError, match="exact hold replay"):
        recover(journal, holds, hold_id, permit_digest)


def test_released_original_hold_allows_cleanup_without_quarantine(tmp_path):
    restore, journal, holds = setup(tmp_path)
    hold_id, permit_digest = stale_permit(journal, restore, "case-a")
    IntegritySignedRetirementRestoreHoldStore.place(
        holds,
        owner_id="alice",
        restore_id=restore.restore_id,
        hold_key="case-a",
        reason_code="litigation",
        actor=actor(),
        restore_journal=journal,
        now=5.0,
    )
    holds.release(
        hold_id,
        owner_id="alice",
        confirm_hold_id=hold_id,
        actor=actor(),
        now=6.0,
    )

    receipt, changed = recover(
        journal,
        holds,
        hold_id,
        permit_digest,
    )
    assert changed is True
    assert receipt.classification == "released_hold_cleanup"
    assert receipt.quarantine_hold_id is None
    assert receipt.quarantine_hold_digest is None


def test_age_confirmation_and_active_deletion_marker_fail_closed(
    tmp_path,
    monkeypatch,
):
    restore, journal, holds = setup(tmp_path)
    hold_id, permit_digest = stale_permit(journal, restore, "case-a")

    with pytest.raises(ValueError, match="confirmation"):
        recover_abandoned_hold_placement_permit(
            restore_journal=journal,
            hold_store=holds,
            owner_id="alice",
            hold_id=hold_id,
            confirm_hold_id="f" * 64,
            confirm_permit_digest=permit_digest,
            actor=actor(),
            minimum_age_seconds=60,
            now=100.0,
        )
    with pytest.raises(RuntimeError, match="recovery age"):
        recover(journal, holds, hold_id, permit_digest, now=30.0)

    monkeypatch.setattr(
        "tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery._marker_row",
        lambda connection, restore_id: {"state": "active"},
    )
    with pytest.raises(RuntimeError, match="deletion control"):
        recover(journal, holds, hold_id, permit_digest)


def test_recovery_receipt_tampering_fails_closed(tmp_path):
    restore, journal, holds = setup(tmp_path)
    hold_id, permit_digest = stale_permit(journal, restore, "case-a")
    receipt, _changed = recover(journal, holds, hold_id, permit_digest)

    with journal._lock, journal._connect() as connection:
        connection.execute(
            "UPDATE signed_retirement_restore_hold_permit_recoveries "
            "SET actor_id='tampered' WHERE recovery_id=?",
            (receipt.recovery_id,),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        get_hold_permit_recovery(journal, receipt.recovery_id)
