from __future__ import annotations

import os

import pytest

from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
    deterministic_signed_retirement_id,
)
from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)


def attempt(*, now=1.0, max_attempts=3):
    return SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id="1" * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
        max_attempts=max_attempts,
        now=now,
    )


def test_retirement_identity_is_deterministic_and_scope_bound():
    first = attempt()
    second = attempt(now=9.0)
    assert first.retirement_id == second.retirement_id
    assert first.retirement_id == deterministic_signed_retirement_id(
        owner_id="alice",
        publication_operation_id="1" * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="5" * 64,
    )
    changed = SignedPublicationRetirementAttempt.create(
        owner_id="alice",
        publication_operation_id="1" * 64,
        graph_set_key="review",
        signed_candidate_set_id="2" * 64,
        signed_candidate_set_digest="3" * 64,
        authorization_candidate_set_id="4" * 64,
        signed_authority_digest="6" * 64,
        now=1.0,
    )
    assert changed.retirement_id != first.retirement_id


def test_journal_runs_monotonic_retirement_lifecycle(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    seeded = journal.seed(attempt())
    claimed = journal.claim(
        seeded.retirement_id,
        worker_id="worker",
        lease_seconds=30,
        now=2.0,
    )
    assert claimed.state == "running"
    assert claimed.attempt_count == 1
    assert claimed.lease_expires_at == 32.0

    intent = journal.record_pointer_restore_intent(
        seeded.retirement_id, worker_id="worker", now=3.0
    )
    assert intent.phase == "pointer_restore_intent"
    safe = journal.record_pointer_safe(
        seeded.retirement_id, worker_id="worker", now=4.0
    )
    assert safe.phase == "pointer_safe"
    retired = journal.record_authorization_retired(
        seeded.retirement_id,
        worker_id="worker",
        final_pointer_set_id="2" * 64,
        now=5.0,
    )
    assert retired.phase == "authorization_retired"
    completed = journal.complete(
        seeded.retirement_id,
        worker_id="worker",
        verification_digest="6" * 64,
        final_pointer_set_id="2" * 64,
        now=6.0,
    )
    assert completed.state == "completed"
    assert completed.phase == "verified"
    assert completed.completed_at == 6.0
    assert completed.lease_owner is None
    with pytest.raises(RuntimeError, match="not claimable"):
        journal.claim(
            seeded.retirement_id,
            worker_id="other",
            lease_seconds=30,
            now=40.0,
        )


def test_expired_claim_is_recovered_without_losing_phase(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    seeded = journal.seed(attempt())
    journal.claim(seeded.retirement_id, worker_id="one", lease_seconds=2, now=2.0)
    journal.record_pointer_restore_intent(
        seeded.retirement_id, worker_id="one", now=3.0
    )

    recovered = journal.claim(
        seeded.retirement_id,
        worker_id="two",
        lease_seconds=10,
        now=5.0,
    )
    assert recovered.phase == "pointer_restore_intent"
    assert recovered.attempt_count == 2
    assert recovered.lease_owner == "two"


def test_failure_retry_preserves_recovery_phase_and_attempt_ceiling(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    seeded = journal.seed(attempt(max_attempts=2))
    journal.claim(seeded.retirement_id, worker_id="worker", now=2.0)
    journal.record_pointer_restore_intent(
        seeded.retirement_id, worker_id="worker", now=3.0
    )
    failed = journal.fail(
        seeded.retirement_id,
        worker_id="worker",
        failure_type="PointerFailure",
        now=4.0,
    )
    assert failed.state == "failed"
    assert failed.phase == "pointer_restore_intent"

    retried = journal.retry(
        seeded.retirement_id,
        owner_id="alice",
        confirm_retirement_id=seeded.retirement_id,
        now=5.0,
    )
    assert retried.state == "planned"
    assert retried.phase == "pointer_restore_intent"
    journal.claim(retried.retirement_id, worker_id="worker", now=6.0)
    journal.fail(
        retried.retirement_id,
        worker_id="worker",
        failure_type="Again",
        now=7.0,
    )
    with pytest.raises(RuntimeError, match="attempt ceiling"):
        journal.retry(
            retried.retirement_id,
            owner_id="alice",
            confirm_retirement_id=retried.retirement_id,
            now=8.0,
        )


def test_cancel_is_exact_and_only_before_pointer_intent(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    seeded = journal.seed(attempt())
    with pytest.raises(ValueError, match="confirmation"):
        journal.cancel(
            seeded.retirement_id,
            owner_id="alice",
            confirm_retirement_id="f" * 64,
            now=2.0,
        )
    cancelled = journal.cancel(
        seeded.retirement_id,
        owner_id="alice",
        confirm_retirement_id=seeded.retirement_id,
        now=2.0,
    )
    assert cancelled.state == "cancelled"

    second = journal.seed(
        SignedPublicationRetirementAttempt.create(
            owner_id="alice",
            publication_operation_id="a" * 64,
            graph_set_key="review",
            signed_candidate_set_id="2" * 64,
            signed_candidate_set_digest="3" * 64,
            authorization_candidate_set_id="4" * 64,
            signed_authority_digest="5" * 64,
            now=1.0,
        )
    )
    journal.claim(second.retirement_id, worker_id="worker", now=2.0)
    journal.record_pointer_restore_intent(
        second.retirement_id, worker_id="worker", now=3.0
    )
    journal.fail(
        second.retirement_id,
        worker_id="worker",
        failure_type="Stopped",
        now=4.0,
    )
    with pytest.raises(RuntimeError, match="unstarted"):
        journal.cancel(
            second.retirement_id,
            owner_id="alice",
            confirm_retirement_id=second.retirement_id,
            now=5.0,
        )


def test_seed_is_idempotent_but_detects_scope_collision(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    value = attempt()
    assert journal.seed(value) == value
    assert journal.seed(attempt(now=9.0)).retirement_id == value.retirement_id

    with journal._lock, journal._connect() as connection:
        connection.execute(
            "UPDATE evidence_graph_set_signed_retirements "
            "SET signed_candidate_set_digest=? WHERE retirement_id=?",
            ("f" * 64, value.retirement_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        journal.get(value.retirement_id)


def test_database_identity_and_redirect_changes_fail_closed(tmp_path):
    path = tmp_path / "retirements.sqlite3"
    journal = SignedPublicationRetirementJournal(path)
    journal.seed(attempt())
    replacement = tmp_path / "replacement.sqlite3"
    replacement.write_bytes(path.read_bytes())
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="identity changed"):
        journal.get(attempt().retirement_id)


def test_queue_and_list_are_owner_scoped_and_bounded(tmp_path):
    journal = SignedPublicationRetirementJournal(tmp_path / "retirements.sqlite3")
    first = journal.seed(attempt(now=1.0))
    second = journal.seed(
        SignedPublicationRetirementAttempt.create(
            owner_id="alice",
            publication_operation_id="a" * 64,
            graph_set_key="other",
            signed_candidate_set_id="b" * 64,
            signed_candidate_set_digest="c" * 64,
            authorization_candidate_set_id=None,
            signed_authority_digest="d" * 64,
            now=2.0,
        )
    )
    assert journal.next_claimable_id(owner_id="alice", now=3.0) == first.retirement_id
    values = journal.list(owner_id="alice", limit=10)
    assert {value.retirement_id for value in values} == {
        first.retirement_id,
        second.retirement_id,
    }
    with pytest.raises(ValueError, match="between"):
        journal.list(owner_id="alice", limit=0)
