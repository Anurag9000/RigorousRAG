import hashlib

import pytest

from tools.migration_target_population import (
    TargetPopulationIdentity,
    TargetPopulationJournal,
    reconcile_target_population,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def identity(operation: str = "op") -> TargetPopulationIdentity:
    return TargetPopulationIdentity(
        operation_id=digest(operation),
        owner_id="alice",
        doc_id="doc-1",
        target_collection_id=digest("collection"),
        target_profile_fingerprint=digest("profile"),
        content_sha256=digest("content"),
        target_artifact_digest=digest("artifact"),
        expected_vector_rows=3,
    )


def test_intent_is_idempotent_and_identity_collision_fails_closed(tmp_path):
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    selected = identity()
    first = journal.ensure_intent(selected, now=1.0)
    second = journal.ensure_intent(selected, now=2.0)
    assert first == second
    assert first.state == "planned"
    conflicting = TargetPopulationIdentity(
        operation_id=selected.operation_id,
        owner_id=selected.owner_id,
        doc_id=selected.doc_id,
        target_collection_id=digest("other-collection"),
        target_profile_fingerprint=selected.target_profile_fingerprint,
        content_sha256=selected.content_sha256,
        target_artifact_digest=selected.target_artifact_digest,
        expected_vector_rows=selected.expected_vector_rows,
    )
    with pytest.raises(RuntimeError, match="identity collision"):
        journal.ensure_intent(conflicting, now=3.0)


def test_expired_takeover_increments_fence_and_stale_same_worker_is_rejected(tmp_path):
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    selected = identity()
    journal.ensure_intent(selected, now=1.0)
    first = journal.claim(
        selected.operation_id,
        worker_id="worker-a",
        now=2.0,
        lease_seconds=5.0,
    )
    with pytest.raises(RuntimeError, match="live executor"):
        journal.claim(
            selected.operation_id,
            worker_id="worker-b",
            now=3.0,
            lease_seconds=5.0,
        )
    second = journal.claim(
        selected.operation_id,
        worker_id="worker-a",
        now=8.0,
        lease_seconds=5.0,
    )
    assert second.fencing_token == first.fencing_token + 1
    with pytest.raises(RuntimeError, match="stale or fenced"):
        journal.assert_claim(first, now=8.5)
    journal.assert_claim(second, now=8.5)
    journal.release(second, now=9.0)
    assert journal.get(selected.operation_id).attempt == 2


def test_population_receipts_and_reconciliation_are_deterministic(tmp_path):
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    selected = identity()
    planned = journal.ensure_intent(selected, now=1.0)
    missing = reconcile_target_population(
        selected,
        planned,
        observed_rows=0,
        exact_population_match=False,
        route_collection_id=None,
    )
    assert missing.state == "missing_with_intent"
    populated = journal.mark_populated(
        selected,
        row_digest=digest("rows"),
        now=2.0,
    )
    assert populated.state == "populated"
    visible = journal.mark_visible(
        selected,
        row_digest=digest("rows"),
        route_digest=digest("route"),
        generation_sequence=9,
        now=3.0,
    )
    assert visible.state == "visible"
    assert visible.population_digest == populated.population_digest
    again = journal.mark_visible(
        selected,
        row_digest=digest("rows"),
        route_digest=digest("route"),
        generation_sequence=9,
        now=4.0,
    )
    assert again.receipt_digest == visible.receipt_digest
    reconciled = reconcile_target_population(
        selected,
        again,
        observed_rows=3,
        exact_population_match=True,
        route_collection_id=selected.target_collection_id,
    )
    assert reconciled.state == "visible"
    assert reconciled == reconcile_target_population(
        selected,
        again,
        observed_rows=3,
        exact_population_match=True,
        route_collection_id=selected.target_collection_id,
    )


def test_orphan_partial_and_authority_conflict_are_distinct():
    selected = identity()
    orphan = reconcile_target_population(
        selected,
        None,
        observed_rows=3,
        exact_population_match=True,
        route_collection_id=None,
    )
    assert orphan.state == "orphan"
    journal_record = None
    no_intent = reconcile_target_population(
        selected,
        journal_record,
        observed_rows=0,
        exact_population_match=False,
        route_collection_id=None,
    )
    assert no_intent.state == "missing_without_intent"
    conflict = reconcile_target_population(
        selected,
        None,
        observed_rows=2,
        exact_population_match=False,
        route_collection_id=selected.target_collection_id,
    )
    assert conflict.state == "authority_conflict"


def test_abort_receipt_refuses_revival(tmp_path):
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    selected = identity()
    journal.ensure_intent(selected, now=1.0)
    aborted = journal.mark_aborted(selected, now=2.0)
    assert aborted.state == "aborted"
    with pytest.raises(RuntimeError, match="cannot be revived"):
        journal.mark_populated(
            selected,
            row_digest=digest("rows"),
            now=3.0,
        )
