import pytest

from tools.migration_cutover_durable_blue_green import (
    DurableBlueGreenCutoverBackendAdapter,
)
from tools.migration_cutover_saga import execute_cutover_saga
from tools.migration_target_population import TargetPopulationJournal
from tests.unit.test_migration_cutover_blue_green import Fixture


def durable_adapter(fixture, journal, *, worker_id="worker-a"):
    return DurableBlueGreenCutoverBackendAdapter(
        registry=fixture.registry,
        provider=fixture.provider,
        sparse=fixture.sparse,
        generations=fixture.generations,
        shadow=fixture.shadow,
        population_journal=journal,
        worker_id=worker_id,
        lease_seconds=100.0,
        clock=lambda: 10.0,
    )


def test_durable_blue_green_success_seals_visible_receipt(tmp_path):
    fixture = Fixture(tmp_path)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    result = execute_cutover_saga(
        fixture.operation,
        durable_adapter(fixture, journal),
    )
    assert result.outcome == "published"
    record = journal.get(fixture.operation.operation_id)
    assert record is not None
    assert record.state == "visible"
    assert record.population_digest is not None
    assert record.receipt_digest is not None
    assert record.attempt == 1


def test_crash_after_hidden_write_is_resumed_without_orphaning_rows(tmp_path):
    fixture = Fixture(tmp_path)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    first = durable_adapter(fixture, journal, worker_id="worker-a")
    with first.exclusive_lock(fixture.operation):
        first.current_identity(fixture.operation)
        publication = first.write_hidden_target(fixture.operation)
    record = journal.get(fixture.operation.operation_id)
    assert record is not None and record.state == "planned"
    target_rows = dict(fixture.provider.collection(fixture.target_spec).rows)
    assert target_rows

    second = durable_adapter(fixture, journal, worker_id="worker-b")
    result = execute_cutover_saga(fixture.operation, second)
    assert result.outcome == "published"
    assert journal.get(fixture.operation.operation_id).state == "visible"
    assert fixture.provider.collection(fixture.target_spec).rows == target_rows
    assert publication.publication_id == result.publication_id


def test_unexplained_preexisting_target_rows_fail_closed_without_deletion(tmp_path):
    fixture = Fixture(tmp_path)
    target = fixture.provider.collection(fixture.target_spec)
    target.upsert(
        ids=["orphan"],
        documents=["untracked"],
        metadatas=[{"owner_id": fixture.owner, "doc_id": fixture.doc_id}],
        embeddings=[[0.1, 0.2, 0.3]],
    )
    before = dict(target.rows)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    result = execute_cutover_saga(
        fixture.operation,
        durable_adapter(fixture, journal),
    )
    assert result.outcome == "aborted"
    assert result.phases == ()
    assert target.rows == before
    assert journal.get(fixture.operation.operation_id) is None


def test_hidden_fault_discards_target_and_seals_abort_receipt(tmp_path):
    fixture = Fixture(tmp_path)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")

    def fault(phase):
        if phase == "hidden_target_written":
            raise RuntimeError("private crash")

    result = execute_cutover_saga(
        fixture.operation,
        durable_adapter(fixture, journal),
        fault_hook=fault,
    )
    assert result.outcome == "aborted"
    record = journal.get(fixture.operation.operation_id)
    assert record is not None and record.state == "aborted"
    assert record.receipt_digest is not None
    assert fixture.provider.collection(fixture.target_spec).rows == {}


def test_visible_receipt_can_be_finalized_after_post_route_crash(tmp_path):
    fixture = Fixture(tmp_path)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    first = durable_adapter(fixture, journal, worker_id="worker-a")
    with first.exclusive_lock(fixture.operation):
        first.current_identity(fixture.operation)
        publication = first.write_hidden_target(fixture.operation)
        first.validate_hidden_target(fixture.operation, publication)
        first.commit_visibility(fixture.operation, publication)
    assert journal.get(fixture.operation.operation_id).state == "populated"

    second = durable_adapter(fixture, journal, worker_id="worker-b")
    reconciliation = second.reconcile_population(fixture.operation)
    assert reconciliation.state == "visible"
    with second.exclusive_lock(fixture.operation):
        second.finalize_visible_recovery(fixture.operation)
    assert journal.get(fixture.operation.operation_id).state == "visible"


def test_postvisibility_rollback_records_terminal_rolled_back_receipt(tmp_path):
    fixture = Fixture(tmp_path)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")

    def fault(phase):
        if phase == "visibility_committed":
            raise RuntimeError("private visible failure")

    result = execute_cutover_saga(
        fixture.operation,
        durable_adapter(fixture, journal),
        fault_hook=fault,
    )
    assert result.outcome == "rolled_back"
    record = journal.get(fixture.operation.operation_id)
    assert record is not None and record.state == "rolled_back"
    assert record.population_digest is not None
    assert record.receipt_digest is not None


def test_stale_executor_token_cannot_mutate_after_takeover(tmp_path):
    fixture = Fixture(tmp_path)
    journal = TargetPopulationJournal(tmp_path / "population.sqlite3")
    adapter = durable_adapter(fixture, journal, worker_id="shared-name")
    identity = adapter._population_identity(fixture.operation)
    journal.ensure_intent(identity, now=1.0)
    stale = journal.claim(
        fixture.operation.operation_id,
        worker_id="shared-name",
        now=2.0,
        lease_seconds=2.0,
    )
    current = journal.claim(
        fixture.operation.operation_id,
        worker_id="shared-name",
        now=5.0,
        lease_seconds=10.0,
    )
    assert current.fencing_token > stale.fencing_token
    with pytest.raises(RuntimeError, match="stale or fenced"):
        journal.assert_claim(stale, now=6.0)
