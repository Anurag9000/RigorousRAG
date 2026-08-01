from dataclasses import replace

import pytest

from tools.migration_rollback_artifact import capture_rollback_payload
from tools.migration_rollback_reconstruction import reconstruct_rollback_snapshots
from tools.migration_rollback_staging import (
    InMemoryRollbackStagingStore,
    staging_identity,
    verify_in_isolated_staging,
)
from tests.unit.test_migration_rollback_artifact import aligned_preflight


def reconstructed():
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    return preflight, reconstruct_rollback_snapshots(preflight, payload)


def test_isolated_staging_resnapshots_exact_preflight_identities():
    preflight, rollback = reconstructed()
    store = InMemoryRollbackStagingStore()
    result = verify_in_isolated_staging(preflight, rollback, store=store, now=10)
    assert result.staging_id == staging_identity(preflight)
    assert result.vector_snapshot_digest == preflight.vector_snapshot_digest
    assert result.sparse_snapshot_digest == preflight.sparse_snapshot_digest
    assert result.vector_rows == preflight.source_vector_rows
    assert result.sparse_fields == preflight.source_sparse_fields
    assert store.count() == 1
    assert len(result.verification_digest) == 64


def test_stage_copies_nested_metadata_and_is_idempotent():
    preflight, rollback = reconstructed()
    store = InMemoryRollbackStagingStore()
    identity = staging_identity(preflight)
    first = store.stage(identity, rollback)
    second = store.stage(identity, rollback)
    assert first == second
    assert first is not second
    assert first.vector.metadatas[0] is not rollback.vector.metadatas[0]
    assert first.sparse.metadata is not rollback.sparse.metadata
    assert first.sparse.fields[0].metadata is not rollback.sparse.fields[0].metadata


def test_staging_identity_collision_with_different_snapshot_is_refused():
    preflight, rollback = reconstructed()
    store = InMemoryRollbackStagingStore()
    identity = staging_identity(preflight)
    store.stage(identity, rollback)
    changed_generation = replace(
        rollback.generation,
        committed_at=rollback.generation.committed_at + 1,
    )
    changed = rollback.__class__(rollback.vector, rollback.sparse, changed_generation)
    with pytest.raises(RuntimeError, match="different snapshots"):
        store.stage(identity, changed)


def test_changed_preflight_or_snapshot_is_refused():
    preflight, rollback = reconstructed()
    changed = replace(preflight, source_sequence=preflight.source_sequence + 1)
    with pytest.raises(RuntimeError):
        verify_in_isolated_staging(changed, rollback)

    changed_generation = replace(
        rollback.generation,
        content_sha256="9" * 64,
    )
    changed_rollback = rollback.__class__(
        rollback.vector,
        rollback.sparse,
        changed_generation,
    )
    with pytest.raises(RuntimeError, match="does not match"):
        verify_in_isolated_staging(preflight, changed_rollback)


def test_staging_store_limit_and_missing_snapshot_are_bounded():
    first_preflight, first = reconstructed()
    store = InMemoryRollbackStagingStore(maximum_entries=1)
    store.stage(staging_identity(first_preflight), first)
    second_preflight = replace(
        first_preflight,
        task_id="9" * 64,
        promotion_report_digest="8" * 64,
    )
    with pytest.raises(RuntimeError, match="entry limit"):
        store.stage(staging_identity(second_preflight), first)
    with pytest.raises(KeyError):
        store.snapshot("7" * 64)
    assert store.remove(staging_identity(first_preflight)) is True
    assert store.remove(staging_identity(first_preflight)) is False
