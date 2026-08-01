from dataclasses import replace

import pytest

from tools.migration_rollback_artifact import capture_rollback_payload
from tools.migration_rollback_reconstruction import reconstruct_rollback_snapshots
from tests.unit.test_migration_rollback_artifact import aligned_preflight


def test_reconstructs_public_vector_sparse_and_generation_types():
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    result = reconstruct_rollback_snapshots(preflight, payload)
    assert result.vector.ids == ("v1",)
    assert result.vector.documents == ("one",)
    assert result.sparse.fields[0].field_id == "f1"
    assert result.sparse.fields[0].text == "one"
    assert result.generation.sequence == preflight.source_sequence
    assert result.generation.content_sha256 == preflight.source_content_sha256


def test_reconstruction_is_in_memory_and_does_not_mutate_payload():
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    before = repr(payload)
    first = reconstruct_rollback_snapshots(preflight, payload)
    second = reconstruct_rollback_snapshots(preflight, payload)
    assert first == second
    assert repr(payload) == before


def test_payload_or_preflight_identity_tamper_is_refused():
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    payload["sparse_snapshot"]["fields"][0]["text"] = "changed"
    with pytest.raises(RuntimeError, match="digest"):
        reconstruct_rollback_snapshots(preflight, payload)

    payload = capture_rollback_payload(preflight, snapshot)
    changed = replace(preflight, source_sequence=preflight.source_sequence + 1)
    with pytest.raises(RuntimeError):
        reconstruct_rollback_snapshots(changed, payload)


def test_generation_count_or_profile_inconsistency_is_refused():
    preflight, snapshot = aligned_preflight()
    payload = capture_rollback_payload(preflight, snapshot)
    payload["generation"]["vector_rows"] = 2
    with pytest.raises(RuntimeError):
        reconstruct_rollback_snapshots(preflight, payload)
