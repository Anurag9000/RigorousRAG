from dataclasses import replace

import pytest

from tools.migration_cutover_saga import (
    BackendStateIdentity,
    CutoverRecoveryError,
    execute_cutover_saga,
)
from tests.unit.test_migration_cutover_saga import FakeAdapter, ready_operation


def test_drift_rejection_verifies_observed_source_remained_unchanged():
    operation = ready_operation()
    adapter = FakeAdapter(operation)
    prepared = BackendStateIdentity.from_preparation(operation.preparation)
    reads = iter(
        (
            replace(prepared, source_sequence=prepared.source_sequence + 1),
            replace(prepared, source_sequence=prepared.source_sequence + 2),
        )
    )

    def changing_identity(current_operation):
        adapter._call("current_identity")
        return next(reads)

    adapter.current_identity = changing_identity
    with pytest.raises(CutoverRecoveryError) as captured:
        execute_cutover_saga(operation, adapter)
    assert captured.value.phase == "lock_acquired"
    assert captured.value.failure_type == "RuntimeError"
    assert captured.value.recovery_failure_type == "RuntimeError"
    assert "write_hidden_target" not in adapter.calls
