from contextlib import contextmanager
from dataclasses import replace

import pytest

from tools.migration_cutover_control import CutoverOperation
from tools.migration_cutover_saga import (
    BackendStateIdentity,
    CutoverRecoveryError,
    TargetPublication,
    execute_cutover_saga,
)
from tests.unit.test_migration_cutover_journal import preparation


def ready_operation():
    value = preparation()
    return CutoverOperation(
        operation_id=value.operation_id,
        preparation=value,
        state="ready",
        attempt=1,
        created_at=1,
        updated_at=2,
    )


class FakeAdapter:
    def __init__(self, operation, *, fail_at=None, source_drift=False, wrong_target=False):
        self.operation = operation
        self.fail_at = fail_at
        self.source_drift = source_drift
        self.wrong_target = wrong_target
        self.calls = []
        self.locked = False
        self.hidden = None
        self.visible = False
        self.rolled_back = False

    def _call(self, name):
        assert self.locked, f"{name} ran outside exclusive lock"
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"private-{name}-details")

    @contextmanager
    def exclusive_lock(self, operation):
        self.calls.append("lock_enter")
        if self.fail_at == "lock_enter":
            raise RuntimeError("private-lock-details")
        self.locked = True
        try:
            yield
        finally:
            self.locked = False
            self.calls.append("lock_exit")

    def current_identity(self, operation):
        self._call("current_identity")
        identity = BackendStateIdentity.from_preparation(operation.preparation)
        if self.source_drift:
            return replace(identity, source_sequence=identity.source_sequence + 1)
        return identity

    def write_hidden_target(self, operation):
        self._call("write_hidden_target")
        publication = TargetPublication.expected(operation.preparation)
        if self.wrong_target:
            publication = replace(publication, vector_rows=publication.vector_rows + 1)
        self.hidden = publication
        return publication

    def validate_hidden_target(self, operation, publication):
        self._call("validate_hidden_target")
        return publication

    def commit_visibility(self, operation, publication):
        self._call("commit_visibility")
        self.visible = True

    def validate_visible_target(self, operation, publication):
        self._call("validate_visible_target")
        assert self.visible

    def discard_hidden_target(self, operation, publication):
        self._call("discard_hidden_target")
        self.hidden = None

    def restore_rollback(self, operation):
        self._call("restore_rollback")
        self.visible = False
        self.rolled_back = True

    def validate_rollback(self, operation):
        self._call("validate_rollback")
        assert self.rolled_back and not self.visible


def test_successful_saga_orders_hidden_validation_before_visibility():
    operation = ready_operation()
    adapter = FakeAdapter(operation)
    result = execute_cutover_saga(operation, adapter)
    assert result.outcome == "published"
    assert result.rollback_verified is False
    assert result.phases == (
        "lock_acquired",
        "source_revalidated",
        "hidden_target_written",
        "hidden_target_validated",
        "visibility_committed",
        "visible_target_validated",
    )
    assert adapter.calls.index("validate_hidden_target") < adapter.calls.index(
        "commit_visibility"
    )
    assert adapter.calls[-1] == "lock_exit"


def test_source_drift_aborts_before_any_target_write():
    operation = ready_operation()
    adapter = FakeAdapter(operation, source_drift=True)
    result = execute_cutover_saga(operation, adapter)
    assert result.outcome == "aborted"
    assert "write_hidden_target" not in adapter.calls
    assert result.failure_type == "RuntimeError"
    assert "private" not in result.failure_type


def test_hidden_validation_failure_discards_target_under_same_lock():
    operation = ready_operation()
    adapter = FakeAdapter(operation, fail_at="validate_hidden_target")
    result = execute_cutover_saga(operation, adapter)
    assert result.outcome == "aborted"
    assert result.phases[-1] == "hidden_target_discarded"
    assert adapter.hidden is None
    assert adapter.calls.index("discard_hidden_target") < adapter.calls.index("lock_exit")


def test_wrong_hidden_target_identity_is_discarded():
    operation = ready_operation()
    adapter = FakeAdapter(operation, wrong_target=True)
    result = execute_cutover_saga(operation, adapter)
    assert result.outcome == "aborted"
    assert adapter.hidden is None
    assert "commit_visibility" not in adapter.calls


def test_visible_validation_failure_restores_and_verifies_rollback():
    operation = ready_operation()
    adapter = FakeAdapter(operation, fail_at="validate_visible_target")
    result = execute_cutover_saga(operation, adapter)
    assert result.outcome == "rolled_back"
    assert result.rollback_verified is True
    assert result.phases[-2:] == ("rollback_restored", "rollback_validated")
    assert adapter.rolled_back is True
    assert adapter.calls.index("validate_rollback") < adapter.calls.index("lock_exit")


def test_fault_after_visibility_commit_also_rolls_back():
    operation = ready_operation()
    adapter = FakeAdapter(operation)

    def hook(phase):
        if phase == "visibility_committed":
            raise LookupError("private fault")

    result = execute_cutover_saga(operation, adapter, fault_hook=hook)
    assert result.outcome == "rolled_back"
    assert result.failure_type == "LookupError"
    assert result.rollback_verified is True


def test_rollback_failure_raises_bounded_recovery_error():
    operation = ready_operation()
    adapter = FakeAdapter(
        operation,
        fail_at="validate_visible_target",
    )
    original = adapter.restore_rollback

    def failing_restore(current):
        original(current)
        raise OSError("private disk error")

    adapter.restore_rollback = failing_restore
    with pytest.raises(CutoverRecoveryError) as captured:
        execute_cutover_saga(operation, adapter)
    assert captured.value.failure_type == "RuntimeError"
    assert captured.value.recovery_failure_type == "OSError"
    assert "private" not in str(captured.value)


def test_lock_acquisition_failure_is_bounded_and_never_recovers_outside_lock():
    operation = ready_operation()
    adapter = FakeAdapter(operation, fail_at="lock_enter")
    result = execute_cutover_saga(operation, adapter)
    assert result.outcome == "aborted"
    assert result.phases == ()
    assert result.failure_type == "RuntimeError"
    assert adapter.calls == ["lock_enter"]


def test_nonready_operation_and_incomplete_adapter_are_refused():
    operation = ready_operation()
    planned = replace(operation, state="planned", attempt=0)
    with pytest.raises(ValueError, match="ready"):
        execute_cutover_saga(planned, FakeAdapter(planned))
    with pytest.raises(ValueError, match="required contract"):
        execute_cutover_saga(operation, object())


def test_trace_digest_is_deterministic_and_contains_no_error_message():
    operation = ready_operation()
    first = execute_cutover_saga(operation, FakeAdapter(operation))
    second = execute_cutover_saga(operation, FakeAdapter(operation))
    assert first.trace_digest == second.trace_digest
    assert "private" not in repr(first)
