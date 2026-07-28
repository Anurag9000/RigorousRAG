import threading
import time

import pytest

from tools.due_scheduler import DueScheduler


def test_scheduler_starts_no_thread_until_first_schedule():
    scheduler = DueScheduler(name="test-due-lazy")
    try:
        assert scheduler.thread_started() is False
        assert scheduler.pending_count() == 0
    finally:
        scheduler.shutdown()
    assert scheduler.thread_started() is False


def test_due_callback_runs_once():
    scheduler = DueScheduler(name="test-due-once")
    fired = threading.Event()
    calls = []
    try:
        assert scheduler.schedule(
            "job-1",
            time.time() + 0.02,
            lambda value: (calls.append(value), fired.set()),
            "done",
        )
        assert scheduler.thread_started() is True
        assert fired.wait(1.0)
        assert calls == ["done"]
        assert scheduler.pending_count() == 0
    finally:
        scheduler.shutdown()


def test_rescheduling_same_key_invalidates_older_entry():
    scheduler = DueScheduler(name="test-due-replace")
    fired = threading.Event()
    calls = []
    try:
        assert scheduler.schedule(
            "job-1",
            time.time() + 0.2,
            lambda: calls.append("old"),
        )
        assert scheduler.schedule(
            "job-1",
            time.time() + 0.02,
            lambda: (calls.append("new"), fired.set()),
        )
        assert scheduler.pending_count() == 1
        assert fired.wait(1.0)
        time.sleep(0.25)
        assert calls == ["new"]
    finally:
        scheduler.shutdown()


def test_repeated_far_future_replacement_compacts_stale_heap():
    scheduler = DueScheduler(name="test-due-compact")
    try:
        for index in range(5000):
            assert scheduler.schedule(
                "job-1",
                time.time() + 10_000 + index,
                lambda: None,
            )
        assert scheduler.pending_count() == 1
        assert scheduler.heap_size() <= 1024
    finally:
        scheduler.shutdown()


def test_cancel_and_shutdown_prevent_callbacks():
    scheduler = DueScheduler(name="test-due-cancel")
    calls = []
    assert scheduler.schedule("job-1", time.time() + 0.05, calls.append, "cancelled")
    assert scheduler.cancel("job-1") is True
    assert scheduler.cancel("job-1") is False
    assert scheduler.pending_count() == 0
    time.sleep(0.1)
    assert calls == []

    assert scheduler.schedule("job-2", time.time() + 10, calls.append, "shutdown")
    scheduler.shutdown(timeout=float("nan"))
    assert scheduler.pending_count() == 0
    assert scheduler.thread_started() is False
    assert scheduler.schedule("job-3", time.time(), calls.append, "late") is False
    assert calls == []


def test_callback_failure_does_not_kill_scheduler():
    scheduler = DueScheduler(name="test-due-error")
    fired = threading.Event()
    try:
        def fail():
            raise RuntimeError("boom")

        assert scheduler.schedule("bad", time.time(), fail)
        assert scheduler.schedule("good", time.time() + 0.02, fired.set)
        assert fired.wait(1.0)
    finally:
        scheduler.shutdown()


def test_scheduler_rejects_nonfinite_deadlines_and_invalid_keys():
    scheduler = DueScheduler(name="test-due-invalid")
    try:
        for deadline in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="finite"):
                scheduler.schedule("job", deadline, lambda: None)
        with pytest.raises(ValueError, match="between 1 and 200"):
            scheduler.schedule("", time.time(), lambda: None)
        with pytest.raises(ValueError, match="between 1 and 200"):
            scheduler.schedule("j" * 201, time.time(), lambda: None)
        with pytest.raises(ValueError, match="64"):
            scheduler.schedule("job", time.time(), lambda *_args: None, *range(65))
    finally:
        scheduler.shutdown()


def test_scheduler_key_capacity_fails_closed():
    scheduler = DueScheduler(name="test-due-capacity", max_pending_keys=1)
    try:
        assert scheduler.schedule("job-1", time.time() + 100, lambda: None)
        with pytest.raises(RuntimeError, match="capacity"):
            scheduler.schedule("job-2", time.time() + 100, lambda: None)
    finally:
        scheduler.shutdown()


def test_scheduler_constructor_validates_capacity():
    with pytest.raises(ValueError, match="integer"):
        DueScheduler(max_pending_keys="bad")

    scheduler = DueScheduler(max_pending_keys=999999999, name="x" * 500)
    try:
        assert scheduler.max_pending_keys == 1_000_000
    finally:
        scheduler.shutdown()
