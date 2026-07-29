import threading
import time

import pytest

import tools.due_scheduler as due_scheduler
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


def test_callback_failure_including_system_exit_does_not_kill_scheduler():
    scheduler = DueScheduler(name="test-due-error")
    fired = threading.Event()
    try:
        def fail():
            raise SystemExit("do not kill scheduler")

        assert scheduler.schedule("bad", time.time(), fail)
        assert scheduler.schedule("good", time.time() + 0.02, fired.set)
        assert fired.wait(1.0)
        assert scheduler.thread_started() is True
    finally:
        scheduler.shutdown()


def test_scheduler_rejects_nonfinite_negative_deadlines_and_invalid_keys():
    scheduler = DueScheduler(name="test-due-invalid")
    try:
        for deadline in (float("nan"), float("inf"), float("-inf"), -1):
            with pytest.raises(ValueError, match="finite and non-negative"):
                scheduler.schedule("job", deadline, lambda: None)
        for key in (None, object(), "", "j" * 201, "bad\nkey", "bad\x00key"):
            with pytest.raises(ValueError, match="keys"):
                scheduler.schedule(key, time.time(), lambda: None)
        with pytest.raises(ValueError, match="64"):
            scheduler.schedule("job", time.time(), lambda *_args: None, *range(65))
        with pytest.raises(TypeError, match="callable"):
            scheduler.schedule("job", time.time(), object())
    finally:
        scheduler.shutdown()


def test_scheduler_key_capacity_fails_closed_without_replacing_existing_key():
    scheduler = DueScheduler(name="test-due-capacity", max_pending_keys=1)
    try:
        assert scheduler.schedule("job-1", time.time() + 100, lambda: None)
        with pytest.raises(RuntimeError, match="capacity"):
            scheduler.schedule("job-2", time.time() + 100, lambda: None)
        assert scheduler.pending_count() == 1
        assert scheduler.schedule("job-1", time.time() + 200, lambda: None)
        assert scheduler.pending_count() == 1
    finally:
        scheduler.shutdown()


def test_scheduler_constructor_validates_capacity_and_name():
    for invalid in ("bad", True, 1.5, 0, -1):
        with pytest.raises(ValueError):
            DueScheduler(max_pending_keys=invalid)
    with pytest.raises(ValueError, match="name"):
        DueScheduler(name=object())

    scheduler = DueScheduler(
        max_pending_keys=999999999,
        name=" scheduler\r\nInjected: yes\x00 " + "x" * 500,
    )
    try:
        assert scheduler.max_pending_keys == 1_000_000
        assert len(scheduler._thread_name) == 100
        assert "\r" not in scheduler._thread_name
        assert "\n" not in scheduler._thread_name
        assert "\x00" not in scheduler._thread_name
    finally:
        scheduler.shutdown()


def test_invalid_shutdown_wait_does_not_transition_scheduler():
    scheduler = DueScheduler(name="test-due-shutdown-validation")
    try:
        with pytest.raises(ValueError, match="wait"):
            scheduler.shutdown(wait="yes")
        assert scheduler._shutdown is False
        fired = threading.Event()
        assert scheduler.schedule("job", time.time(), fired.set)
        assert fired.wait(1.0)
    finally:
        scheduler.shutdown()


def test_thread_start_failure_does_not_publish_broken_thread(monkeypatch):
    scheduler = DueScheduler(name="test-due-start-failure")
    original_start = threading.Thread.start
    monkeypatch.setattr(
        threading.Thread,
        "start",
        lambda _self: (_ for _ in ()).throw(RuntimeError("cannot start")),
    )
    try:
        with pytest.raises(RuntimeError, match="cannot start"):
            scheduler.schedule("job", time.time(), lambda: None)
        assert scheduler._thread is None
        assert scheduler.pending_count() == 0
    finally:
        monkeypatch.setattr(threading.Thread, "start", original_start)
        scheduler.shutdown()


def test_invalid_wall_clock_does_not_fire_callback_early(monkeypatch):
    scheduler = DueScheduler(name="test-due-invalid-clock")
    calls = []
    original_time = due_scheduler.time.time
    clock_values = iter([float("nan"), float("inf"), original_time() + 100])
    monkeypatch.setattr(due_scheduler.time, "time", lambda: next(clock_values))
    try:
        assert scheduler.schedule("job", 1, calls.append, "fired")
        deadline = time.monotonic() + 1.5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == ["fired"]
    finally:
        monkeypatch.setattr(due_scheduler.time, "time", original_time)
        scheduler.shutdown()
