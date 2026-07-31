import threading
from decimal import Decimal
from fractions import Fraction

import pytest

from tools.bounded_pool import BoundedExecutor


def test_pool_rejects_excess_work_and_releases_after_completion():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="test-bounded",
    )
    release = threading.Event()
    started = threading.Event()
    try:
        def blocking():
            started.set()
            release.wait(1.0)
            return "done"

        first = pool.submit(blocking)
        assert first is not None
        assert started.wait(1.0)
        assert pool.available_slots() == 0
        assert pool.submit(lambda: "overflow") is None

        release.set()
        assert first.result(timeout=1.0) == "done"
        second = pool.submit(lambda: "next")
        assert second is not None
        assert second.result(timeout=1.0) == "next"
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_pending_queue_is_bounded_including_running_work():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=2,
        thread_name_prefix="test-pending",
    )
    release = threading.Event()
    try:
        running = pool.submit(lambda: release.wait(1.0))
        queued = pool.submit(lambda: "queued")
        rejected = pool.submit(lambda: "rejected")

        assert running is not None
        assert queued is not None
        assert rejected is None
        assert pool.available_slots() == 0

        release.set()
        assert running.result(timeout=1.0) is True
        assert queued.result(timeout=1.0) == "queued"
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_available_slots_does_not_consume_real_admission():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=2,
        thread_name_prefix="test-diagnostic",
    )
    try:
        assert pool.available_slots() == 2
        assert pool.available_slots() == 2
        first = pool.submit(lambda: "one")
        second = pool.submit(lambda: "two")
        assert first is not None
        assert second is not None
        assert first.result(timeout=1.0) == "one"
        assert second.result(timeout=1.0) == "two"
    finally:
        pool.shutdown(wait=True)


def test_cancelled_queued_future_releases_admission():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=2,
        thread_name_prefix="test-cancel",
    )
    release = threading.Event()
    try:
        running = pool.submit(lambda: release.wait(1.0))
        queued = pool.submit(lambda: "queued")
        assert running is not None
        assert queued is not None
        assert queued.cancel() is True
        assert pool.available_slots() == 1
        replacement = pool.submit(lambda: "replacement")
        assert replacement is not None
        release.set()
        assert running.result(timeout=1.0) is True
        assert replacement.result(timeout=1.0) == "replacement"
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_task_exception_releases_admission():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="test-error",
    )
    try:
        def fail():
            raise RuntimeError("task failure")

        future = pool.submit(fail)
        assert future is not None
        with pytest.raises(RuntimeError, match="task failure"):
            future.result(timeout=1.0)
        assert pool.available_slots() == 1
        replacement = pool.submit(lambda: "ok")
        assert replacement is not None
        assert replacement.result(timeout=1.0) == "ok"
    finally:
        pool.shutdown(wait=True)


def test_executor_submit_failure_releases_admission(monkeypatch):
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="test-submit-failure",
    )
    original_submit = pool._executor.submit
    try:
        monkeypatch.setattr(
            pool._executor,
            "submit",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("closed")),
        )
        assert pool.submit(lambda: "never") is None
        assert pool.available_slots() == 1

        monkeypatch.setattr(pool._executor, "submit", original_submit)
        future = pool.submit(lambda: "recovered")
        assert future is not None
        assert future.result(timeout=1.0) == "recovered"
    finally:
        pool.shutdown(wait=True)


def test_constructor_bounds_and_strictly_validates_limits():
    invalid = [
        {"max_workers": "bad", "max_pending": 1},
        {"max_workers": True, "max_pending": 1},
        {"max_workers": 1.5, "max_pending": 2},
        {"max_workers": Decimal("1.5"), "max_pending": 2},
        {"max_workers": Fraction(3, 2), "max_pending": 2},
        {"max_workers": 0, "max_pending": 1},
        {"max_workers": 1, "max_pending": False},
        {"max_workers": 1, "max_pending": 1.5},
        {"max_workers": 1, "max_pending": Decimal("1.5")},
        {"max_workers": 1, "max_pending": 0},
    ]
    for arguments in invalid:
        with pytest.raises(ValueError):
            BoundedExecutor(
                **arguments,
                thread_name_prefix="test",
            )

    with pytest.raises(ValueError, match="thread_name_prefix"):
        BoundedExecutor(
            max_workers=1,
            max_pending=1,
            thread_name_prefix=object(),
        )

    pool = BoundedExecutor(
        max_workers=999999,
        max_pending=999999999,
        thread_name_prefix="x" * 500,
    )
    try:
        assert pool.max_workers == 256
        assert pool.max_pending == 100_000
        assert len(pool._executor._thread_name_prefix) == 100
    finally:
        pool.shutdown(wait=True)


def test_constructor_accepts_exact_index_protocol_limits():
    class ExactInteger:
        def __index__(self):
            return 2

    pool = BoundedExecutor(
        max_workers=ExactInteger(),
        max_pending=ExactInteger(),
        thread_name_prefix="index-protocol",
    )
    try:
        assert pool.max_workers == 2
        assert pool.max_pending == 2
    finally:
        pool.shutdown(wait=True)


def test_pending_capacity_is_never_below_worker_count():
    pool = BoundedExecutor(
        max_workers=4,
        max_pending=1,
        thread_name_prefix="test-capacity",
    )
    try:
        assert pool.max_workers == 4
        assert pool.max_pending == 4
        assert pool.available_slots() == 4
    finally:
        pool.shutdown(wait=True)


def test_thread_prefix_controls_are_removed():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix=" worker\r\nInjected: yes\x00 ",
    )
    try:
        prefix = pool._executor._thread_name_prefix
        assert "\r" not in prefix
        assert "\n" not in prefix
        assert "\x00" not in prefix
        assert prefix == "worker Injected: yes"
    finally:
        pool.shutdown(wait=True)


def test_invalid_shutdown_flags_do_not_transition_pool_to_shutdown():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="test-shutdown-validation",
    )
    try:
        with pytest.raises(ValueError, match="wait"):
            pool.shutdown(wait="yes")
        with pytest.raises(ValueError, match="cancel_futures"):
            pool.shutdown(cancel_futures=1)
        assert pool._shutdown is False
        future = pool.submit(lambda: "still-open")
        assert future is not None
        assert future.result(timeout=1.0) == "still-open"
    finally:
        pool.shutdown(wait=True)


def test_submit_after_shutdown_fails_closed():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="test-shutdown",
    )
    pool.shutdown()

    assert pool.submit(lambda: "late") is None
