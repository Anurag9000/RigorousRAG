import threading

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


def test_constructor_bounds_and_validates_limits():
    with pytest.raises(ValueError, match="must be integers"):
        BoundedExecutor(
            max_workers="bad",
            max_pending=1,
            thread_name_prefix="test",
        )

    pool = BoundedExecutor(
        max_workers=999999,
        max_pending=999999999,
        thread_name_prefix="x" * 500,
    )
    try:
        assert pool.max_workers == 256
        assert pool.max_pending == 100_000
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
