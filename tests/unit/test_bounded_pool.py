import threading

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


def test_submit_after_shutdown_fails_closed():
    pool = BoundedExecutor(
        max_workers=1,
        max_pending=1,
        thread_name_prefix="test-shutdown",
    )
    pool.shutdown()

    assert pool.submit(lambda: "late") is None
