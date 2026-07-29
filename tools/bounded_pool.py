"""Process-wide executor with explicit running-plus-queued admission."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional, Set


def _positive_integer(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError("Executor limits must be integers.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Executor limits must be integers.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("Executor limits must be integers.")
    if parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return min(parsed, maximum)


def _thread_prefix(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("thread_name_prefix must be a string.")
    cleaned = " ".join(
        value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split()
    )
    return cleaned[:100] or "bounded-worker"


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean.")
    return value


class BoundedExecutor:
    """Reject excess work instead of using ThreadPoolExecutor's unbounded queue."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_pending: int,
        thread_name_prefix: str,
    ) -> None:
        workers = _positive_integer(
            max_workers,
            "max_workers",
            maximum=256,
        )
        pending = _positive_integer(
            max_pending,
            "max_pending",
            maximum=100_000,
        )
        pending = max(workers, pending)
        prefix = _thread_prefix(thread_name_prefix)
        self.max_workers = workers
        self.max_pending = pending
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=prefix,
        )
        self._admission = threading.BoundedSemaphore(pending)
        self._lock = threading.Lock()
        self._shutdown = False
        self._inflight = 0
        self._tracked_futures: Set[int] = set()

    def _release(self, future: Future[Any]) -> None:
        """Release one admitted future exactly once, including cancellation paths."""

        token = id(future)
        with self._lock:
            if token not in self._tracked_futures:
                return
            self._tracked_futures.remove(token)
            if self._inflight <= 0:
                return
            self._inflight -= 1
            self._admission.release()

    def submit(
        self,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[Future[Any]]:
        """Return a future, or ``None`` when saturated or shutting down."""

        if not callable(function):
            raise TypeError("function must be callable.")
        with self._lock:
            if self._shutdown:
                return None
            if not self._admission.acquire(blocking=False):
                return None
            try:
                future = self._executor.submit(function, *args, **kwargs)
            except Exception:
                self._admission.release()
                return None
            self._inflight += 1
            self._tracked_futures.add(id(future))
        try:
            future.add_done_callback(self._release)
        except Exception:
            future.cancel()
            self._release(future)
            return None
        return future

    def shutdown(
        self,
        *,
        wait: bool = False,
        cancel_futures: bool = True,
    ) -> None:
        wait_value = _boolean(wait, "wait")
        cancel_value = _boolean(cancel_futures, "cancel_futures")
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(
            wait=wait_value,
            cancel_futures=cancel_value,
        )

    def available_slots(self) -> int:
        """Return a diagnostic count without consuming admission permits."""

        with self._lock:
            return max(self.max_pending - self._inflight, 0)
