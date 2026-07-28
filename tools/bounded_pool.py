"""Process-wide executor with explicit running-plus-queued admission."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional


class BoundedExecutor:
    """Reject excess work instead of using ThreadPoolExecutor's unbounded queue."""

    def __init__(
        self,
        *,
        max_workers: int,
        max_pending: int,
        thread_name_prefix: str,
    ) -> None:
        try:
            workers = int(max_workers)
            pending = int(max_pending)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Executor limits must be integers.") from exc
        workers = max(1, min(workers, 256))
        pending = max(workers, min(pending, 100_000))
        prefix = str(thread_name_prefix or "bounded-worker").strip()[:100]
        self.max_workers = workers
        self.max_pending = pending
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=prefix or "bounded-worker",
        )
        self._admission = threading.BoundedSemaphore(pending)
        self._lock = threading.Lock()
        self._shutdown = False
        self._inflight = 0

    def _release(self, _future: Future[Any]) -> None:
        with self._lock:
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
        future.add_done_callback(self._release)
        return future

    def shutdown(
        self,
        *,
        wait: bool = False,
        cancel_futures: bool = True,
    ) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=bool(wait), cancel_futures=bool(cancel_futures))

    def available_slots(self) -> int:
        """Return a diagnostic count without consuming admission permits."""

        with self._lock:
            return max(self.max_pending - self._inflight, 0)
