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
        workers = max(1, int(max_workers))
        pending = max(workers, int(max_pending))
        self.max_workers = workers
        self.max_pending = pending
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._admission = threading.BoundedSemaphore(pending)
        self._lock = threading.Lock()
        self._shutdown = False

    def _release(self, _future: Future[Any]) -> None:
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
        self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    def available_slots(self) -> int:
        """Return an approximate diagnostic count without changing admission state."""

        acquired = 0
        while self._admission.acquire(blocking=False):
            acquired += 1
        for _ in range(acquired):
            self._admission.release()
        return acquired
