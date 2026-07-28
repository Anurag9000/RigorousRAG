"""Single-thread keyed deadline scheduler for crash-persistent job queues.

The durable database remains the authority for whether work is actually due.  This
helper only avoids one ``threading.Timer`` per delayed job inside one process.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(order=True)
class _ScheduledEntry:
    due_at: float
    sequence: int
    key: str = field(compare=False)
    callback: Callable[..., Any] = field(compare=False)
    args: Tuple[Any, ...] = field(compare=False, default_factory=tuple)


class DueScheduler:
    """Run the latest callback for each key when its wall-clock deadline is due."""

    def __init__(self, *, name: str = "rigorousrag-due-scheduler") -> None:
        self._condition = threading.Condition()
        self._heap: list[_ScheduledEntry] = []
        self._current: Dict[str, int] = {}
        self._sequence = itertools.count(1)
        self._shutdown = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def schedule(
        self,
        key: str,
        due_at: float,
        callback: Callable[..., Any],
        *args: Any,
    ) -> bool:
        """Schedule or replace one keyed callback.

        Returns ``False`` after shutdown. Replacing a key invalidates its older heap
        entry without requiring an O(n) heap deletion.
        """

        identifier = str(key or "").strip()
        if not identifier:
            raise ValueError("Scheduler keys may not be empty.")
        deadline = max(0.0, float(due_at))
        if not callable(callback):
            raise TypeError("callback must be callable.")
        with self._condition:
            if self._shutdown:
                return False
            sequence = next(self._sequence)
            self._current[identifier] = sequence
            heapq.heappush(
                self._heap,
                _ScheduledEntry(deadline, sequence, identifier, callback, tuple(args)),
            )
            self._condition.notify()
            return True

    def cancel(self, key: str) -> bool:
        """Invalidate one pending callback without scanning the heap."""

        identifier = str(key or "").strip()
        with self._condition:
            removed = self._current.pop(identifier, None) is not None
            if removed:
                self._condition.notify()
            return removed

    def pending_count(self) -> int:
        with self._condition:
            return len(self._current)

    def shutdown(self, *, wait: bool = True, timeout: Optional[float] = 2.0) -> None:
        """Cancel pending callbacks and stop the scheduler thread."""

        with self._condition:
            if not self._shutdown:
                self._shutdown = True
                self._current.clear()
                self._heap.clear()
                self._condition.notify_all()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)

    def _discard_stale_locked(self) -> None:
        while self._heap:
            entry = self._heap[0]
            if self._current.get(entry.key) == entry.sequence:
                return
            heapq.heappop(self._heap)

    def _run(self) -> None:
        while True:
            entry: Optional[_ScheduledEntry] = None
            with self._condition:
                while entry is None:
                    if self._shutdown:
                        return
                    self._discard_stale_locked()
                    if not self._heap:
                        self._condition.wait()
                        continue
                    candidate = self._heap[0]
                    delay = candidate.due_at - time.time()
                    if delay > 0:
                        self._condition.wait(timeout=delay)
                        continue
                    heapq.heappop(self._heap)
                    if self._current.get(candidate.key) != candidate.sequence:
                        continue
                    self._current.pop(candidate.key, None)
                    entry = candidate
            try:
                entry.callback(*entry.args)
            except Exception:
                # Scheduler liveness must not depend on one callback's correctness.
                continue
