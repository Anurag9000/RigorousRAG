"""Single-thread keyed deadline scheduler for crash-persistent job queues.

The durable database remains authoritative for whether work is actually due. This
helper avoids one ``threading.Timer`` per delayed job, starts lazily, and compacts
invalidated heap entries so repeated key replacement remains bounded.
"""

from __future__ import annotations

import heapq
import itertools
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


def _positive_integer(value: Any, label: str, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if parsed <= 0:
        raise ValueError(f"{label} must be positive.")
    return min(parsed, maximum)


def _thread_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("name must be a string.")
    cleaned = " ".join(
        value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split()
    )
    return cleaned[:100] or "rigorousrag-due-scheduler"


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean.")
    return value


@dataclass(order=True)
class _ScheduledEntry:
    due_at: float
    sequence: int
    key: str = field(compare=False)
    callback: Callable[..., Any] = field(compare=False)
    args: Tuple[Any, ...] = field(compare=False, default_factory=tuple)


class DueScheduler:
    """Run the latest callback for each key when its wall-clock deadline is due."""

    def __init__(
        self,
        *,
        name: str = "rigorousrag-due-scheduler",
        max_pending_keys: int = 100_000,
    ) -> None:
        self.max_pending_keys = _positive_integer(
            max_pending_keys,
            "max_pending_keys",
            maximum=1_000_000,
        )
        self._condition = threading.Condition()
        self._heap: list[_ScheduledEntry] = []
        self._current: Dict[str, int] = {}
        self._sequence = itertools.count(1)
        self._shutdown = False
        self._thread_name = _thread_name(name)
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _identifier(key: object) -> str:
        if not isinstance(key, str):
            raise ValueError("Scheduler keys must be strings.")
        identifier = key.strip()
        if (
            not identifier
            or len(identifier) > 200
            or any(ord(character) < 32 or ord(character) == 127 for character in identifier)
        ):
            raise ValueError(
                "Scheduler keys must contain between 1 and 200 valid characters."
            )
        return identifier

    @staticmethod
    def _deadline(value: object) -> float:
        try:
            deadline = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Scheduler deadlines must be numeric.") from exc
        if not math.isfinite(deadline) or deadline < 0:
            raise ValueError("Scheduler deadlines must be finite and non-negative.")
        return deadline

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        thread = threading.Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        thread.start()
        self._thread = thread

    def _compact_locked(self) -> None:
        threshold = max(1024, len(self._current) * 4)
        if len(self._heap) <= threshold:
            return
        self._heap = [
            entry
            for entry in self._heap
            if self._current.get(entry.key) == entry.sequence
        ]
        heapq.heapify(self._heap)

    def schedule(
        self,
        key: str,
        due_at: float,
        callback: Callable[..., Any],
        *args: Any,
    ) -> bool:
        """Schedule or replace one keyed callback; return ``False`` after shutdown."""

        identifier = self._identifier(key)
        deadline = self._deadline(due_at)
        if not callable(callback):
            raise TypeError("callback must be callable.")
        if len(args) > 64:
            raise ValueError(
                "Scheduler callbacks may receive at most 64 positional arguments."
            )
        with self._condition:
            if self._shutdown:
                return False
            if (
                identifier not in self._current
                and len(self._current) >= self.max_pending_keys
            ):
                raise RuntimeError("Scheduler key capacity is exhausted.")
            self._ensure_thread_locked()
            sequence = next(self._sequence)
            self._current[identifier] = sequence
            heapq.heappush(
                self._heap,
                _ScheduledEntry(
                    deadline,
                    sequence,
                    identifier,
                    callback,
                    tuple(args),
                ),
            )
            self._compact_locked()
            self._condition.notify()
            return True

    def cancel(self, key: str) -> bool:
        """Invalidate one pending callback without scanning the heap."""

        identifier = self._identifier(key)
        with self._condition:
            removed = self._current.pop(identifier, None) is not None
            if removed:
                self._compact_locked()
                self._condition.notify()
            return removed

    def pending_count(self) -> int:
        with self._condition:
            return len(self._current)

    def heap_size(self) -> int:
        """Expose bounded heap size for diagnostics and deterministic tests."""

        with self._condition:
            return len(self._heap)

    def thread_started(self) -> bool:
        with self._condition:
            return bool(self._thread is not None and self._thread.is_alive())

    def shutdown(
        self,
        *,
        wait: bool = True,
        timeout: Optional[float] = 2.0,
    ) -> None:
        """Cancel pending callbacks and stop the scheduler thread."""

        wait_value = _boolean(wait, "wait")
        join_timeout: Optional[float]
        if timeout is None:
            join_timeout = None
        else:
            try:
                parsed_timeout = float(timeout)
            except (TypeError, ValueError, OverflowError):
                parsed_timeout = 2.0
            join_timeout = (
                max(0.0, min(parsed_timeout, 60.0))
                if math.isfinite(parsed_timeout)
                else 2.0
            )
        with self._condition:
            if not self._shutdown:
                self._shutdown = True
                self._current.clear()
                self._heap.clear()
                self._condition.notify_all()
            thread = self._thread
        if (
            wait_value
            and thread is not None
            and threading.current_thread() is not thread
        ):
            thread.join(timeout=join_timeout)

    def _discard_stale_locked(self) -> None:
        while self._heap:
            entry = self._heap[0]
            if self._current.get(entry.key) == entry.sequence:
                return
            heapq.heappop(self._heap)

    @staticmethod
    def _wall_time() -> Optional[float]:
        try:
            current = float(time.time())
        except (TypeError, ValueError, OverflowError):
            return None
        return current if math.isfinite(current) and current >= 0 else None

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
                    current = self._wall_time()
                    if current is None:
                        self._condition.wait(timeout=0.05)
                        continue
                    delay = candidate.due_at - current
                    if delay > 0:
                        self._condition.wait(timeout=min(delay, 86_400.0))
                        continue
                    heapq.heappop(self._heap)
                    if self._current.get(candidate.key) != candidate.sequence:
                        continue
                    self._current.pop(candidate.key, None)
                    entry = candidate
            try:
                entry.callback(*entry.args)
            except BaseException:
                continue
