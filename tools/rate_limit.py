"""Small in-process sliding-window rate limiter for the single-worker service."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class SlidingWindowRateLimiter:
    def __init__(self, requests_per_minute: int = 60) -> None:
        self.limit = max(int(requests_per_minute), 1)
        self.window_seconds = 60.0
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def retry_after(self, key: str, now: float | None = None) -> float:
        current = now if now is not None else time.monotonic()
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return max(events[0] + self.window_seconds - current, 0.001)
            events.append(current)
            if not events:
                self._events.pop(key, None)
            return 0.0
