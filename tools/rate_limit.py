"""Bounded in-process sliding-window rate limiting."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class SlidingWindowRateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        *,
        max_keys: int = 100_000,
    ) -> None:
        try:
            parsed_limit = int(requests_per_minute)
            parsed_keys = int(max_keys)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Rate-limit settings must be integers.") from exc
        self.limit = max(1, min(parsed_limit, 1_000_000))
        self.max_keys = max(1, min(parsed_keys, 1_000_000))
        self.window_seconds = 60.0
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    @staticmethod
    def _key(value: object) -> str:
        key = str(value or "").strip()
        if not key or len(key) > 200:
            raise ValueError("Rate-limit keys must contain between 1 and 200 characters.")
        return key

    @staticmethod
    def _time(value: object) -> float:
        try:
            current = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Rate-limit time must be numeric.") from exc
        if not math.isfinite(current) or current < 0:
            raise ValueError("Rate-limit time must be finite and non-negative.")
        return current

    def _prune_stale_keys(self, cutoff: float) -> None:
        stale = [
            key
            for key, events in self._events.items()
            if not events or events[-1] <= cutoff
        ]
        for key in stale:
            self._events.pop(key, None)

    def retry_after(self, key: str, now: float | None = None) -> float:
        identifier = self._key(key)
        current = self._time(time.monotonic() if now is None else now)
        cutoff = current - self.window_seconds
        with self._lock:
            if identifier not in self._events and len(self._events) >= self.max_keys:
                self._prune_stale_keys(cutoff)
                if len(self._events) >= self.max_keys:
                    raise RuntimeError("Rate-limit key capacity is exhausted.")
            events = self._events[identifier]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry = events[0] + self.window_seconds - current
                return max(float(retry), 0.001)
            events.append(current)
            return 0.0
