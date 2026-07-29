"""Bounded in-process sliding-window rate limiting."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Deque, Dict


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError("Rate-limit settings must be integers.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Rate-limit settings must be integers.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("Rate-limit settings must be integers.")
    if not 1 <= parsed <= 1_000_000:
        raise ValueError(f"{label} must be between 1 and 1,000,000.")
    return parsed


class SlidingWindowRateLimiter:
    def __init__(
        self,
        requests_per_minute: int = 60,
        *,
        max_keys: int = 100_000,
    ) -> None:
        self.limit = _positive_integer(
            requests_per_minute,
            "requests_per_minute",
        )
        self.max_keys = _positive_integer(max_keys, "max_keys")
        self.window_seconds = 60.0
        self._events: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Rate-limit keys must be strings.")
        key = value.strip()
        if (
            not key
            or len(key) > 200
            or any(ord(character) < 32 or ord(character) == 127 for character in key)
        ):
            raise ValueError(
                "Rate-limit keys must contain between 1 and 200 valid characters."
            )
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
            events = self._events.get(identifier)
            if events is None:
                if len(self._events) >= self.max_keys:
                    self._prune_stale_keys(cutoff)
                    if len(self._events) >= self.max_keys:
                        raise RuntimeError("Rate-limit key capacity is exhausted.")
                events = deque()
                self._events[identifier] = events
            if events and current < events[-1]:
                raise ValueError("Rate-limit time must not move backwards for a key.")
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry = events[0] + self.window_seconds - current
                return min(max(float(retry), 0.001), self.window_seconds)
            events.append(current)
            return 0.0
