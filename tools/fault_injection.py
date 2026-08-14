"""Deterministic named failure injection for recovery and distributed-systems tests."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


class InjectedFault(RuntimeError):
    def __init__(self, point: str, invocation: int) -> None:
        self.point = point
        self.invocation = invocation
        super().__init__(f"injected fault at {point!r} invocation {invocation}")


@dataclass(frozen=True)
class FaultEvent:
    point: str
    invocation: int
    injected: bool
    timestamp: float


class FaultInjector:
    """Thread-safe fail-on-Nth checkpoint scheduler with an auditable event trail."""

    def __init__(
        self,
        rules: Mapping[str, Sequence[int]] | None = None,
        *,
        clock: Callable[[], float] = lambda: 0.0,
    ) -> None:
        normalized: dict[str, frozenset[int]] = {}
        for point, invocations in (rules or {}).items():
            selected = self._point(point)
            values = frozenset(int(value) for value in invocations)
            if any(value < 1 for value in values):
                raise ValueError("fault invocation numbers must be positive")
            normalized[selected] = values
        self._rules = normalized
        self._clock = clock
        self._counts: dict[str, int] = {}
        self._events: list[FaultEvent] = []
        self._lock = threading.Lock()

    @staticmethod
    def _point(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("fault point must be non-empty")
        point = value.strip()
        if len(point) > 128 or any(ord(ch) < 32 or ord(ch) == 127 for ch in point):
            raise ValueError("fault point is invalid")
        return point

    def checkpoint(self, point: str) -> None:
        selected = self._point(point)
        with self._lock:
            invocation = self._counts.get(selected, 0) + 1
            self._counts[selected] = invocation
            injected = invocation in self._rules.get(selected, frozenset())
            self._events.append(
                FaultEvent(selected, invocation, injected, float(self._clock()))
            )
        if injected:
            raise InjectedFault(selected, invocation)

    def events(self) -> tuple[FaultEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def count(self, point: str) -> int:
        selected = self._point(point)
        with self._lock:
            return self._counts.get(selected, 0)


__all__ = ["FaultEvent", "FaultInjector", "InjectedFault"]
