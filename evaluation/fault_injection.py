"""Deterministic fault injection for retry, lease, and rollback verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar

T = TypeVar("T")
_MAX_RULES = 1_000


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > 300 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


class InjectedFault(RuntimeError):
    def __init__(self, stage: str, occurrence: int, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.occurrence = occurrence


@dataclass(frozen=True)
class FaultRule:
    stage: str
    occurrence: int = 1
    message: str = "deterministic injected fault"

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _identifier(self.stage, "stage"))
        if isinstance(self.occurrence, bool) or not isinstance(self.occurrence, int) or not 1 <= self.occurrence <= 1_000_000:
            raise ValueError("occurrence must be a positive bounded integer.")
        object.__setattr__(self, "message", _identifier(self.message, "message"))


@dataclass(frozen=True)
class FaultEvent:
    stage: str
    occurrence: int
    injected: bool


class FaultInjector:
    """Call-count fault scheduler with no sleeping, randomness, or global state."""

    def __init__(self, rules: tuple[FaultRule, ...] = ()) -> None:
        if not isinstance(rules, tuple) or len(rules) > _MAX_RULES or any(not isinstance(rule, FaultRule) for rule in rules):
            raise ValueError("rules must be a bounded tuple of FaultRule values.")
        keys = [(rule.stage, rule.occurrence) for rule in rules]
        if len(keys) != len(set(keys)):
            raise ValueError("fault rules must have unique stage/occurrence pairs.")
        self._rules = {(rule.stage, rule.occurrence): rule for rule in rules}
        self._counts: dict[str, int] = {}
        self._events: list[FaultEvent] = []

    @property
    def counts(self) -> Mapping[str, int]:
        return dict(self._counts)

    @property
    def events(self) -> tuple[FaultEvent, ...]:
        return tuple(self._events)

    def checkpoint(self, stage: str) -> None:
        selected = _identifier(stage, "stage")
        occurrence = self._counts.get(selected, 0) + 1
        self._counts[selected] = occurrence
        rule = self._rules.get((selected, occurrence))
        self._events.append(FaultEvent(selected, occurrence, rule is not None))
        if rule is not None:
            raise InjectedFault(selected, occurrence, rule.message)

    def call(self, stage: str, function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if not callable(function):
            raise ValueError("function must be callable.")
        self.checkpoint(stage)
        return function(*args, **kwargs)

    def reset(self) -> None:
        self._counts.clear()
        self._events.clear()


__all__ = ["FaultEvent", "FaultInjector", "FaultRule", "InjectedFault"]
