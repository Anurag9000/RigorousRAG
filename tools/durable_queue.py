"""Durable work-queue contracts and a deterministic in-memory reference implementation.

The in-memory implementation is suitable for unit tests and single-process development.
Production deployments should provide an external durable queue with equivalent
idempotency, visibility-timeout, retry, acknowledgement, and dead-letter semantics.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol


class CoordinationError(RuntimeError):
    """Raised when a queue state transition is invalid."""


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    payload: Mapping[str, object]
    idempotency_key: str
    attempts: int


@dataclass(frozen=True)
class ClaimedMessage(QueueMessage):
    receipt: str
    owner: str
    visibility_deadline: float


class DurableQueue(Protocol):
    def enqueue(self, payload: Mapping[str, object], *, idempotency_key: str) -> str: ...

    def claim(self, owner: str, *, visibility_timeout: float) -> ClaimedMessage | None: ...

    def ack(self, receipt: str) -> None: ...

    def nack(self, receipt: str, *, retry_delay: float = 0.0) -> None: ...


@dataclass
class _QueuedState:
    message_id: str
    payload: dict[str, object]
    idempotency_key: str
    sequence: int
    available_at: float
    invisible_until: float = 0.0
    receipt: str | None = None
    owner: str | None = None
    attempts: int = 0
    acked: bool = False
    dead_lettered: bool = False


class InMemoryDurableQueue:
    """Reference at-least-once queue with idempotency, visibility, retries, and a DLQ."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer.")
        self._max_attempts = int(max_attempts)
        self._clock = clock
        self._sequence = 0
        self._receipt_sequence = 0
        self._messages: dict[str, _QueuedState] = {}
        self._idempotency: dict[str, str] = {}

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty.")
        return value.strip()

    def enqueue(self, payload: Mapping[str, object], *, idempotency_key: str) -> str:
        key = self._identifier(idempotency_key, "idempotency_key")
        existing = self._idempotency.get(key)
        if existing is not None:
            return existing
        self._sequence += 1
        message_id = f"msg-{self._sequence:016d}"
        self._messages[message_id] = _QueuedState(
            message_id=message_id,
            payload=dict(payload),
            idempotency_key=key,
            sequence=self._sequence,
            available_at=self._clock(),
        )
        self._idempotency[key] = message_id
        return message_id

    def _expire_or_dead_letter(self, state: _QueuedState, now: float) -> None:
        if state.receipt is None or state.invisible_until > now:
            return
        state.receipt = None
        state.owner = None
        state.invisible_until = 0.0
        if state.attempts >= self._max_attempts:
            state.dead_lettered = True

    def claim(self, owner: str, *, visibility_timeout: float) -> ClaimedMessage | None:
        holder = self._identifier(owner, "owner")
        timeout = float(visibility_timeout)
        if timeout <= 0.0:
            raise ValueError("visibility_timeout must be positive.")
        now = self._clock()
        for state in sorted(self._messages.values(), key=lambda item: item.sequence):
            self._expire_or_dead_letter(state, now)
            if (
                state.acked
                or state.dead_lettered
                or state.available_at > now
                or state.receipt is not None
            ):
                continue
            state.attempts += 1
            self._receipt_sequence += 1
            state.receipt = f"receipt-{self._receipt_sequence:016d}"
            state.owner = holder
            state.invisible_until = now + timeout
            return ClaimedMessage(
                state.message_id,
                dict(state.payload),
                state.idempotency_key,
                state.attempts,
                state.receipt,
                holder,
                state.invisible_until,
            )
        return None

    def _by_receipt(self, receipt: str) -> _QueuedState:
        token = self._identifier(receipt, "receipt")
        for state in self._messages.values():
            if state.receipt == token and not state.acked and not state.dead_lettered:
                if state.invisible_until <= self._clock():
                    self._expire_or_dead_letter(state, self._clock())
                    break
                return state
        raise CoordinationError("receipt is invalid or expired")

    def ack(self, receipt: str) -> None:
        state = self._by_receipt(receipt)
        state.acked = True
        state.receipt = None
        state.owner = None
        state.invisible_until = 0.0

    def nack(self, receipt: str, *, retry_delay: float = 0.0) -> None:
        delay = float(retry_delay)
        if delay < 0.0:
            raise ValueError("retry_delay must not be negative.")
        state = self._by_receipt(receipt)
        state.receipt = None
        state.owner = None
        state.invisible_until = 0.0
        if state.attempts >= self._max_attempts:
            state.dead_lettered = True
        else:
            state.available_at = self._clock() + delay

    def dead_letters(self) -> tuple[QueueMessage, ...]:
        now = self._clock()
        for state in self._messages.values():
            self._expire_or_dead_letter(state, now)
        return tuple(
            QueueMessage(
                item.message_id,
                dict(item.payload),
                item.idempotency_key,
                item.attempts,
            )
            for item in sorted(self._messages.values(), key=lambda value: value.sequence)
            if item.dead_lettered
        )


__all__ = [
    "ClaimedMessage",
    "CoordinationError",
    "DurableQueue",
    "InMemoryDurableQueue",
    "QueueMessage",
]
