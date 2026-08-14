"""Dependency-free rate limiting, backpressure, and overload admission controls."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class AdmissionAction(str, Enum):
    ADMIT = "admit"
    BACKPRESSURE = "backpressure"
    SHED = "shed"


@dataclass(frozen=True)
class AdmissionDecision:
    action: AdmissionAction
    reason: str
    inflight: int
    queue_depth: int


@dataclass(frozen=True)
class AdmissionLease:
    lease_id: str


class TokenBucket:
    """Thread-safe token bucket with an injectable monotonic clock."""

    def __init__(
        self,
        *,
        capacity: float,
        refill_per_second: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0.0 or refill_per_second < 0.0:
            raise ValueError("token bucket capacity must be positive and refill non-negative")
        self._capacity = float(capacity)
        self._refill = float(refill_per_second)
        self._clock = clock
        self._tokens = float(capacity)
        self._updated_at = float(clock())
        self._lock = threading.Lock()

    def _refill_now(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated_at)
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill)
        self._updated_at = max(self._updated_at, now)

    def allow(self, *, cost: float = 1.0) -> bool:
        required = float(cost)
        if required <= 0.0 or required > self._capacity:
            raise ValueError("token cost must be positive and no greater than capacity")
        with self._lock:
            self._refill_now(float(self._clock()))
            if self._tokens < required:
                return False
            self._tokens -= required
            return True

    def available(self) -> float:
        with self._lock:
            self._refill_now(float(self._clock()))
            return self._tokens


class AdmissionController:
    """Bound in-flight work and distinguish retryable backpressure from hard load shedding."""

    def __init__(
        self,
        *,
        max_inflight: int,
        backpressure_queue_depth: int,
        shed_queue_depth: int,
    ) -> None:
        if isinstance(max_inflight, bool) or max_inflight < 1:
            raise ValueError("max_inflight must be a positive integer")
        if backpressure_queue_depth < 0 or shed_queue_depth < backpressure_queue_depth:
            raise ValueError("queue-depth thresholds are invalid")
        self._max_inflight = int(max_inflight)
        self._backpressure_depth = int(backpressure_queue_depth)
        self._shed_depth = int(shed_queue_depth)
        self._leases: set[str] = set()
        self._lock = threading.Lock()

    def decide(self, *, queue_depth: int) -> AdmissionDecision:
        if isinstance(queue_depth, bool) or queue_depth < 0:
            raise ValueError("queue_depth must be a non-negative integer")
        with self._lock:
            inflight = len(self._leases)
            if queue_depth >= self._shed_depth:
                return AdmissionDecision(
                    AdmissionAction.SHED,
                    "queue_depth_shed_threshold",
                    inflight,
                    queue_depth,
                )
            if inflight >= self._max_inflight:
                return AdmissionDecision(
                    AdmissionAction.BACKPRESSURE,
                    "inflight_limit",
                    inflight,
                    queue_depth,
                )
            if queue_depth >= self._backpressure_depth:
                return AdmissionDecision(
                    AdmissionAction.BACKPRESSURE,
                    "queue_depth_backpressure_threshold",
                    inflight,
                    queue_depth,
                )
            return AdmissionDecision(AdmissionAction.ADMIT, "capacity_available", inflight, queue_depth)

    def acquire(self, *, queue_depth: int) -> tuple[AdmissionDecision, AdmissionLease | None]:
        if isinstance(queue_depth, bool) or queue_depth < 0:
            raise ValueError("queue_depth must be a non-negative integer")
        with self._lock:
            inflight = len(self._leases)
            if queue_depth >= self._shed_depth:
                return (
                    AdmissionDecision(
                        AdmissionAction.SHED,
                        "queue_depth_shed_threshold",
                        inflight,
                        queue_depth,
                    ),
                    None,
                )
            if inflight >= self._max_inflight or queue_depth >= self._backpressure_depth:
                reason = (
                    "inflight_limit"
                    if inflight >= self._max_inflight
                    else "queue_depth_backpressure_threshold"
                )
                return (
                    AdmissionDecision(AdmissionAction.BACKPRESSURE, reason, inflight, queue_depth),
                    None,
                )
            lease = AdmissionLease(secrets.token_hex(16))
            self._leases.add(lease.lease_id)
            return (
                AdmissionDecision(AdmissionAction.ADMIT, "capacity_available", inflight, queue_depth),
                lease,
            )

    def release(self, lease: AdmissionLease) -> bool:
        with self._lock:
            if lease.lease_id not in self._leases:
                return False
            self._leases.remove(lease.lease_id)
            return True

    @property
    def inflight(self) -> int:
        with self._lock:
            return len(self._leases)


__all__ = [
    "AdmissionAction",
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionLease",
    "TokenBucket",
]
