"""Bounded retry, partial-failure execution, and caller-priced agent accounting."""

from __future__ import annotations

import math
import time
from concurrent.futures import CancelledError, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from tools.bounded_pool import BoundedExecutor


class AgentFailureKind(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    TRANSIENT = "transient"
    CANCELLED = "cancelled"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if not 1 <= self.max_attempts <= 20:
            raise ValueError("max_attempts must be between 1 and 20")
        for value, label in (
            (self.base_delay_seconds, "base_delay_seconds"),
            (self.max_delay_seconds, "max_delay_seconds"),
            (self.multiplier, "multiplier"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def delay_for_retry(self, completed_attempts: int) -> float:
        if completed_attempts <= 0:
            raise ValueError("completed_attempts must be positive")
        return min(
            self.base_delay_seconds * (self.multiplier ** (completed_attempts - 1)),
            self.max_delay_seconds,
        )


def classify_agent_failure(exc: BaseException) -> AgentFailureKind:
    """Classify common provider/tool failures without depending on provider SDKs."""

    if isinstance(exc, CancelledError):
        return AgentFailureKind.CANCELLED
    if isinstance(exc, (TimeoutError, FutureTimeoutError)):
        return AgentFailureKind.TIMEOUT
    status = getattr(exc, "status_code", None)
    if status == 429:
        return AgentFailureKind.RATE_LIMITED
    if isinstance(status, int) and 500 <= status <= 599:
        return AgentFailureKind.TRANSIENT
    if isinstance(exc, (ConnectionError, BrokenPipeError)):
        return AgentFailureKind.TRANSIENT
    return AgentFailureKind.PERMANENT


_RETRYABLE = {
    AgentFailureKind.TIMEOUT,
    AgentFailureKind.RATE_LIMITED,
    AgentFailureKind.TRANSIENT,
}


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    failure_kind: Optional[AgentFailureKind]
    elapsed_seconds: float


@dataclass(frozen=True)
class RetryResult:
    value: Any = None
    attempts: tuple[AttemptRecord, ...] = ()
    error: Optional[BaseException] = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


def execute_with_retry(
    operation: Callable[[], Any],
    *,
    policy: RetryPolicy = RetryPolicy(),
    classifier: Callable[[BaseException], AgentFailureKind] = classify_agent_failure,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> RetryResult:
    """Execute a synchronous operation with bounded retries and auditable attempts."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    records: list[AttemptRecord] = []
    for attempt in range(1, policy.max_attempts + 1):
        started = clock()
        try:
            value = operation()
        except BaseException as exc:  # preserve the original provider/tool exception
            kind = classifier(exc)
            elapsed = max(clock() - started, 0.0)
            records.append(AttemptRecord(attempt, kind, elapsed))
            if kind not in _RETRYABLE or attempt >= policy.max_attempts:
                return RetryResult(attempts=tuple(records), error=exc)
            sleeper(policy.delay_for_retry(attempt))
            continue
        elapsed = max(clock() - started, 0.0)
        records.append(AttemptRecord(attempt, None, elapsed))
        return RetryResult(value=value, attempts=tuple(records))
    raise AssertionError("unreachable retry loop")


@dataclass(frozen=True)
class AgentTask:
    task_id: str
    operation: Callable[[], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be non-empty")
        if not callable(self.operation):
            raise TypeError("operation must be callable")


@dataclass(frozen=True)
class AgentTaskOutcome:
    task_id: str
    status: str
    value: Any = None
    attempts: tuple[AttemptRecord, ...] = ()
    failure_kind: Optional[AgentFailureKind] = None
    error_type: Optional[str] = None


class BoundedAgentRunner:
    """Run independent agent/tool tasks concurrently while retaining partial success."""

    def __init__(self, *, max_workers: int = 4, max_pending: int = 16) -> None:
        self._pool = BoundedExecutor(
            max_workers=max_workers,
            max_pending=max_pending,
            thread_name_prefix="agent-runner",
        )

    def run(
        self,
        tasks: Sequence[AgentTask],
        *,
        retry_policy: RetryPolicy = RetryPolicy(),
        timeout_seconds: Optional[float] = None,
    ) -> tuple[AgentTaskOutcome, ...]:
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        seen: set[str] = set()
        submitted: list[tuple[AgentTask, Optional[Future[RetryResult]]]] = []
        for task in tasks:
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id: {task.task_id}")
            seen.add(task.task_id)
            future = self._pool.submit(
                execute_with_retry,
                task.operation,
                policy=retry_policy,
            )
            submitted.append((task, future))

        outcomes: list[AgentTaskOutcome] = []
        for task, future in submitted:
            if future is None:
                outcomes.append(AgentTaskOutcome(task.task_id, "rejected"))
                continue
            try:
                result = future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                future.cancel()
                outcomes.append(
                    AgentTaskOutcome(
                        task.task_id,
                        "timeout",
                        failure_kind=AgentFailureKind.TIMEOUT,
                        error_type="TimeoutError",
                    )
                )
                continue
            if result.succeeded:
                outcomes.append(
                    AgentTaskOutcome(task.task_id, "succeeded", result.value, result.attempts)
                )
            else:
                kind = result.attempts[-1].failure_kind if result.attempts else None
                outcomes.append(
                    AgentTaskOutcome(
                        task.task_id,
                        "failed",
                        attempts=result.attempts,
                        failure_kind=kind,
                        error_type=type(result.error).__name__ if result.error is not None else None,
                    )
                )
        return tuple(outcomes)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True)
class UsageRecord:
    agent: str
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 1
    latency_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not self.agent.strip():
            raise ValueError("agent must be non-empty")
        for value, label in (
            (self.input_tokens, "input_tokens"),
            (self.output_tokens, "output_tokens"),
            (self.requests, "requests"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative and finite")


@dataclass(frozen=True)
class Pricing:
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    per_request: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.input_per_million, self.output_per_million, self.per_request):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("pricing values must be non-negative and finite")


@dataclass
class AgentCostLedger:
    """Aggregate usage using only explicit caller-supplied pricing."""

    pricing: Mapping[str, Pricing] = field(default_factory=dict)
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    def summarize(self) -> dict[str, Any]:
        by_agent: dict[str, dict[str, float]] = {}
        for record in self.records:
            bucket = by_agent.setdefault(
                record.agent,
                {"input_tokens": 0.0, "output_tokens": 0.0, "requests": 0.0, "latency_seconds": 0.0, "cost": 0.0},
            )
            bucket["input_tokens"] += record.input_tokens
            bucket["output_tokens"] += record.output_tokens
            bucket["requests"] += record.requests
            bucket["latency_seconds"] += record.latency_seconds
            price = self.pricing.get(record.agent)
            if price is not None:
                bucket["cost"] += (
                    record.input_tokens * price.input_per_million / 1_000_000
                    + record.output_tokens * price.output_per_million / 1_000_000
                    + record.requests * price.per_request
                )
        total_cost = sum(item["cost"] for item in by_agent.values())
        total_latency = sum(item["latency_seconds"] for item in by_agent.values())
        return {"agents": by_agent, "total_cost": total_cost, "coordination_latency_seconds": total_latency}
