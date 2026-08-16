"""Durable transport handoff for owner-scoped recomputation.

The dependency invalidation ledger remains authoritative. Queue payloads contain only one
opaque ``task_id``; no query, evidence, citation, project, owner, or artifact content is
serialized into transport. Exact claims are backend-neutral across SQLite/PostgreSQL and
execution may be routed by the authoritative task kind.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

from tools.dependency_invalidation import DependencyInvalidationStore, RecomputeTask
from tools.durable_queue import ClaimedMessage, DurableQueue
from tools.recompute_executor import RecomputeOutcome
from tools.recompute_ledger_ops import ExactClaimDecision, claim_exact_recompute_task
from tools.security import normalize_owner_id

_MAX_TASK_ID = 256
_MAX_WORKER_ID = 256
WorkState = Literal["idle", "completed", "failed", "busy", "duplicate", "invalid"]


def _identifier(value: str, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _positive_seconds(value: float, label: str, *, maximum: float = 86_400.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    seconds = float(value)
    if not 0.0 < seconds <= maximum:
        raise ValueError(f"{label} is invalid")
    return seconds


class ClaimedTaskExecutor(Protocol):
    def execute_claimed(self, owner_id: str, task_id: str) -> RecomputeOutcome: ...


@dataclass(frozen=True)
class DistributedRecomputeResult:
    state: WorkState
    task_id: str = ""
    outcome: RecomputeOutcome | None = None


class DistributedRecomputeBridge:
    """Publish and execute opaque recompute handoffs for one research owner."""

    def __init__(
        self,
        *,
        owner_id: str,
        invalidations: DependencyInvalidationStore,
        executor: ClaimedTaskExecutor,
        queue: DurableQueue,
        max_attempts: int = 5,
        claim_timeout_seconds: float = 900.0,
    ) -> None:
        self.owner_id = normalize_owner_id(owner_id)
        self.invalidations = invalidations
        if not callable(getattr(executor, "execute_claimed", None)):
            raise TypeError("executor must expose execute_claimed")
        self.executor = executor
        self.queue = queue
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts is invalid")
        self.max_attempts = max_attempts
        self.claim_timeout_seconds = _positive_seconds(claim_timeout_seconds, "claim_timeout_seconds")

    def _idempotency_key(self, task: RecomputeTask) -> str:
        digest = hashlib.sha256(
            f"{self.owner_id}\x1f{task.task_id}\x1f{task.attempts}".encode("utf-8")
        ).hexdigest()
        return f"research-recompute:{digest}"

    def publish_ready(self, *, limit: int = 1000) -> tuple[str, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        tasks = self.invalidations.list_recompute(self.owner_id, status="queued", limit=limit)
        message_ids: list[str] = []
        for task in reversed(tasks):
            if task.attempts >= self.max_attempts:
                continue
            message_ids.append(
                self.queue.enqueue(
                    {"task_id": task.task_id},
                    idempotency_key=self._idempotency_key(task),
                )
            )
        return tuple(message_ids)

    @staticmethod
    def _task_id_from_message(message: ClaimedMessage) -> str:
        payload = message.payload
        if set(payload) != {"task_id"}:
            raise ValueError("recompute queue payload must contain only task_id")
        return _identifier(payload.get("task_id"), "task_id", _MAX_TASK_ID)

    def work_one(
        self,
        *,
        worker_id: str,
        visibility_timeout: float = 1800.0,
        busy_retry_delay: float = 30.0,
    ) -> DistributedRecomputeResult:
        worker = _identifier(worker_id, "worker_id", _MAX_WORKER_ID)
        visibility = _positive_seconds(visibility_timeout, "visibility_timeout")
        if isinstance(busy_retry_delay, bool):
            raise ValueError("busy_retry_delay is invalid")
        retry_delay = float(busy_retry_delay)
        if not 0.0 <= retry_delay <= 86_400.0:
            raise ValueError("busy_retry_delay is invalid")
        message = self.queue.claim(worker, visibility_timeout=visibility)
        if message is None:
            return DistributedRecomputeResult("idle")
        try:
            task_id = self._task_id_from_message(message)
        except Exception:
            self.queue.nack(message.receipt, retry_delay=retry_delay)
            return DistributedRecomputeResult("invalid")

        decision: ExactClaimDecision = claim_exact_recompute_task(
            self.invalidations,
            self.owner_id,
            task_id,
            max_attempts=self.max_attempts,
            claim_timeout_seconds=self.claim_timeout_seconds,
        )
        if decision.state == "busy":
            self.queue.nack(message.receipt, retry_delay=retry_delay)
            return DistributedRecomputeResult("busy", task_id)
        if decision.state in {"terminal", "exhausted"}:
            self.queue.ack(message.receipt)
            return DistributedRecomputeResult("duplicate", task_id)
        if decision.state == "missing" or decision.task is None:
            self.queue.nack(message.receipt, retry_delay=retry_delay)
            return DistributedRecomputeResult("invalid", task_id)

        try:
            outcome = self.executor.execute_claimed(self.owner_id, task_id)
        except Exception:
            # Execution did not settle the authoritative task. Retain the transport handoff
            # so an expired/stale claim may be recovered according to ledger policy.
            self.queue.nack(message.receipt, retry_delay=retry_delay)
            raise
        self.queue.ack(message.receipt)
        return DistributedRecomputeResult(
            "completed" if outcome.success else "failed",
            task_id,
            outcome,
        )


__all__ = [
    "ClaimedTaskExecutor",
    "DistributedRecomputeBridge",
    "DistributedRecomputeResult",
    "ExactClaimDecision",
    "claim_exact_recompute_task",
]
