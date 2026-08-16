"""Durable transport handoff for owner-scoped research recomputation.

The dependency invalidation database remains authoritative. The queue is transport
only and receives a single opaque ``task_id`` field: no query text, evidence, citation,
artifact content, or tenant identifier is serialized into a message payload.

A bridge instance is intentionally bound to exactly one normalized research owner and
one queue namespace. Workers re-claim the referenced task in the authoritative ledger
before any handler runs. Fresh competing claims are retried rather than acknowledged;
terminal duplicate deliveries are acknowledged as harmless no-ops.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from typing import Literal

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef, RecomputeTask
from tools.durable_queue import ClaimedMessage, DurableQueue
from tools.recompute_executor import RecomputeOutcome, ResearchRecomputeExecutor
from tools.security import normalize_owner_id

_MAX_TASK_ID = 256
_MAX_WORKER_ID = 256

ClaimState = Literal["claimed", "busy", "terminal", "missing", "exhausted"]
WorkState = Literal[
    "idle",
    "completed",
    "failed",
    "busy",
    "duplicate",
    "invalid",
]


def _identifier(value: str, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid")
    return cleaned


def _positive_seconds(value: float, label: str, *, maximum: float = 86_400.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is invalid")
    seconds = float(value)
    if not 0.0 < seconds <= maximum:
        raise ValueError(f"{label} is invalid")
    return seconds


def _task_from_row(row: sqlite3.Row) -> RecomputeTask:
    return RecomputeTask(
        task_id=str(row["task_id"]),
        artifact=DependencyRef(str(row["artifact_kind"]), str(row["artifact_id"])),
        triggering_event_sha256=str(row["event_sha256"]),
        reason=str(row["reason"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        created_at=float(row["created_at"]),
        claimed_at=float(row["claimed_at"]) if row["claimed_at"] is not None else None,
        completed_at=float(row["completed_at"]) if row["completed_at"] is not None else None,
        error_type=str(row["error_type"] or ""),
    )


@dataclass(frozen=True)
class ExactClaimDecision:
    state: ClaimState
    task: RecomputeTask | None = None


@dataclass(frozen=True)
class DistributedRecomputeResult:
    state: WorkState
    task_id: str = ""
    outcome: RecomputeOutcome | None = None


def claim_exact_recompute_task(
    store: DependencyInvalidationStore,
    owner_id: str,
    task_id: str,
    *,
    max_attempts: int = 5,
    claim_timeout_seconds: float = 900.0,
) -> ExactClaimDecision:
    """Atomically claim exactly ``task_id`` from the owner-scoped ledger.

    A stale claim may be recovered after ``claim_timeout_seconds``. The attempt counter
    is incremented for every successful claim/recovery. Exhausted queued/stale tasks
    are transitioned to ``failed`` instead of remaining permanently unclaimable.
    """

    owner = normalize_owner_id(owner_id)
    task = _identifier(task_id, "task_id", _MAX_TASK_ID)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 100:
        raise ValueError("max_attempts is invalid")
    timeout = _positive_seconds(claim_timeout_seconds, "claim_timeout_seconds")
    now = time.time()
    cutoff = now - timeout
    connection = sqlite3.connect(str(store.path), timeout=30.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM recompute_tasks WHERE owner_id=? AND task_id=?",
            (owner, task),
        ).fetchone()
        if row is None:
            connection.commit()
            return ExactClaimDecision("missing")

        status = str(row["status"])
        attempts = int(row["attempts"])
        claimed_at = float(row["claimed_at"]) if row["claimed_at"] is not None else None
        if status in {"completed", "failed", "cancelled"}:
            connection.commit()
            return ExactClaimDecision("terminal", _task_from_row(row))
        if status == "claimed" and claimed_at is not None and claimed_at > cutoff:
            connection.commit()
            return ExactClaimDecision("busy", _task_from_row(row))
        if status not in {"queued", "claimed"}:
            connection.commit()
            return ExactClaimDecision("terminal", _task_from_row(row))
        if attempts >= max_attempts:
            connection.execute(
                """UPDATE recompute_tasks
                   SET status='failed',completed_at=?,error_type='ClaimAttemptsExhausted'
                   WHERE owner_id=? AND task_id=? AND status IN ('queued','claimed')""",
                (now, owner, task),
            )
            failed = connection.execute(
                "SELECT * FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                (owner, task),
            ).fetchone()
            connection.commit()
            return ExactClaimDecision("exhausted", _task_from_row(failed))

        cursor = connection.execute(
            """UPDATE recompute_tasks
               SET status='claimed',attempts=attempts+1,claimed_at=?,completed_at=NULL,error_type=''
               WHERE owner_id=? AND task_id=? AND status=?""",
            (now, owner, task, status),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return ExactClaimDecision("busy")
        claimed = connection.execute(
            "SELECT * FROM recompute_tasks WHERE owner_id=? AND task_id=?",
            (owner, task),
        ).fetchone()
        connection.commit()
        return ExactClaimDecision("claimed", _task_from_row(claimed))
    except Exception:
        try:
            connection.rollback()
        except Exception:
            pass
        raise
    finally:
        connection.close()


class DistributedRecomputeBridge:
    """Publish and execute opaque recompute handoffs for one research owner."""

    def __init__(
        self,
        *,
        owner_id: str,
        invalidations: DependencyInvalidationStore,
        executor: ResearchRecomputeExecutor,
        queue: DurableQueue,
        max_attempts: int = 5,
        claim_timeout_seconds: float = 900.0,
    ) -> None:
        self.owner_id = normalize_owner_id(owner_id)
        self.invalidations = invalidations
        self.executor = executor
        self.queue = queue
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts is invalid")
        self.max_attempts = max_attempts
        self.claim_timeout_seconds = _positive_seconds(
            claim_timeout_seconds,
            "claim_timeout_seconds",
        )

    def _idempotency_key(self, task: RecomputeTask) -> str:
        """Bind one transport handoff to one authoritative queued attempt epoch.

        Repeated publication while a task remains queued at the same attempt count is
        idempotent. An explicit retry of a failed task preserves the now-incremented
        ledger attempt counter, producing a new handoff instead of resolving to the
        already-acknowledged transport record from the previous execution.
        """

        digest = hashlib.sha256(
            f"{self.owner_id}\x1f{task.task_id}\x1f{task.attempts}".encode("utf-8")
        ).hexdigest()
        return f"research-recompute:{digest}"

    def publish_ready(self, *, limit: int = 1000) -> tuple[str, ...]:
        """Idempotently publish currently queued authoritative tasks."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        tasks = self.invalidations.list_recompute(
            self.owner_id,
            status="queued",
            limit=limit,
        )
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
        value = payload.get("task_id")
        if not isinstance(value, str):
            raise ValueError("recompute task_id must be a string")
        return _identifier(value, "task_id", _MAX_TASK_ID)

    def work_one(
        self,
        *,
        worker_id: str,
        visibility_timeout: float = 1800.0,
        busy_retry_delay: float = 30.0,
    ) -> DistributedRecomputeResult:
        """Lease one handoff, exact-claim it, execute it, and settle transport state."""

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

        decision = claim_exact_recompute_task(
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
        if decision.state == "missing":
            self.queue.nack(message.receipt, retry_delay=retry_delay)
            return DistributedRecomputeResult("invalid", task_id)
        if decision.task is None:
            self.queue.nack(message.receipt, retry_delay=retry_delay)
            return DistributedRecomputeResult("invalid", task_id)

        outcome = self.executor.execute_claimed(self.owner_id, task_id)
        self.queue.ack(message.receipt)
        return DistributedRecomputeResult(
            "completed" if outcome.success else "failed",
            task_id,
            outcome,
        )


__all__ = [
    "DistributedRecomputeBridge",
    "DistributedRecomputeResult",
    "ExactClaimDecision",
    "claim_exact_recompute_task",
]
