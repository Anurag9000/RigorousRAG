"""Backend-neutral exact task operations over SQLite/PostgreSQL invalidation ledgers."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef, RecomputeTask
from tools.security import normalize_owner_id

_TASK_COLUMNS = "task_id,artifact_kind,artifact_id,created_at,event_sha256,reason,status,attempts,claimed_at,completed_at,error_type"


def _task_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("task_id must be a string")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 256 or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError("task_id is invalid")
    return cleaned


def _row(value: Sequence[Any] | Mapping[str, Any], key: str, index: int) -> Any:
    return value[key] if isinstance(value, Mapping) else value[index]


def _task(row: Sequence[Any] | Mapping[str, Any]) -> RecomputeTask:
    claimed = _row(row, "claimed_at", 8)
    completed = _row(row, "completed_at", 9)
    return RecomputeTask(
        task_id=str(_row(row, "task_id", 0)),
        artifact=DependencyRef(str(_row(row, "artifact_kind", 1)), str(_row(row, "artifact_id", 2))),
        triggering_event_sha256=str(_row(row, "event_sha256", 4)),
        reason=str(_row(row, "reason", 5)),
        status=str(_row(row, "status", 6)),
        attempts=int(_row(row, "attempts", 7)),
        created_at=float(_row(row, "created_at", 3)),
        claimed_at=float(claimed) if claimed is not None else None,
        completed_at=float(completed) if completed is not None else None,
        error_type=str(_row(row, "error_type", 10) or ""),
    )


def _is_sqlite(store: Any) -> bool:
    return hasattr(store, "path")


def _pg_transaction(store: Any, operation):
    transaction = getattr(store, "_transaction", None)
    schema = getattr(store, "schema", None)
    if not callable(transaction) or not isinstance(schema, str) or not schema:
        raise TypeError("unsupported recompute ledger backend")
    return transaction(operation)


def load_recompute_task(store: DependencyInvalidationStore, owner_id: str, task_id: str) -> RecomputeTask:
    owner, identifier = normalize_owner_id(owner_id), _task_id(task_id)
    if _is_sqlite(store):
        connection = sqlite3.connect(str(store.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            row = connection.execute(
                "SELECT task_id,artifact_kind,artifact_id,created_at,event_sha256,reason,status,attempts,claimed_at,completed_at,error_type FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                (owner, identifier),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise KeyError(identifier)
        return _task(row)

    def operation(cursor):
        cursor.execute(
            f"SELECT {_TASK_COLUMNS} FROM {store.schema}.recompute_tasks WHERE owner_id=%s AND task_id=%s",
            (owner, identifier),
        )
        row = cursor.fetchone()
        if row is None:
            raise KeyError(identifier)
        return _task(row)

    return _pg_transaction(store, operation)


def requeue_failed_recompute(store: DependencyInvalidationStore, owner_id: str, task_id: str) -> bool:
    owner, identifier = normalize_owner_id(owner_id), _task_id(task_id)
    if _is_sqlite(store):
        connection = sqlite3.connect(str(store.path), timeout=30.0, isolation_level=None)
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            cursor = connection.execute(
                """UPDATE recompute_tasks SET status='queued',claimed_at=NULL,completed_at=NULL,error_type=''
                   WHERE owner_id=? AND task_id=? AND status='failed'""",
                (owner, identifier),
            )
            return bool(cursor.rowcount)
        finally:
            connection.close()

    def operation(cursor):
        cursor.execute(
            f"""UPDATE {store.schema}.recompute_tasks
                SET status='queued',claimed_at=NULL,completed_at=NULL,error_type=''
                WHERE owner_id=%s AND task_id=%s AND status='failed'""",
            (owner, identifier),
        )
        return bool(cursor.rowcount)

    return _pg_transaction(store, operation)


@dataclass(frozen=True)
class ExactClaimDecision:
    state: str
    task: RecomputeTask | None = None


def claim_exact_recompute_task(
    store: DependencyInvalidationStore,
    owner_id: str,
    task_id: str,
    *,
    max_attempts: int = 5,
    claim_timeout_seconds: float = 900.0,
) -> ExactClaimDecision:
    owner, identifier = normalize_owner_id(owner_id), _task_id(task_id)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 100:
        raise ValueError("max_attempts is invalid")
    if isinstance(claim_timeout_seconds, bool):
        raise ValueError("claim_timeout_seconds is invalid")
    timeout = float(claim_timeout_seconds)
    if not 0.0 < timeout <= 86_400.0:
        raise ValueError("claim_timeout_seconds is invalid")
    now, cutoff = time.time(), time.time() - timeout

    def decide(row: Any, update, reload):
        if row is None:
            return ExactClaimDecision("missing")
        current = _task(row)
        if current.status in {"completed", "failed", "cancelled"}:
            return ExactClaimDecision("terminal", current)
        if current.status == "claimed" and current.claimed_at is not None and current.claimed_at > cutoff:
            return ExactClaimDecision("busy", current)
        if current.status not in {"queued", "claimed"}:
            return ExactClaimDecision("terminal", current)
        if current.attempts >= max_attempts:
            update("failed", now, "ClaimAttemptsExhausted", current.status)
            return ExactClaimDecision("exhausted", reload())
        update("claimed", now, "", current.status)
        return ExactClaimDecision("claimed", reload())

    if _is_sqlite(store):
        connection = sqlite3.connect(str(store.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"SELECT {_TASK_COLUMNS} FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                (owner, identifier),
            ).fetchone()

            def reload():
                value = connection.execute(
                    f"SELECT {_TASK_COLUMNS} FROM recompute_tasks WHERE owner_id=? AND task_id=?",
                    (owner, identifier),
                ).fetchone()
                if value is None:
                    raise RuntimeError("recompute task disappeared during exact claim")
                return _task(value)

            def update(status: str, timestamp: float, error: str, expected: str):
                if status == "claimed":
                    cursor = connection.execute(
                        """UPDATE recompute_tasks SET status='claimed',attempts=attempts+1,claimed_at=?,completed_at=NULL,error_type=''
                           WHERE owner_id=? AND task_id=? AND status=?""",
                        (timestamp, owner, identifier, expected),
                    )
                else:
                    cursor = connection.execute(
                        """UPDATE recompute_tasks SET status='failed',completed_at=?,error_type=?
                           WHERE owner_id=? AND task_id=? AND status=?""",
                        (timestamp, error, owner, identifier, expected),
                    )
                if cursor.rowcount != 1:
                    raise RuntimeError("recompute exact claim lost a concurrent update")

            decision = decide(row, update, reload)
            connection.commit()
            return decision
        except Exception:
            try:
                connection.rollback()
            except Exception:
                pass
            raise
        finally:
            connection.close()

    def operation(cursor):
        cursor.execute(
            f"SELECT {_TASK_COLUMNS} FROM {store.schema}.recompute_tasks WHERE owner_id=%s AND task_id=%s FOR UPDATE",
            (owner, identifier),
        )
        row = cursor.fetchone()

        def reload():
            cursor.execute(
                f"SELECT {_TASK_COLUMNS} FROM {store.schema}.recompute_tasks WHERE owner_id=%s AND task_id=%s",
                (owner, identifier),
            )
            value = cursor.fetchone()
            if value is None:
                raise RuntimeError("recompute task disappeared during exact claim")
            return _task(value)

        def update(status: str, timestamp: float, error: str, expected: str):
            if status == "claimed":
                cursor.execute(
                    f"""UPDATE {store.schema}.recompute_tasks
                        SET status='claimed',attempts=attempts+1,claimed_at=%s,completed_at=NULL,error_type=''
                        WHERE owner_id=%s AND task_id=%s AND status=%s""",
                    (timestamp, owner, identifier, expected),
                )
            else:
                cursor.execute(
                    f"""UPDATE {store.schema}.recompute_tasks SET status='failed',completed_at=%s,error_type=%s
                        WHERE owner_id=%s AND task_id=%s AND status=%s""",
                    (timestamp, error, owner, identifier, expected),
                )
            if cursor.rowcount != 1:
                raise RuntimeError("recompute exact claim lost a concurrent update")

        return decide(row, update, reload)

    return _pg_transaction(store, operation)


__all__ = [
    "ExactClaimDecision",
    "claim_exact_recompute_task",
    "load_recompute_task",
    "requeue_failed_recompute",
]
