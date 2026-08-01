"""Runtime factories and atomic claims for migration shadow validation only."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from tools.migration_journal import MigrationJournal
from tools.migration_runtime import get_migration_journal
from tools.migration_shadow_builder import MigrationShadowBuilder
from tools.migration_shadow_executor import ShadowExecutionResult, execute_shadow_task
from tools.migration_shadow_store import MigrationShadowStore
from tools.migration_types import MigrationTask, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_LOCK = threading.RLock()
_STORES: dict[str, MigrationShadowStore] = {}


def _root(value: str | os.PathLike[str] | None = None) -> str:
    selected = value if value is not None else os.getenv(
        "MIGRATION_SHADOW_ROOT",
        "data/migration_shadows",
    )
    rendered = os.fspath(selected)
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return str(candidate.absolute())


def get_migration_shadow_store(
    root: str | os.PathLike[str] | None = None,
) -> MigrationShadowStore:
    selected = _root(root)
    with _LOCK:
        store = _STORES.get(selected)
        if store is None:
            store = MigrationShadowStore(selected)
            _STORES[selected] = store
        else:
            store._verify_root()
        return store


def clear_migration_shadow_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


def claim_shadow_build_task(
    journal: MigrationJournal,
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int = 300,
    max_attempts: int = 3,
    now: float | None = None,
) -> MigrationTask | None:
    """Atomically claim only work that still needs a shadow build."""

    if not isinstance(journal, MigrationJournal):
        raise ValueError("journal must be a MigrationJournal.")
    owner = normalize_owner_id(owner_id)
    worker = identifier(worker_id, "worker_id", 128)
    lease = exact_integer(lease_seconds, "lease_seconds", 1, 86_400)
    attempts = exact_integer(max_attempts, "max_attempts", 1, 100)
    current = timestamp(time.time() if now is None else now)
    expires = current + lease
    with journal._lock, journal._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            row = connection.execute(
                """
                SELECT * FROM migration_tasks
                WHERE owner_id = ? AND (
                    state = 'planned'
                    OR (state = 'failed' AND attempt < ?)
                    OR (
                        state = 'running'
                        AND lease_expires_at <= ?
                        AND attempt < ?
                    )
                )
                ORDER BY
                    CASE state
                        WHEN 'running' THEN 0
                        WHEN 'failed' THEN 1
                        ELSE 2
                    END,
                    created_at,
                    task_id
                LIMIT 1
                """,
                (owner, attempts, current, attempts),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            new_attempt = int(row["attempt"]) + 1
            connection.execute(
                """
                UPDATE migration_tasks
                SET state='running', attempt=?, updated_at=?,
                    lease_owner=?, lease_expires_at=?,
                    validation_digest=NULL, failure_type=NULL
                WHERE task_id=?
                """,
                (
                    new_attempt,
                    current,
                    worker,
                    expires,
                    row["task_id"],
                ),
            )
            claimed = connection.execute(
                "SELECT * FROM migration_tasks WHERE task_id=?",
                (row["task_id"],),
            ).fetchone()
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
    return journal._task(claimed)


def execute_next_shadow_build(
    *,
    owner_id: str,
    worker_id: str,
    lease_seconds: int = 300,
    max_attempts: int = 3,
    journal: MigrationJournal | None = None,
    shadows: MigrationShadowStore | None = None,
    builder: Any = None,
    now: float | None = None,
) -> ShadowExecutionResult | None:
    selected_journal = journal or get_migration_journal()
    selected_shadows = shadows or get_migration_shadow_store()
    selected_builder = builder or MigrationShadowBuilder()
    task = claim_shadow_build_task(
        selected_journal,
        owner_id=owner_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
        now=now,
    )
    if task is None:
        return None
    from tools.sparse_runtime import get_generation_store

    return execute_shadow_task(
        task,
        worker_id=worker_id,
        journal=selected_journal,
        generations=get_generation_store(),
        shadows=selected_shadows,
        builder=selected_builder,
        now=now,
    )


__all__ = [
    "claim_shadow_build_task",
    "clear_migration_shadow_store_cache",
    "execute_next_shadow_build",
    "get_migration_shadow_store",
]
