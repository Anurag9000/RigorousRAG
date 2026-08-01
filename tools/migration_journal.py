"""Durable resumable migration task journal with expiring worker leases."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Iterable

from tools.migration_planner import migration_task_id
from tools.migration_types import (
    MigrationCandidate,
    MigrationTask,
    digest,
    exact_integer,
    identifier,
    timestamp,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4_096
_MAX_TASKS = 100_000


def _redirecting(value: os.stat_result) -> bool:
    return stat.S_ISLNK(value.st_mode) or bool(
        int(getattr(value, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("migration journal path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered)
    ):
        raise ValueError("migration journal path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("migration journal path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("migration journal path may not contain redirects.")
    return absolute


class MigrationJournal:
    """SQLite task journal; retained source paths are intentionally never stored."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("migration journal parent must be a regular directory.")
        self._parent_identity = (parent.st_dev, parent.st_ino)
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._database_file_identity()

    def _database_file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("migration journal is not a regular file.")
        return info.st_dev, info.st_ino

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or (parent.st_dev, parent.st_ino) != self._parent_identity
        ):
            raise RuntimeError("migration journal parent identity changed.")
        if self._database_file_identity() != self._database_identity:
            raise RuntimeError("migration journal identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        ) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS migration_tasks (
                    task_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    source_sequence INTEGER NOT NULL,
                    source_profile_fingerprint TEXT NOT NULL,
                    target_profile_name TEXT NOT NULL,
                    target_profile_fingerprint TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    validation_digest TEXT,
                    failure_type TEXT,
                    schema_version INTEGER NOT NULL,
                    UNIQUE (
                        owner_id,
                        doc_id,
                        source_sequence,
                        target_profile_fingerprint
                    )
                );
                CREATE INDEX IF NOT EXISTS migration_tasks_claim
                    ON migration_tasks(owner_id, state, lease_expires_at, created_at);
                """
            )

    @staticmethod
    def _task(row: sqlite3.Row) -> MigrationTask:
        if int(row["schema_version"]) != 1:
            raise RuntimeError("migration task schema is unsupported.")
        return MigrationTask(
            task_id=row["task_id"],
            owner_id=row["owner_id"],
            doc_id=row["doc_id"],
            source_sequence=row["source_sequence"],
            source_profile_fingerprint=row["source_profile_fingerprint"],
            target_profile_name=row["target_profile_name"],
            target_profile_fingerprint=row["target_profile_fingerprint"],
            state=row["state"],
            attempt=row["attempt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            validation_digest=row["validation_digest"],
            failure_type=row["failure_type"],
        )

    def seed(
        self,
        candidates: Iterable[MigrationCandidate],
        *,
        now: float | None = None,
    ) -> tuple[MigrationTask, ...]:
        current = timestamp(time.time() if now is None else now)
        if isinstance(candidates, (str, bytes, bytearray)):
            raise ValueError("candidates must be an iterable.")
        values: list[MigrationCandidate] = []
        try:
            for candidate in candidates:
                if len(values) >= _MAX_TASKS:
                    raise ValueError("migration seed exceeds the task limit.")
                if not isinstance(candidate, MigrationCandidate):
                    raise ValueError("every candidate must be a MigrationCandidate.")
                if candidate.eligible:
                    values.append(candidate)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("candidates are not safely iterable.") from exc
        tasks: list[MigrationTask] = []
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for candidate in values:
                    task_id = migration_task_id(candidate)
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO migration_tasks VALUES
                        (?, ?, ?, ?, ?, ?, ?, 'planned', 0, ?, ?,
                         NULL, NULL, NULL, NULL, 1)
                        """,
                        (
                            task_id,
                            candidate.owner_id,
                            candidate.doc_id,
                            candidate.source_sequence,
                            candidate.source_profile_fingerprint,
                            candidate.target_profile_name,
                            candidate.target_profile_fingerprint,
                            current,
                            current,
                        ),
                    )
                    row = connection.execute(
                        "SELECT * FROM migration_tasks WHERE task_id = ?",
                        (task_id,),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("migration task could not be persisted.")
                    task = self._task(row)
                    immutable = (
                        task.owner_id,
                        task.doc_id,
                        task.source_sequence,
                        task.source_profile_fingerprint,
                        task.target_profile_name,
                        task.target_profile_fingerprint,
                    )
                    expected = (
                        candidate.owner_id,
                        candidate.doc_id,
                        candidate.source_sequence,
                        candidate.source_profile_fingerprint,
                        candidate.target_profile_name,
                        candidate.target_profile_fingerprint,
                    )
                    if immutable != expected:
                        raise RuntimeError("migration task identity collision detected.")
                    tasks.append(task)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return tuple(tasks)

    def get(self, task_id: str) -> MigrationTask | None:
        identifier_value = identifier(task_id, "task_id", 64)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM migration_tasks WHERE task_id = ?",
                (identifier_value,),
            ).fetchone()
        return None if row is None else self._task(row)

    def list_tasks(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 1_000,
    ) -> tuple[MigrationTask, ...]:
        owner = normalize_owner_id(owner_id)
        count = exact_integer(limit, "limit", 1, 10_000)
        params: list[object] = [owner]
        query = "SELECT * FROM migration_tasks WHERE owner_id = ?"
        if state is not None:
            state_value = identifier(state, "state", 20)
            if state_value not in {
                "planned",
                "running",
                "validated",
                "committed",
                "failed",
                "cancelled",
            }:
                raise ValueError("state is invalid.")
            query += " AND state = ?"
            params.append(state_value)
        query += " ORDER BY created_at, task_id LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._task(row) for row in rows)

    def claim(
        self,
        *,
        owner_id: str,
        worker_id: str,
        lease_seconds: int = 300,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> MigrationTask | None:
        owner = normalize_owner_id(owner_id)
        worker = identifier(worker_id, "worker_id", 128)
        lease = exact_integer(lease_seconds, "lease_seconds", 1, 86_400)
        attempts = exact_integer(max_attempts, "max_attempts", 1, 100)
        current = timestamp(time.time() if now is None else now)
        expires = current + lease
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT * FROM migration_tasks
                    WHERE owner_id = ? AND (
                        state = 'planned'
                        OR (state = 'failed' AND attempt < ?)
                        OR (
                            state IN ('running', 'validated')
                            AND lease_expires_at <= ?
                            AND attempt < ?
                        )
                    )
                    ORDER BY
                        CASE state
                            WHEN 'validated' THEN 0
                            WHEN 'running' THEN 1
                            WHEN 'failed' THEN 2
                            ELSE 3
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
                old_state = str(row["state"])
                new_state = "validated" if old_state == "validated" else "running"
                new_attempt = int(row["attempt"]) + (0 if old_state == "validated" else 1)
                connection.execute(
                    """
                    UPDATE migration_tasks
                    SET state = ?, attempt = ?, updated_at = ?,
                        lease_owner = ?, lease_expires_at = ?,
                        failure_type = CASE WHEN ? = 'running' THEN NULL ELSE failure_type END,
                        validation_digest = CASE
                            WHEN ? = 'running' THEN NULL ELSE validation_digest END
                    WHERE task_id = ?
                    """,
                    (
                        new_state,
                        new_attempt,
                        current,
                        worker,
                        expires,
                        new_state,
                        new_state,
                        row["task_id"],
                    ),
                )
                claimed = connection.execute(
                    "SELECT * FROM migration_tasks WHERE task_id = ?",
                    (row["task_id"],),
                ).fetchone()
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self._task(claimed)

    def renew(
        self,
        *,
        task_id: str,
        worker_id: str,
        lease_seconds: int = 300,
        now: float | None = None,
    ) -> MigrationTask:
        task = identifier(task_id, "task_id", 64)
        worker = identifier(worker_id, "worker_id", 128)
        lease = exact_integer(lease_seconds, "lease_seconds", 1, 86_400)
        current = timestamp(time.time() if now is None else now)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE migration_tasks
                SET updated_at = ?, lease_expires_at = ?
                WHERE task_id = ? AND lease_owner = ?
                  AND state IN ('running', 'validated')
                  AND lease_expires_at > ?
                """,
                (current, current + lease, task, worker, current),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("migration lease is unavailable or expired.")
        result = self.get(task)
        if result is None:
            raise RuntimeError("migration task disappeared after lease renewal.")
        return result

    def _transition(
        self,
        *,
        task_id: str,
        worker_id: str,
        expected_state: str,
        new_state: str,
        now: float | None,
        validation_digest: str | None = None,
        failure_type: str | None = None,
        clear_lease: bool = False,
    ) -> MigrationTask:
        task = identifier(task_id, "task_id", 64)
        worker = identifier(worker_id, "worker_id", 128)
        current = timestamp(time.time() if now is None else now)
        validation = (
            None
            if validation_digest is None
            else digest(validation_digest, "validation_digest")
        )
        failure = (
            None
            if failure_type is None
            else identifier(failure_type, "failure_type", 200)
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE migration_tasks
                SET state = ?, updated_at = ?, validation_digest = ?,
                    failure_type = ?,
                    lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
                    lease_expires_at = CASE WHEN ? THEN NULL ELSE lease_expires_at END
                WHERE task_id = ? AND state = ? AND lease_owner = ?
                  AND lease_expires_at > ?
                """,
                (
                    new_state,
                    current,
                    validation,
                    failure,
                    int(clear_lease),
                    int(clear_lease),
                    task,
                    expected_state,
                    worker,
                    current,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("migration state or lease changed concurrently.")
        result = self.get(task)
        if result is None:
            raise RuntimeError("migration task disappeared after transition.")
        return result

    def mark_validated(
        self,
        *,
        task_id: str,
        worker_id: str,
        validation_digest: str,
        now: float | None = None,
    ) -> MigrationTask:
        return self._transition(
            task_id=task_id,
            worker_id=worker_id,
            expected_state="running",
            new_state="validated",
            validation_digest=validation_digest,
            now=now,
        )

    def mark_committed(
        self,
        *,
        task_id: str,
        worker_id: str,
        now: float | None = None,
    ) -> MigrationTask:
        current = self.get(task_id)
        if current is None or current.validation_digest is None:
            raise RuntimeError("migration validation is unavailable.")
        return self._transition(
            task_id=task_id,
            worker_id=worker_id,
            expected_state="validated",
            new_state="committed",
            validation_digest=current.validation_digest,
            now=now,
            clear_lease=True,
        )

    def mark_failed(
        self,
        *,
        task_id: str,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> MigrationTask:
        current = self.get(task_id)
        if current is None or current.state not in {"running", "validated"}:
            raise RuntimeError("migration task is not active.")
        return self._transition(
            task_id=task_id,
            worker_id=worker_id,
            expected_state=current.state,
            new_state="failed",
            failure_type=failure_type,
            now=now,
            clear_lease=True,
        )

    def cancel(self, *, task_id: str, now: float | None = None) -> MigrationTask:
        task = identifier(task_id, "task_id", 64)
        current = timestamp(time.time() if now is None else now)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE migration_tasks
                SET state = 'cancelled', updated_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE task_id = ? AND state IN ('planned', 'failed')
                """,
                (current, task),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("only planned or failed tasks may be cancelled.")
        result = self.get(task)
        if result is None:
            raise RuntimeError("migration task disappeared after cancellation.")
        return result


__all__ = ["MigrationJournal"]
