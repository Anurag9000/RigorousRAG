"""Durable leased journal for non-executing migration cutover preparation."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.migration_cutover_control import CutoverOperation, CutoverPreparation
from tools.migration_types import exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_JSON = 100_000


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("cutover journal path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("cutover journal path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("cutover journal path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("cutover journal path may not contain redirects.")
    return absolute


def _encoded(preparation: CutoverPreparation) -> str:
    payload = json.dumps(
        asdict(preparation),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(payload.encode("utf-8")) > _MAX_JSON:
        raise ValueError("cutover preparation exceeds the journal byte limit.")
    return payload


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _decoded(value: Any) -> CutoverPreparation:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAX_JSON:
        raise RuntimeError("stored cutover preparation is corrupt.")
    try:
        raw = json.loads(
            value,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
        if not isinstance(raw, dict):
            raise ValueError("not object")
        return CutoverPreparation(**raw)
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise RuntimeError("stored cutover preparation is corrupt.") from exc


class MigrationCutoverJournal:
    """Preparation-only journal; it intentionally has no executing/committed state."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("cutover journal parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("cutover journal is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("cutover journal parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("cutover journal identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS cutover_operations (
                    operation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    preparation_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    failure_type TEXT,
                    schema_version INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS cutover_task_operation
                    ON cutover_operations(task_id, operation_id);
                CREATE INDEX IF NOT EXISTS cutover_claim
                    ON cutover_operations(owner_id, state, lease_expires_at, created_at);
                """
            )

    @staticmethod
    def _operation(row: sqlite3.Row) -> CutoverOperation:
        if int(row["schema_version"]) != 1:
            raise RuntimeError("cutover journal schema is unsupported.")
        return CutoverOperation(
            operation_id=row["operation_id"],
            preparation=_decoded(row["preparation_json"]),
            state=row["state"],
            attempt=row["attempt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            failure_type=row["failure_type"],
        )

    def seed(
        self,
        preparation: CutoverPreparation,
        *,
        now: float | None = None,
    ) -> CutoverOperation:
        if not isinstance(preparation, CutoverPreparation):
            raise ValueError("preparation must be CutoverPreparation.")
        current = timestamp(time.time() if now is None else now)
        operation_id = preparation.operation_id
        payload = _encoded(preparation)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO cutover_operations(
                        operation_id, owner_id, task_id, preparation_json,
                        state, attempt, created_at, updated_at,
                        lease_owner, lease_expires_at, failure_type, schema_version
                    ) VALUES (?, ?, ?, ?, 'planned', 0, ?, ?, NULL, NULL, NULL, 1)
                    """,
                    (
                        operation_id,
                        preparation.owner_id,
                        preparation.task_id,
                        payload,
                        current,
                        current,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM cutover_operations WHERE operation_id=?",
                    (operation_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("cutover operation could not be persisted.")
                operation = self._operation(row)
                if operation.preparation.operation_id != preparation.operation_id:
                    raise RuntimeError("cutover operation identity collision detected.")
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return operation

    def get(self, operation_id: str) -> CutoverOperation | None:
        selected = identifier(operation_id, "operation_id", 64)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cutover_operations WHERE operation_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._operation(row)

    def list_operations(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 1000,
    ) -> tuple[CutoverOperation, ...]:
        owner = normalize_owner_id(owner_id)
        count = exact_integer(limit, "limit", 1, 10_000)
        params: list[Any] = [owner]
        query = "SELECT * FROM cutover_operations WHERE owner_id=?"
        if state is not None:
            selected = identifier(state, "state", 20)
            if selected not in {"planned", "running", "ready", "failed", "cancelled"}:
                raise ValueError("state is invalid.")
            query += " AND state=?"
            params.append(selected)
        query += " ORDER BY created_at, operation_id LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._operation(row) for row in rows)

    def claim(
        self,
        operation_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        max_attempts: int = 3,
        now: float | None = None,
    ) -> CutoverOperation:
        operation = identifier(operation_id, "operation_id", 64)
        worker = identifier(worker_id, "worker_id", 128)
        lease = exact_integer(lease_seconds, "lease_seconds", 1, 86_400)
        attempts = exact_integer(max_attempts, "max_attempts", 1, 100)
        current = timestamp(time.time() if now is None else now)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cutover_operations
                SET state='running', attempt=attempt+1, updated_at=?,
                    lease_owner=?, lease_expires_at=?, failure_type=NULL
                WHERE operation_id=? AND attempt < ? AND (
                    state='planned'
                    OR state='failed'
                    OR (state='running' AND lease_expires_at <= ?)
                )
                """,
                (
                    current,
                    worker,
                    current + lease,
                    operation,
                    attempts,
                    current,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("cutover operation is unavailable for preparation.")
        result = self.get(operation)
        if result is None:
            raise RuntimeError("cutover operation disappeared after claim.")
        return result

    def _transition(
        self,
        operation_id: str,
        *,
        worker_id: str,
        new_state: str,
        failure_type: str | None = None,
        now: float | None = None,
    ) -> CutoverOperation:
        operation = identifier(operation_id, "operation_id", 64)
        worker = identifier(worker_id, "worker_id", 128)
        current = timestamp(time.time() if now is None else now)
        failure = (
            identifier(failure_type, "failure_type", 200)
            if failure_type is not None
            else None
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cutover_operations
                SET state=?, updated_at=?, lease_owner=NULL,
                    lease_expires_at=NULL, failure_type=?
                WHERE operation_id=? AND state='running'
                  AND lease_owner=? AND lease_expires_at > ?
                """,
                (new_state, current, failure, operation, worker, current),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("cutover operation lease or state changed.")
        result = self.get(operation)
        if result is None:
            raise RuntimeError("cutover operation disappeared after transition.")
        return result

    def mark_ready(
        self,
        operation_id: str,
        *,
        worker_id: str,
        now: float | None = None,
    ) -> CutoverOperation:
        return self._transition(
            operation_id,
            worker_id=worker_id,
            new_state="ready",
            now=now,
        )

    def mark_failed(
        self,
        operation_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> CutoverOperation:
        return self._transition(
            operation_id,
            worker_id=worker_id,
            new_state="failed",
            failure_type=failure_type,
            now=now,
        )

    def cancel(self, operation_id: str, *, now: float | None = None) -> CutoverOperation:
        operation = identifier(operation_id, "operation_id", 64)
        current = timestamp(time.time() if now is None else now)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE cutover_operations
                SET state='cancelled', updated_at=?, lease_owner=NULL,
                    lease_expires_at=NULL
                WHERE operation_id=? AND state IN ('planned', 'failed')
                """,
                (current, operation),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("only planned or failed cutover preparation may be cancelled.")
        result = self.get(operation)
        if result is None:
            raise RuntimeError("cutover operation disappeared after cancellation.")
        return result


__all__ = ["MigrationCutoverJournal"]
