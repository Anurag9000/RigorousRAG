"""SQLite journal for crash-recoverable empty-target retirement restores."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    SignedRetirementRestoreAttempt,
    _MAX_LIMIT,
    _STATES,
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("restore database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("restore database path is invalid.")
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
            raise ValueError("restore database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("restore database path may not contain redirects.")
    return absolute


class SignedRetirementRestoreJournal:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("restore database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("restore database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("restore database identity changed.")

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
        with sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        ) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_graph_set_signed_retirement_restores (
                    restore_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    target_path_digest TEXT NOT NULL,
                    snapshot_record_count INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    target_verification_digest TEXT,
                    failure_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS signed_retirement_restore_queue
                    ON evidence_graph_set_signed_retirement_restores(
                        owner_id, state, updated_at, restore_id
                    );
                """
            )

    @staticmethod
    def _attempt(row: sqlite3.Row) -> SignedRetirementRestoreAttempt:
        try:
            return SignedRetirementRestoreAttempt(
                restore_id=row["restore_id"],
                owner_id=row["owner_id"],
                snapshot_digest=row["snapshot_digest"],
                target_path_digest=row["target_path_digest"],
                snapshot_record_count=int(row["snapshot_record_count"]),
                state=row["state"],
                phase=row["phase"],
                attempt_count=int(row["attempt_count"]),
                max_attempts=int(row["max_attempts"]),
                lease_owner=row["lease_owner"],
                lease_expires_at=row["lease_expires_at"],
                target_verification_digest=row["target_verification_digest"],
                failure_type=row["failure_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored restore attempt is corrupt.") from exc

    def seed(
        self,
        attempt: SignedRetirementRestoreAttempt,
    ) -> SignedRetirementRestoreAttempt:
        if not isinstance(attempt, SignedRetirementRestoreAttempt):
            raise ValueError("attempt must be SignedRetirementRestoreAttempt.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirement_restores "
                    "WHERE restore_id=?",
                    (attempt.restore_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO evidence_graph_set_signed_retirement_restores "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            attempt.restore_id,
                            attempt.owner_id,
                            attempt.snapshot_digest,
                            attempt.target_path_digest,
                            attempt.snapshot_record_count,
                            attempt.state,
                            attempt.phase,
                            attempt.attempt_count,
                            attempt.max_attempts,
                            attempt.lease_owner,
                            attempt.lease_expires_at,
                            attempt.target_verification_digest,
                            attempt.failure_type,
                            attempt.created_at,
                            attempt.updated_at,
                            attempt.completed_at,
                        ),
                    )
                    connection.execute("COMMIT")
                    return attempt
                stored = self._attempt(row)
                if stored.immutable_digest != attempt.immutable_digest:
                    raise RuntimeError(
                        "restore operation identity collision detected."
                    )
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, restore_id: str) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_signed_retirement_restores "
                "WHERE restore_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._attempt(row)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedRetirementRestoreAttempt, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = (
            None if state is None else _identifier(state, "state", 30)
        )
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM evidence_graph_set_signed_retirement_restores "
            "WHERE owner_id=?"
        )
        params: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY created_at DESC, restore_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._attempt(row) for row in rows)

    def claim(
        self,
        restore_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirement_restores "
                    "WHERE restore_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                reclaimable = bool(
                    current.state == "running"
                    and current.lease_expires_at is not None
                    and current.lease_expires_at <= timestamp
                )
                if current.state != "planned" and not reclaimable:
                    raise RuntimeError("restore attempt is not claimable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError(
                        "restore attempt exhausted its attempt ceiling."
                    )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET state='running', attempt_count=attempt_count+1, "
                    "lease_owner=?, lease_expires_at=?, failure_type=NULL, "
                    "updated_at=? WHERE restore_id=?",
                    (worker, timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def _require_running(
        self,
        connection: sqlite3.Connection,
        *,
        restore_id: str,
        worker_id: str,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM evidence_graph_set_signed_retirement_restores "
            "WHERE restore_id=?",
            (restore_id,),
        ).fetchone()
        if row is None:
            raise KeyError(restore_id)
        if row["state"] != "running" or row["lease_owner"] != worker_id:
            raise RuntimeError("restore attempt is not leased by this worker.")
        if row["lease_expires_at"] is None or float(
            row["lease_expires_at"]
        ) <= now:
            raise RuntimeError("restore attempt lease expired.")
        return row

    def renew(
        self,
        restore_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    restore_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET lease_expires_at=?, updated_at=? WHERE restore_id=?",
                    (timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def record_target_committed(
        self,
        restore_id: str,
        *,
        worker_id: str,
        target_verification_digest: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        worker = _identifier(worker_id, "worker_id", 200)
        verification = _digest(
            target_verification_digest,
            "target_verification_digest",
        )
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection,
                    restore_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                current = self._attempt(row)
                if current.phase not in {"planned", "target_committed"}:
                    raise RuntimeError(
                        "restore target cannot be recorded in this phase."
                    )
                if (
                    current.phase == "target_committed"
                    and current.target_verification_digest != verification
                ):
                    raise RuntimeError(
                        "restore target verification changed during replay."
                    )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET phase='target_committed', "
                    "target_verification_digest=?, updated_at=? "
                    "WHERE restore_id=?",
                    (verification, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def complete(
        self,
        restore_id: str,
        *,
        worker_id: str,
        target_verification_digest: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        worker = _identifier(worker_id, "worker_id", 200)
        verification = _digest(
            target_verification_digest,
            "target_verification_digest",
        )
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection,
                    restore_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                current = self._attempt(row)
                if (
                    current.phase not in {"target_committed", "verified"}
                    or current.target_verification_digest != verification
                ):
                    raise RuntimeError(
                        "restore completion verification is invalid."
                    )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET state='completed', phase='verified', "
                    "lease_owner=NULL, lease_expires_at=NULL, "
                    "failure_type=NULL, updated_at=?, completed_at=? "
                    "WHERE restore_id=?",
                    (timestamp, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def fail(
        self,
        restore_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        worker = _identifier(worker_id, "worker_id", 200)
        failure = _identifier(failure_type, "failure_type", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    restore_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET state='failed', lease_owner=NULL, "
                    "lease_expires_at=NULL, failure_type=?, updated_at=? "
                    "WHERE restore_id=?",
                    (failure, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def retry(
        self,
        restore_id: str,
        *,
        owner_id: str,
        confirm_restore_id: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        if selected != _digest(confirm_restore_id, "confirm_restore_id"):
            raise ValueError("restore confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirement_restores "
                    "WHERE restore_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner or current.state != "failed":
                    raise RuntimeError(
                        "restore attempt is not retryable in this scope."
                    )
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError(
                        "restore attempt exhausted its attempt ceiling."
                    )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET state='planned', lease_owner=NULL, "
                    "lease_expires_at=NULL, failure_type=NULL, updated_at=? "
                    "WHERE restore_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def cancel(
        self,
        restore_id: str,
        *,
        owner_id: str,
        confirm_restore_id: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreAttempt:
        selected = _digest(restore_id, "restore_id")
        if selected != _digest(confirm_restore_id, "confirm_restore_id"):
            raise ValueError("restore confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirement_restores "
                    "WHERE restore_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if (
                    current.owner_id != owner
                    or current.state not in {"planned", "failed"}
                    or current.phase != "planned"
                ):
                    raise RuntimeError(
                        "only unstarted restores may be cancelled."
                    )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirement_restores "
                    "SET state='cancelled', lease_owner=NULL, "
                    "lease_expires_at=NULL, failure_type=NULL, "
                    "updated_at=?, completed_at=? WHERE restore_id=?",
                    (timestamp, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def next_claimable_id(
        self,
        *,
        owner_id: str,
        now: float | None = None,
    ) -> str | None:
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT restore_id "
                "FROM evidence_graph_set_signed_retirement_restores "
                "WHERE owner_id=? AND attempt_count < max_attempts AND ("
                "state='planned' OR "
                "(state='running' AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at<=?)) "
                "ORDER BY updated_at, restore_id LIMIT 1",
                (owner, timestamp),
            ).fetchone()
        return None if row is None else str(row["restore_id"])


__all__ = ["SignedRetirementRestoreJournal"]
