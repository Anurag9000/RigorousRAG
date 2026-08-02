"""SQLite journal for durable pre-restore custody artifact publication."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _digest,
    _identifier,
    _integer,
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_contracts import (
    RestoreCustodyArtifactAttempt,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_STATES = frozenset(
    {"planned", "running", "completed", "orphaned", "failed", "cancelled"}
)


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("artifact journal path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("artifact journal path is invalid.")
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
            raise ValueError("artifact journal path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("artifact journal path may not contain redirects.")
    return absolute


class RestoreCustodyArtifactJournal:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("artifact journal parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("artifact journal is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("artifact journal identity changed.")

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
                CREATE TABLE IF NOT EXISTS evidence_graph_restore_custody_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    target_path_digest TEXT NOT NULL,
                    backup_path_digest TEXT NOT NULL,
                    receipt_path_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    backup_sha256 TEXT,
                    backup_size_bytes INTEGER,
                    receipt_digest TEXT,
                    receipt_actor_id TEXT,
                    receipt_binding_method TEXT,
                    receipt_binding_digest TEXT,
                    disposition TEXT,
                    failure_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS restore_custody_artifact_queue
                    ON evidence_graph_restore_custody_artifacts(
                        owner_id, state, updated_at, artifact_id
                    );
                CREATE INDEX IF NOT EXISTS restore_custody_artifact_scope
                    ON evidence_graph_restore_custody_artifacts(
                        owner_id, snapshot_digest, target_path_digest, created_at
                    );
                """
            )

    @staticmethod
    def _attempt(row: sqlite3.Row) -> RestoreCustodyArtifactAttempt:
        try:
            return RestoreCustodyArtifactAttempt(
                artifact_id=row["artifact_id"],
                owner_id=row["owner_id"],
                snapshot_digest=row["snapshot_digest"],
                target_path_digest=row["target_path_digest"],
                backup_path_digest=row["backup_path_digest"],
                receipt_path_digest=row["receipt_path_digest"],
                state=row["state"],
                phase=row["phase"],
                attempt_count=int(row["attempt_count"]),
                max_attempts=int(row["max_attempts"]),
                lease_owner=row["lease_owner"],
                lease_expires_at=row["lease_expires_at"],
                backup_sha256=row["backup_sha256"],
                backup_size_bytes=row["backup_size_bytes"],
                receipt_digest=row["receipt_digest"],
                receipt_actor_id=row["receipt_actor_id"],
                receipt_binding_method=row["receipt_binding_method"],
                receipt_binding_digest=row["receipt_binding_digest"],
                disposition=row["disposition"],
                failure_type=row["failure_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored custody artifact attempt is corrupt.") from exc

    def seed(self, attempt: RestoreCustodyArtifactAttempt) -> RestoreCustodyArtifactAttempt:
        if not isinstance(attempt, RestoreCustodyArtifactAttempt):
            raise ValueError("attempt must be RestoreCustodyArtifactAttempt.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_restore_custody_artifacts "
                    "WHERE artifact_id=?",
                    (attempt.artifact_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO evidence_graph_restore_custody_artifacts "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)",
                        (
                            attempt.artifact_id,
                            attempt.owner_id,
                            attempt.snapshot_digest,
                            attempt.target_path_digest,
                            attempt.backup_path_digest,
                            attempt.receipt_path_digest,
                            attempt.state,
                            attempt.phase,
                            attempt.attempt_count,
                            attempt.max_attempts,
                            attempt.lease_owner,
                            attempt.lease_expires_at,
                            attempt.backup_sha256,
                            attempt.backup_size_bytes,
                            attempt.receipt_digest,
                            attempt.receipt_actor_id,
                            attempt.receipt_binding_method,
                            attempt.receipt_binding_digest,
                            attempt.disposition,
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
                    raise RuntimeError("custody artifact identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, artifact_id: str) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_restore_custody_artifacts "
                "WHERE artifact_id=?",
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
    ) -> tuple[RestoreCustodyArtifactAttempt, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("artifact state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM evidence_graph_restore_custody_artifacts "
            "WHERE owner_id=?"
        )
        parameters: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        query += " ORDER BY created_at DESC, artifact_id DESC LIMIT ?"
        parameters.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._attempt(row) for row in rows)

    def _require_running(
        self,
        connection: sqlite3.Connection,
        *,
        artifact_id: str,
        worker_id: str,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM evidence_graph_restore_custody_artifacts "
            "WHERE artifact_id=?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        if row["state"] != "running" or row["lease_owner"] != worker_id:
            raise RuntimeError("artifact attempt is not leased by this worker.")
        expires = row["lease_expires_at"]
        if expires is None or float(expires) <= now:
            raise RuntimeError("artifact attempt lease expired.")
        return row

    def claim(
        self,
        artifact_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_restore_custody_artifacts "
                    "WHERE artifact_id=?",
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
                    raise RuntimeError("artifact attempt is not claimable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError("artifact attempt exhausted its attempt ceiling.")
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "state='running', attempt_count=attempt_count+1, "
                    "lease_owner=?, lease_expires_at=?, failure_type=NULL, "
                    "updated_at=? WHERE artifact_id=?",
                    (worker, timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def renew(
        self,
        artifact_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    artifact_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "lease_expires_at=?, updated_at=? WHERE artifact_id=?",
                    (timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def record_publication_intent(
        self,
        artifact_id: str,
        *,
        worker_id: str,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        worker = _identifier(worker_id, "worker_id", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection,
                    artifact_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                if row["phase"] not in {"planned", "publication_intent"}:
                    raise RuntimeError("artifact attempt is past publication intent.")
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "phase='publication_intent', updated_at=? WHERE artifact_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def complete(
        self,
        artifact_id: str,
        *,
        worker_id: str,
        backup_sha256: str,
        backup_size_bytes: int,
        receipt_digest: str,
        receipt_actor_id: str,
        receipt_binding_method: str,
        receipt_binding_digest: str,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        worker = _identifier(worker_id, "worker_id", 200)
        backup = _digest(backup_sha256, "backup_sha256")
        size = _integer(
            backup_size_bytes,
            "backup_size_bytes",
            1,
            1024 * 1024 * 1024 * 1024,
        )
        receipt = _digest(receipt_digest, "receipt_digest")
        actor = _identifier(receipt_actor_id, "receipt_actor_id", 200)
        method = _identifier(receipt_binding_method, "receipt_binding_method", 50)
        binding = _digest(receipt_binding_digest, "receipt_binding_digest")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    artifact_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "state='completed', phase='verified', lease_owner=NULL, "
                    "lease_expires_at=NULL, backup_sha256=?, backup_size_bytes=?, "
                    "receipt_digest=?, receipt_actor_id=?, receipt_binding_method=?, "
                    "receipt_binding_digest=?, disposition='paired', failure_type=NULL, "
                    "updated_at=?, completed_at=? WHERE artifact_id=?",
                    (
                        backup,
                        size,
                        receipt,
                        actor,
                        method,
                        binding,
                        timestamp,
                        timestamp,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def orphan(
        self,
        artifact_id: str,
        *,
        worker_id: str,
        disposition: str,
        backup_sha256: str | None = None,
        backup_size_bytes: int | None = None,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        worker = _identifier(worker_id, "worker_id", 200)
        selected_disposition = _identifier(disposition, "disposition", 80)
        if selected_disposition not in {
            "backup_without_receipt",
            "receipt_without_backup",
            "artifact_collision",
        }:
            raise ValueError("orphan disposition is unsupported.")
        backup = (
            None if backup_sha256 is None else _digest(backup_sha256, "backup_sha256")
        )
        size = (
            None
            if backup_size_bytes is None
            else _integer(
                backup_size_bytes,
                "backup_size_bytes",
                1,
                1024 * 1024 * 1024 * 1024,
            )
        )
        if (backup is None) != (size is None):
            raise ValueError("orphan backup digest and size must be paired.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    artifact_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "state='orphaned', phase='observed', lease_owner=NULL, "
                    "lease_expires_at=NULL, backup_sha256=?, backup_size_bytes=?, "
                    "receipt_digest=NULL, receipt_actor_id=NULL, "
                    "receipt_binding_method=NULL, receipt_binding_digest=NULL, "
                    "disposition=?, failure_type=NULL, updated_at=?, completed_at=? "
                    "WHERE artifact_id=?",
                    (backup, size, selected_disposition, timestamp, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def fail(
        self,
        artifact_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        worker = _identifier(worker_id, "worker_id", 200)
        failure = _identifier(failure_type, "failure_type", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    artifact_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "state='failed', lease_owner=NULL, lease_expires_at=NULL, "
                    "failure_type=?, updated_at=? WHERE artifact_id=?",
                    (failure, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def retry(
        self,
        artifact_id: str,
        *,
        owner_id: str,
        confirm_artifact_id: str,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        confirmation = _digest(confirm_artifact_id, "confirm_artifact_id")
        owner = normalize_owner_id(owner_id)
        if confirmation != selected:
            raise ValueError("artifact retry confirmation differs.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_restore_custody_artifacts "
                    "WHERE artifact_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner or current.state != "failed":
                    raise RuntimeError("artifact attempt is not retryable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError("artifact attempt exhausted its attempt ceiling.")
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "state='planned', failure_type=NULL, updated_at=? "
                    "WHERE artifact_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def cancel(
        self,
        artifact_id: str,
        *,
        owner_id: str,
        confirm_artifact_id: str,
        now: float | None = None,
    ) -> RestoreCustodyArtifactAttempt:
        selected = _digest(artifact_id, "artifact_id")
        confirmation = _digest(confirm_artifact_id, "confirm_artifact_id")
        owner = normalize_owner_id(owner_id)
        if confirmation != selected:
            raise ValueError("artifact cancellation confirmation differs.")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_restore_custody_artifacts "
                    "WHERE artifact_id=?",
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
                    raise RuntimeError("artifact attempt cannot be cancelled.")
                connection.execute(
                    "UPDATE evidence_graph_restore_custody_artifacts SET "
                    "state='cancelled', failure_type=NULL, updated_at=?, "
                    "completed_at=? WHERE artifact_id=?",
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
            rows = connection.execute(
                "SELECT * FROM evidence_graph_restore_custody_artifacts "
                "WHERE owner_id=? AND (state='planned' OR "
                "(state='running' AND lease_expires_at<=?)) "
                "ORDER BY updated_at, artifact_id LIMIT 100",
                (owner, timestamp),
            ).fetchall()
        for row in rows:
            value = self._attempt(row)
            if value.attempt_count < value.max_attempts:
                return value.artifact_id
        return None


__all__ = ["RestoreCustodyArtifactJournal"]
