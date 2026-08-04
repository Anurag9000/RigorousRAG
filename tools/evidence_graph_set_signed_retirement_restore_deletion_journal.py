"""SQLite journal for crash-recoverable restore-intent deletion attempts."""

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
from tools.evidence_graph_set_signed_retirement_restore_deletion_execution_contracts import (
    SignedRetirementRestoreDeletionAttempt,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_MAX_LIMIT = 10_000
_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("deletion database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("deletion database path is invalid.")
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
            raise ValueError(
                "deletion database path could not be validated."
            ) from exc
        if _redirecting(info):
            raise ValueError(
                "deletion database path may not contain redirects."
            )
    return absolute


class SignedRetirementRestoreDeletionJournal:
    """Append-only deletion scope with lease-guarded monotonic phases."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("deletion database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("deletion database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino))
            != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("deletion database identity changed.")

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
                CREATE TABLE IF NOT EXISTS signed_retirement_restore_deletions (
                    deletion_id TEXT PRIMARY KEY,
                    authorization_id TEXT NOT NULL,
                    authorization_digest TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    restore_id TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL,
                    target_path_digest TEXT NOT NULL,
                    restore_state TEXT NOT NULL,
                    restore_phase TEXT NOT NULL,
                    restore_record_digest TEXT NOT NULL,
                    custody_id TEXT,
                    custody_manifest_digest TEXT,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    marker_digest TEXT,
                    tombstone_digest TEXT,
                    failure_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS signed_restore_deletion_queue
                    ON signed_retirement_restore_deletions(
                        owner_id, state, updated_at, deletion_id
                    );
                CREATE INDEX IF NOT EXISTS signed_restore_deletion_scope
                    ON signed_retirement_restore_deletions(
                        owner_id, restore_id, created_at, deletion_id
                    );
                """
            )

    @staticmethod
    def _value(row: sqlite3.Row) -> SignedRetirementRestoreDeletionAttempt:
        try:
            return SignedRetirementRestoreDeletionAttempt(
                **{
                    name: row[name]
                    for name in SignedRetirementRestoreDeletionAttempt.__dataclass_fields__
                }
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored deletion attempt is corrupt.") from exc

    def seed(
        self,
        attempt: SignedRetirementRestoreDeletionAttempt,
    ) -> SignedRetirementRestoreDeletionAttempt:
        if not isinstance(attempt, SignedRetirementRestoreDeletionAttempt):
            raise ValueError(
                "attempt must be SignedRetirementRestoreDeletionAttempt."
            )
        fields = tuple(attempt.__dataclass_fields__)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_retirement_restore_deletions "
                    "WHERE deletion_id=?",
                    (attempt.deletion_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO signed_retirement_restore_deletions "
                        f"({','.join(fields)}) VALUES "
                        f"({','.join('?' for _ in fields)})",
                        tuple(getattr(attempt, field) for field in fields),
                    )
                    connection.execute("COMMIT")
                    return attempt
                stored = self._value(row)
                if stored.immutable_digest != attempt.immutable_digest:
                    raise RuntimeError("deletion identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, deletion_id: str) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM signed_retirement_restore_deletions "
                "WHERE deletion_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._value(row)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedRetirementRestoreDeletionAttempt, ...]:
        owner = normalize_owner_id(owner_id)
        selected_state = (
            None if state is None else _identifier(state, "state", 20)
        )
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("deletion state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = (
            "SELECT * FROM signed_retirement_restore_deletions "
            "WHERE owner_id=?"
        )
        params: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY created_at DESC, deletion_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._value(row) for row in rows)

    def _require_running(
        self,
        connection: sqlite3.Connection,
        *,
        deletion_id: str,
        worker_id: str,
        now: float,
    ) -> SignedRetirementRestoreDeletionAttempt:
        row = connection.execute(
            "SELECT * FROM signed_retirement_restore_deletions "
            "WHERE deletion_id=?",
            (deletion_id,),
        ).fetchone()
        if row is None:
            raise KeyError(deletion_id)
        value = self._value(row)
        if (
            value.state != "running"
            or value.lease_owner != worker_id
            or value.lease_expires_at is None
            or value.lease_expires_at <= now
        ):
            raise RuntimeError(
                "deletion attempt is not leased by this worker."
            )
        return value

    def claim(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_retirement_restore_deletions "
                    "WHERE deletion_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._value(row)
                reclaimable = bool(
                    current.state == "running"
                    and current.lease_expires_at is not None
                    and current.lease_expires_at <= timestamp
                )
                if current.state != "planned" and not reclaimable:
                    raise RuntimeError("deletion attempt is not claimable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError(
                        "deletion attempt exhausted its attempt ceiling."
                    )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions SET "
                    "state='running', attempt_count=attempt_count+1, "
                    "lease_owner=?, lease_expires_at=?, failure_type=NULL, "
                    "updated_at=? WHERE deletion_id=?",
                    (
                        worker,
                        timestamp + duration,
                        timestamp,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def renew(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    deletion_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions "
                    "SET lease_expires_at=?, updated_at=? WHERE deletion_id=?",
                    (timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def _record_phase(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        allowed_phases: set[str],
        phase: str,
        marker_digest: str | None = None,
        tombstone_digest: str | None = None,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        worker = _identifier(worker_id, "worker_id", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_running(
                    connection,
                    deletion_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                if current.phase not in allowed_phases:
                    raise RuntimeError(
                        "deletion phase transition is invalid."
                    )
                marker = (
                    current.marker_digest
                    if marker_digest is None
                    else _digest(marker_digest, "marker_digest")
                )
                tombstone = (
                    current.tombstone_digest
                    if tombstone_digest is None
                    else _digest(tombstone_digest, "tombstone_digest")
                )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions SET "
                    "phase=?, marker_digest=?, tombstone_digest=?, updated_at=? "
                    "WHERE deletion_id=?",
                    (phase, marker, tombstone, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def record_marker_active(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        marker_digest: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        return self._record_phase(
            deletion_id,
            worker_id=worker_id,
            allowed_phases={"planned", "marker_active"},
            phase="marker_active",
            marker_digest=marker_digest,
            now=now,
        )

    def record_restore_deleted(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        marker_digest: str,
        tombstone_digest: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        return self._record_phase(
            deletion_id,
            worker_id=worker_id,
            allowed_phases={"marker_active", "restore_deleted"},
            phase="restore_deleted",
            marker_digest=marker_digest,
            tombstone_digest=tombstone_digest,
            now=now,
        )

    def complete(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        marker_digest: str,
        tombstone_digest: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        worker = _identifier(worker_id, "worker_id", 200)
        marker = _digest(marker_digest, "marker_digest")
        tombstone = _digest(tombstone_digest, "tombstone_digest")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._require_running(
                    connection,
                    deletion_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                if current.phase not in {"restore_deleted", "verified"}:
                    raise RuntimeError(
                        "deletion is not ready for completion."
                    )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions SET "
                    "state='completed', phase='verified', marker_digest=?, "
                    "tombstone_digest=?, lease_owner=NULL, "
                    "lease_expires_at=NULL, failure_type=NULL, updated_at=?, "
                    "completed_at=? WHERE deletion_id=?",
                    (
                        marker,
                        tombstone,
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

    def fail(
        self,
        deletion_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        worker = _identifier(worker_id, "worker_id", 200)
        failure = _identifier(failure_type, "failure_type", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    deletion_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions SET "
                    "state='failed', lease_owner=NULL, lease_expires_at=NULL, "
                    "failure_type=?, updated_at=? WHERE deletion_id=?",
                    (failure, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def retry(
        self,
        deletion_id: str,
        *,
        owner_id: str,
        confirm_deletion_id: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        if selected != _digest(confirm_deletion_id, "confirm_deletion_id"):
            raise ValueError("deletion confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_retirement_restore_deletions "
                    "WHERE deletion_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._value(row)
                if (
                    current.owner_id != owner
                    or current.state != "failed"
                    or current.attempt_count >= current.max_attempts
                ):
                    raise RuntimeError("deletion attempt is not retryable.")
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions SET "
                    "state='planned', failure_type=NULL, updated_at=? "
                    "WHERE deletion_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def cancel(
        self,
        deletion_id: str,
        *,
        owner_id: str,
        confirm_deletion_id: str,
        now: float | None = None,
    ) -> SignedRetirementRestoreDeletionAttempt:
        selected = _digest(deletion_id, "deletion_id")
        if selected != _digest(confirm_deletion_id, "confirm_deletion_id"):
            raise ValueError("deletion confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM signed_retirement_restore_deletions "
                    "WHERE deletion_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._value(row)
                if (
                    current.owner_id != owner
                    or current.state not in {"planned", "failed"}
                    or current.phase != "planned"
                ):
                    raise RuntimeError(
                        "only unstarted deletion work can be cancelled."
                    )
                connection.execute(
                    "UPDATE signed_retirement_restore_deletions SET "
                    "state='cancelled', failure_type=NULL, updated_at=?, "
                    "completed_at=? WHERE deletion_id=?",
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
                "SELECT * FROM signed_retirement_restore_deletions "
                "WHERE owner_id=? AND (state='planned' OR "
                "(state='running' AND lease_expires_at<=?)) "
                "ORDER BY updated_at, deletion_id",
                (owner, timestamp),
            ).fetchall()
        for row in rows:
            value = self._value(row)
            if value.attempt_count < value.max_attempts:
                return value.deletion_id
        return None


__all__ = ["SignedRetirementRestoreDeletionJournal"]
