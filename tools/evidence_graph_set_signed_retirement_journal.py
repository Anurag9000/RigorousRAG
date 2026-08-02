"""SQLite journal for crash-recoverable signed publication retirement sagas."""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any

from tools.evidence_graph_set_signed_retirement_contracts import (
    SignedPublicationRetirementAttempt,
    _MAX_LIMIT,
    _STATES,
    _digest,
    _identifier,
    _integer,
    _optional_digest,
    _timestamp,
)
from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_PATH = 4096
_SCHEMA_VERSION = 1


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("retirement database path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("retirement database path is invalid.")
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
            raise ValueError("retirement database path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("retirement database path may not contain redirects.")
    return absolute


class SignedPublicationRetirementJournal:
    """Append-only-scope retirement attempts with lease-guarded phase changes."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("retirement database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("retirement database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("retirement database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("retirement database identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evidence_graph_set_signed_retirements (
                    retirement_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    publication_operation_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    signed_candidate_set_id TEXT NOT NULL,
                    signed_candidate_set_digest TEXT NOT NULL,
                    authorization_candidate_set_id TEXT,
                    signed_authority_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    final_pointer_set_id TEXT,
                    verification_digest TEXT,
                    failure_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_graph_set_signed_retirement_queue
                    ON evidence_graph_set_signed_retirements(
                        owner_id, state, updated_at, retirement_id
                    );
                CREATE INDEX IF NOT EXISTS evidence_graph_set_signed_retirement_scope
                    ON evidence_graph_set_signed_retirements(
                        owner_id, publication_operation_id, created_at, retirement_id
                    );
                """
            )

    @staticmethod
    def _attempt(row: sqlite3.Row) -> SignedPublicationRetirementAttempt:
        try:
            return SignedPublicationRetirementAttempt(
                retirement_id=row["retirement_id"],
                owner_id=row["owner_id"],
                publication_operation_id=row["publication_operation_id"],
                graph_set_key=row["graph_set_key"],
                signed_candidate_set_id=row["signed_candidate_set_id"],
                signed_candidate_set_digest=row["signed_candidate_set_digest"],
                authorization_candidate_set_id=row["authorization_candidate_set_id"],
                signed_authority_digest=row["signed_authority_digest"],
                state=row["state"],
                phase=row["phase"],
                attempt_count=int(row["attempt_count"]),
                max_attempts=int(row["max_attempts"]),
                lease_owner=row["lease_owner"],
                lease_expires_at=row["lease_expires_at"],
                final_pointer_set_id=row["final_pointer_set_id"],
                verification_digest=row["verification_digest"],
                failure_type=row["failure_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored retirement attempt is corrupt.") from exc

    def seed(
        self, attempt: SignedPublicationRetirementAttempt
    ) -> SignedPublicationRetirementAttempt:
        if not isinstance(attempt, SignedPublicationRetirementAttempt):
            raise ValueError("attempt must be SignedPublicationRetirementAttempt.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirements "
                    "WHERE retirement_id=?",
                    (attempt.retirement_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_set_signed_retirements VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1
                        )
                        """,
                        (
                            attempt.retirement_id,
                            attempt.owner_id,
                            attempt.publication_operation_id,
                            attempt.graph_set_key,
                            attempt.signed_candidate_set_id,
                            attempt.signed_candidate_set_digest,
                            attempt.authorization_candidate_set_id,
                            attempt.signed_authority_digest,
                            attempt.state,
                            attempt.phase,
                            attempt.attempt_count,
                            attempt.max_attempts,
                            attempt.lease_owner,
                            attempt.lease_expires_at,
                            attempt.final_pointer_set_id,
                            attempt.verification_digest,
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
                    raise RuntimeError("retirement identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, retirement_id: str) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_signed_retirements "
                "WHERE retirement_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._attempt(row)

    def list(
        self,
        *,
        owner_id: str,
        publication_operation_id: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[SignedPublicationRetirementAttempt, ...]:
        owner = normalize_owner_id(owner_id)
        operation = None if publication_operation_id is None else _digest(
            publication_operation_id, "publication_operation_id"
        )
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = "SELECT * FROM evidence_graph_set_signed_retirements WHERE owner_id=?"
        params: list[Any] = [owner]
        if operation is not None:
            query += " AND publication_operation_id=?"
            params.append(operation)
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY created_at DESC, retirement_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._attempt(row) for row in rows)

    def claim(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirements "
                    "WHERE retirement_id=?",
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
                    raise RuntimeError("retirement attempt is not claimable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError("retirement attempt exhausted its attempt ceiling.")
                connection.execute(
                    """
                    UPDATE evidence_graph_set_signed_retirements
                    SET state='running', attempt_count=attempt_count+1,
                        lease_owner=?, lease_expires_at=?, failure_type=NULL,
                        updated_at=? WHERE retirement_id=?
                    """,
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
        retirement_id: str,
        worker_id: str,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM evidence_graph_set_signed_retirements "
            "WHERE retirement_id=?",
            (retirement_id,),
        ).fetchone()
        if row is None:
            raise KeyError(retirement_id)
        if row["state"] != "running" or row["lease_owner"] != worker_id:
            raise RuntimeError("retirement attempt is not leased by this worker.")
        lease_expires = row["lease_expires_at"]
        if lease_expires is None or float(lease_expires) <= now:
            raise RuntimeError("retirement attempt lease expired.")
        return row

    def renew(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    retirement_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirements "
                    "SET lease_expires_at=?, updated_at=? WHERE retirement_id=?",
                    (timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def _phase(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        allowed: set[str],
        phase: str,
        final_pointer_set_id: str | None = None,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        worker = _identifier(worker_id, "worker_id", 200)
        pointer = _optional_digest(final_pointer_set_id, "final_pointer_set_id")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection,
                    retirement_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                current = self._attempt(row)
                if current.phase not in allowed:
                    raise RuntimeError("retirement phase transition is invalid.")
                if current.phase == phase and phase == "authorization_retired":
                    if current.final_pointer_set_id != pointer:
                        raise RuntimeError("retirement pointer observation changed on replay.")
                connection.execute(
                    "UPDATE evidence_graph_set_signed_retirements "
                    "SET phase=?, final_pointer_set_id=?, updated_at=? "
                    "WHERE retirement_id=?",
                    (
                        phase,
                        pointer if phase == "authorization_retired" else None,
                        timestamp,
                        selected,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def record_pointer_restore_intent(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        return self._phase(
            retirement_id,
            worker_id=worker_id,
            allowed={"planned", "pointer_restore_intent"},
            phase="pointer_restore_intent",
            now=now,
        )

    def record_pointer_safe(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        return self._phase(
            retirement_id,
            worker_id=worker_id,
            allowed={"pointer_restore_intent", "pointer_safe"},
            phase="pointer_safe",
            now=now,
        )

    def record_authorization_retired(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        final_pointer_set_id: str | None,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        return self._phase(
            retirement_id,
            worker_id=worker_id,
            allowed={"pointer_safe", "authorization_retired"},
            phase="authorization_retired",
            final_pointer_set_id=final_pointer_set_id,
            now=now,
        )

    def complete(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        verification_digest: str,
        final_pointer_set_id: str | None,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        worker = _identifier(worker_id, "worker_id", 200)
        verification = _digest(verification_digest, "verification_digest")
        pointer = _optional_digest(final_pointer_set_id, "final_pointer_set_id")
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(
                    connection,
                    retirement_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                current = self._attempt(row)
                if current.phase != "authorization_retired":
                    raise RuntimeError("retirement cannot complete from this phase.")
                connection.execute(
                    """
                    UPDATE evidence_graph_set_signed_retirements
                    SET state='completed', phase='verified', lease_owner=NULL,
                        lease_expires_at=NULL, final_pointer_set_id=?,
                        verification_digest=?, failure_type=NULL,
                        updated_at=?, completed_at=? WHERE retirement_id=?
                    """,
                    (pointer, verification, timestamp, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def fail(
        self,
        retirement_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        worker = _identifier(worker_id, "worker_id", 200)
        failure = _identifier(failure_type, "failure_type", 200)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection,
                    retirement_id=selected,
                    worker_id=worker,
                    now=timestamp,
                )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_signed_retirements
                    SET state='failed', lease_owner=NULL, lease_expires_at=NULL,
                        failure_type=?, updated_at=? WHERE retirement_id=?
                    """,
                    (failure, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def retry(
        self,
        retirement_id: str,
        *,
        owner_id: str,
        confirm_retirement_id: str,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        confirmation = _digest(confirm_retirement_id, "confirm_retirement_id")
        if selected != confirmation:
            raise ValueError("retirement confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirements "
                    "WHERE retirement_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner:
                    raise RuntimeError("retirement attempt escaped owner scope.")
                if current.state != "failed":
                    raise RuntimeError("only failed retirement attempts may be retried.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError("retirement attempt exhausted its attempt ceiling.")
                connection.execute(
                    """
                    UPDATE evidence_graph_set_signed_retirements
                    SET state='planned', failure_type=NULL, updated_at=?
                    WHERE retirement_id=?
                    """,
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def cancel(
        self,
        retirement_id: str,
        *,
        owner_id: str,
        confirm_retirement_id: str,
        now: float | None = None,
    ) -> SignedPublicationRetirementAttempt:
        selected = _digest(retirement_id, "retirement_id")
        confirmation = _digest(confirm_retirement_id, "confirm_retirement_id")
        if selected != confirmation:
            raise ValueError("retirement confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_signed_retirements "
                    "WHERE retirement_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner:
                    raise RuntimeError("retirement attempt escaped owner scope.")
                if current.state not in {"planned", "failed"} or current.phase != "planned":
                    raise RuntimeError(
                        "only unstarted planned/failed retirements may be cancelled."
                    )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_signed_retirements
                    SET state='cancelled', lease_owner=NULL, lease_expires_at=NULL,
                        failure_type=NULL, updated_at=?, completed_at=?
                    WHERE retirement_id=?
                    """,
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
                """
                SELECT retirement_id FROM evidence_graph_set_signed_retirements
                WHERE owner_id=? AND attempt_count < max_attempts AND (
                    state='planned' OR
                    (state='running' AND lease_expires_at IS NOT NULL
                        AND lease_expires_at<=?)
                )
                ORDER BY updated_at, retirement_id LIMIT 1
                """,
                (owner, timestamp),
            ).fetchone()
        return None if row is None else str(row["retirement_id"])


__all__ = ["SignedPublicationRetirementJournal"]
