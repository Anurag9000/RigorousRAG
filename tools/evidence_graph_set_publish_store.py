"""Storage foundation for durable graph-set publication attempts."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from typing import Any

from tools.evidence_graph_set_publish_contracts import (
    EvidenceGraphSetPublicationAttempt,
    _MAX_LIMIT,
    _SCHEMA_VERSION,
    _STATES,
    _digest,
    _identifier,
    _integer,
    _path,
    _redirecting,
    _timestamp,
)
from tools.security import normalize_owner_id


class _PublicationJournalBase:
    """SQLite phase journal with expiring exclusive leases."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("publication database parent must be a directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("publication database is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("publication database parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("publication database identity changed.")

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
                CREATE TABLE IF NOT EXISTS evidence_graph_set_publications (
                    operation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    graph_set_key TEXT NOT NULL,
                    proposal_ids_json TEXT NOT NULL,
                    expected_current_set_id TEXT,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    previous_graph_set_id TEXT,
                    previous_graph_set_digest TEXT,
                    candidate_graph_set_id TEXT,
                    candidate_graph_set_digest TEXT,
                    member_count INTEGER,
                    edge_count INTEGER,
                    verification_digest TEXT,
                    failure_type TEXT,
                    compensation_errors_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evidence_graph_set_publication_queue
                    ON evidence_graph_set_publications(owner_id, state, updated_at, operation_id);
                CREATE INDEX IF NOT EXISTS evidence_graph_set_publication_scope
                    ON evidence_graph_set_publications(owner_id, graph_set_key, created_at, operation_id);
                """
            )

    @staticmethod
    def _json_tuple(value: Any, label: str) -> tuple[str, ...]:
        if not isinstance(value, str) or len(value) > 20_000_000:
            raise RuntimeError(f"stored {label} is corrupt.")
        try:
            parsed = json.loads(
                value,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise RuntimeError(f"stored {label} is corrupt.") from exc
        if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
            raise RuntimeError(f"stored {label} is corrupt.")
        return tuple(parsed)

    @classmethod
    def _attempt(cls, row: sqlite3.Row) -> EvidenceGraphSetPublicationAttempt:
        try:
            return EvidenceGraphSetPublicationAttempt(
                operation_id=row["operation_id"],
                owner_id=row["owner_id"],
                graph_set_key=row["graph_set_key"],
                proposal_ids=cls._json_tuple(row["proposal_ids_json"], "proposal IDs"),
                expected_current_set_id=row["expected_current_set_id"],
                state=row["state"],
                phase=row["phase"],
                attempt_count=int(row["attempt_count"]),
                max_attempts=int(row["max_attempts"]),
                lease_owner=row["lease_owner"],
                lease_expires_at=row["lease_expires_at"],
                previous_graph_set_id=row["previous_graph_set_id"],
                previous_graph_set_digest=row["previous_graph_set_digest"],
                candidate_graph_set_id=row["candidate_graph_set_id"],
                candidate_graph_set_digest=row["candidate_graph_set_digest"],
                member_count=row["member_count"],
                edge_count=row["edge_count"],
                verification_digest=row["verification_digest"],
                failure_type=row["failure_type"],
                compensation_errors=cls._json_tuple(
                    row["compensation_errors_json"], "compensation errors"
                ),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored publication attempt is corrupt.") from exc

    @staticmethod
    def _proposal_json(values: tuple[str, ...]) -> str:
        return json.dumps(list(values), separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _errors_json(values: tuple[str, ...]) -> str:
        return json.dumps(list(values), separators=(",", ":"), allow_nan=False)

    def seed(
        self, attempt: EvidenceGraphSetPublicationAttempt
    ) -> EvidenceGraphSetPublicationAttempt:
        if not isinstance(attempt, EvidenceGraphSetPublicationAttempt):
            raise ValueError("attempt must be EvidenceGraphSetPublicationAttempt.")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
                    (attempt.operation_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO evidence_graph_set_publications VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1
                        )
                        """,
                        (
                            attempt.operation_id,
                            attempt.owner_id,
                            attempt.graph_set_key,
                            self._proposal_json(attempt.proposal_ids),
                            attempt.expected_current_set_id,
                            attempt.state,
                            attempt.phase,
                            attempt.attempt_count,
                            attempt.max_attempts,
                            attempt.lease_owner,
                            attempt.lease_expires_at,
                            attempt.previous_graph_set_id,
                            attempt.previous_graph_set_digest,
                            attempt.candidate_graph_set_id,
                            attempt.candidate_graph_set_digest,
                            attempt.member_count,
                            attempt.edge_count,
                            attempt.verification_digest,
                            attempt.failure_type,
                            self._errors_json(attempt.compensation_errors),
                            attempt.created_at,
                            attempt.updated_at,
                            attempt.completed_at,
                        ),
                    )
                    connection.execute("COMMIT")
                    return attempt
                stored = self._attempt(row)
                if stored.immutable_digest != attempt.immutable_digest:
                    raise RuntimeError("publication operation identity collision detected.")
                connection.execute("COMMIT")
                return stored
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, operation_id: str) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return self._attempt(row)

    def list(
        self,
        *,
        owner_id: str,
        graph_set_key: str | None = None,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[EvidenceGraphSetPublicationAttempt, ...]:
        owner = normalize_owner_id(owner_id)
        key = None if graph_set_key is None else _identifier(
            graph_set_key, "graph_set_key", 500
        )
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("state is unsupported.")
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        query = "SELECT * FROM evidence_graph_set_publications WHERE owner_id=?"
        params: list[Any] = [owner]
        if key is not None:
            query += " AND graph_set_key=?"
            params.append(key)
        if selected_state is not None:
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY created_at DESC, operation_id DESC LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._attempt(row) for row in rows)

    def _require_running(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        worker_id: str,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
            (operation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(operation_id)
        if row["state"] != "running" or row["lease_owner"] != worker_id:
            raise RuntimeError("publication attempt is not leased by this worker.")
        lease_expires = row["lease_expires_at"]
        if lease_expires is None or float(lease_expires) <= now:
            raise RuntimeError("publication attempt lease expired.")
        return row

    def claim(
        self,
        operation_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM evidence_graph_set_publications WHERE operation_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                reclaimable = current.state == "running" and (
                    current.lease_expires_at is not None
                    and current.lease_expires_at <= timestamp
                )
                if current.state != "planned" and not reclaimable:
                    raise RuntimeError("publication attempt is not claimable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError(
                        "publication attempt exhausted its attempt ceiling."
                    )
                connection.execute(
                    """
                    UPDATE evidence_graph_set_publications
                    SET state='running', attempt_count=attempt_count+1,
                        lease_owner=?, lease_expires_at=?, failure_type=NULL,
                        compensation_errors_json='[]', updated_at=?
                    WHERE operation_id=?
                    """,
                    (worker, timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def renew(
        self,
        operation_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> EvidenceGraphSetPublicationAttempt:
        selected = _digest(operation_id, "operation_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(
                    connection, operation_id=selected, worker_id=worker, now=timestamp
                )
                connection.execute(
                    "UPDATE evidence_graph_set_publications SET lease_expires_at=?, updated_at=? "
                    "WHERE operation_id=?",
                    (timestamp + duration, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)


__all__ = ["_PublicationJournalBase"]
