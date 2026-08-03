"""SQLite journal for crash-recoverable custody timestamp issuance."""

from __future__ import annotations

import hashlib
import json
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
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp import (
    CustodyTimestampAttestation,
    _from_dict,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    CustodyTimestampIssuanceAttempt,
)
from tools.evidence_graph_set_signed_retirement_snapshot import _canonical_bytes
from tools.security import normalize_owner_id

_TABLE = "evidence_graph_restore_custody_timestamp_issuances"
_MAX_LIMIT = 10_000
_STATES = frozenset({"planned", "running", "completed", "failed", "cancelled"})
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("stored timestamp attestation contains duplicate keys.")
        result[key] = value
    return result


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    rendered = os.fspath(value)
    if not isinstance(rendered, str) or not rendered or len(rendered) > 4096:
        raise ValueError("timestamp issuance journal path is invalid.")
    if any(ord(character) < 32 or ord(character) == 127 for character in rendered):
        raise ValueError("timestamp issuance journal path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if _redirecting(info):
            raise ValueError("timestamp issuance journal path may not contain redirects.")
    return absolute


def _attestation_bytes(value: CustodyTimestampAttestation) -> bytes:
    if not isinstance(value, CustodyTimestampAttestation):
        raise ValueError("attestation must be CustodyTimestampAttestation.")
    return _canonical_bytes(value.public_payload())


class CustodyTimestampIssuanceJournal:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("timestamp issuance journal parent is invalid.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("timestamp issuance journal is not a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or not stat.S_ISDIR(parent.st_mode)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
            or self._file_identity() != self._database_identity
        ):
            raise RuntimeError("timestamp issuance journal identity changed.")

    def _connect(self) -> sqlite3.Connection:
        self._verify()
        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with sqlite3.connect(self.path, timeout=30.0, isolation_level=None) as connection:
            connection.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    issuance_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    authority_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    serial TEXT NOT NULL,
                    attestation_digest TEXT NOT NULL,
                    output_path_digest TEXT NOT NULL,
                    attestation_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    verification_digest TEXT,
                    failure_type TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    schema_version INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS custody_timestamp_issuance_serial
                    ON {_TABLE}(owner_id, authority_id, key_id, serial);
                CREATE INDEX IF NOT EXISTS custody_timestamp_issuance_queue
                    ON {_TABLE}(owner_id, state, updated_at, issuance_id);
                """
            )

    @staticmethod
    def _attempt(row: sqlite3.Row) -> CustodyTimestampIssuanceAttempt:
        try:
            return CustodyTimestampIssuanceAttempt(
                issuance_id=row["issuance_id"],
                owner_id=row["owner_id"],
                authority_id=row["authority_id"],
                key_id=row["key_id"],
                serial=row["serial"],
                attestation_digest=row["attestation_digest"],
                output_path_digest=row["output_path_digest"],
                state=row["state"],
                phase=row["phase"],
                attempt_count=int(row["attempt_count"]),
                max_attempts=int(row["max_attempts"]),
                lease_owner=row["lease_owner"],
                lease_expires_at=row["lease_expires_at"],
                verification_digest=row["verification_digest"],
                failure_type=row["failure_type"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                completed_at=row["completed_at"],
                schema_version=int(row["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("stored timestamp issuance attempt is corrupt.") from exc

    @staticmethod
    def _payload(
        row: sqlite3.Row,
        attempt: CustodyTimestampIssuanceAttempt,
    ) -> CustodyTimestampAttestation:
        try:
            raw = json.loads(
                row["attestation_json"],
                object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
            value = _from_dict(raw)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("stored timestamp attestation payload is corrupt.") from exc
        digest = hashlib.sha256(_attestation_bytes(value)).hexdigest()
        if (
            digest != attempt.attestation_digest
            or value.serial != attempt.serial
            or value.owner_id != attempt.owner_id
            or value.authority_id != attempt.authority_id
            or value.key_id != attempt.key_id
        ):
            raise RuntimeError("stored timestamp attestation differs from issuance scope.")
        return value

    def seed(
        self,
        attempt: CustodyTimestampIssuanceAttempt,
        *,
        attestation: CustodyTimestampAttestation,
    ) -> CustodyTimestampIssuanceAttempt:
        if not isinstance(attempt, CustodyTimestampIssuanceAttempt):
            raise ValueError("attempt must be CustodyTimestampIssuanceAttempt.")
        payload = _attestation_bytes(attestation)
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != attempt.attestation_digest
            or attestation.serial != attempt.serial
            or attestation.owner_id != attempt.owner_id
            or attestation.authority_id != attempt.authority_id
            or attestation.key_id != attempt.key_id
        ):
            raise ValueError("timestamp attestation differs from issuance attempt.")
        rendered = payload.decode("utf-8")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
                    (attempt.issuance_id,),
                ).fetchone()
                if row is not None:
                    stored = self._attempt(row)
                    stored_payload = self._payload(row, stored)
                    if (
                        stored.immutable_digest != attempt.immutable_digest
                        or _attestation_bytes(stored_payload) != payload
                    ):
                        raise RuntimeError("timestamp issuance identity collision detected.")
                    connection.execute("COMMIT")
                    return stored
                serial_row = connection.execute(
                    f"SELECT issuance_id FROM {_TABLE} WHERE owner_id=? AND authority_id=? AND key_id=? AND serial=?",
                    (
                        attempt.owner_id,
                        attempt.authority_id,
                        attempt.key_id,
                        attempt.serial,
                    ),
                ).fetchone()
                if serial_row is not None:
                    raise RuntimeError("timestamp serial is already reserved.")
                connection.execute(
                    f"INSERT INTO {_TABLE} VALUES ({','.join('?' for _ in range(20))})",
                    (
                        attempt.issuance_id,
                        attempt.owner_id,
                        attempt.authority_id,
                        attempt.key_id,
                        attempt.serial,
                        attempt.attestation_digest,
                        attempt.output_path_digest,
                        rendered,
                        attempt.state,
                        attempt.phase,
                        attempt.attempt_count,
                        attempt.max_attempts,
                        attempt.lease_owner,
                        attempt.lease_expires_at,
                        attempt.verification_digest,
                        attempt.failure_type,
                        attempt.created_at,
                        attempt.updated_at,
                        attempt.completed_at,
                        attempt.schema_version,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(attempt.issuance_id)

    def _row(self, issuance_id: str) -> sqlite3.Row:
        selected = _digest(issuance_id, "issuance_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
                (selected,),
            ).fetchone()
        if row is None:
            raise KeyError(selected)
        return row

    def get(self, issuance_id: str) -> CustodyTimestampIssuanceAttempt:
        return self._attempt(self._row(issuance_id))

    def get_attestation(self, issuance_id: str) -> CustodyTimestampAttestation:
        row = self._row(issuance_id)
        return self._payload(row, self._attempt(row))

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[CustodyTimestampIssuanceAttempt, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, _MAX_LIMIT)
        selected_state = None if state is None else _identifier(state, "state", 30)
        if selected_state is not None and selected_state not in _STATES:
            raise ValueError("timestamp issuance state is unsupported.")
        query = f"SELECT * FROM {_TABLE} WHERE owner_id=?"
        parameters: list[Any] = [owner]
        if selected_state is not None:
            query += " AND state=?"
            parameters.append(selected_state)
        query += " ORDER BY created_at DESC, issuance_id DESC LIMIT ?"
        parameters.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(parameters)).fetchall()
        return tuple(self._attempt(row) for row in rows)

    def claim(
        self,
        issuance_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        now: float | None = None,
    ) -> CustodyTimestampIssuanceAttempt:
        selected = _digest(issuance_id, "issuance_id")
        worker = _identifier(worker_id, "worker_id", 200)
        duration = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        timestamp = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
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
                    raise RuntimeError("timestamp issuance is not claimable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError("timestamp issuance exhausted its attempt ceiling.")
                connection.execute(
                    f"UPDATE {_TABLE} SET state='running', attempt_count=attempt_count+1, lease_owner=?, lease_expires_at=?, failure_type=NULL, updated_at=? WHERE issuance_id=?",
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
        issuance_id: str,
        worker_id: str,
        now: float,
    ) -> sqlite3.Row:
        row = connection.execute(
            f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
            (issuance_id,),
        ).fetchone()
        if row is None:
            raise KeyError(issuance_id)
        if row["state"] != "running" or row["lease_owner"] != worker_id:
            raise RuntimeError("timestamp issuance is not leased by this worker.")
        if row["lease_expires_at"] is None or float(row["lease_expires_at"]) <= now:
            raise RuntimeError("timestamp issuance lease expired.")
        return row

    def record_output_published(
        self,
        issuance_id: str,
        *,
        worker_id: str,
        now: float,
    ) -> CustodyTimestampIssuanceAttempt:
        selected = _digest(issuance_id, "issuance_id")
        worker = _identifier(worker_id, "worker_id", 200)
        timestamp = _timestamp(now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(connection, selected, worker, timestamp)
                if row["phase"] not in {"planned", "output_published"}:
                    raise RuntimeError("timestamp issuance phase cannot record output.")
                connection.execute(
                    f"UPDATE {_TABLE} SET phase='output_published', updated_at=? WHERE issuance_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def complete(
        self,
        issuance_id: str,
        *,
        worker_id: str,
        verification_digest: str,
        now: float,
    ) -> CustodyTimestampIssuanceAttempt:
        selected = _digest(issuance_id, "issuance_id")
        worker = _identifier(worker_id, "worker_id", 200)
        verification = _digest(verification_digest, "verification_digest")
        timestamp = _timestamp(now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._require_running(connection, selected, worker, timestamp)
                if row["phase"] != "output_published":
                    raise RuntimeError("timestamp issuance output is not durably recorded.")
                connection.execute(
                    f"UPDATE {_TABLE} SET state='completed', phase='verified', lease_owner=NULL, lease_expires_at=NULL, verification_digest=?, failure_type=NULL, updated_at=?, completed_at=? WHERE issuance_id=?",
                    (verification, timestamp, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def fail(
        self,
        issuance_id: str,
        *,
        worker_id: str,
        failure_type: str,
        now: float,
    ) -> CustodyTimestampIssuanceAttempt:
        selected = _digest(issuance_id, "issuance_id")
        worker = _identifier(worker_id, "worker_id", 200)
        failure = _identifier(failure_type, "failure_type", 200)
        timestamp = _timestamp(now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_running(connection, selected, worker, timestamp)
                connection.execute(
                    f"UPDATE {_TABLE} SET state='failed', lease_owner=NULL, lease_expires_at=NULL, failure_type=?, updated_at=? WHERE issuance_id=?",
                    (failure, timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def retry(
        self,
        issuance_id: str,
        *,
        owner_id: str,
        confirm_issuance_id: str,
        now: float,
    ) -> CustodyTimestampIssuanceAttempt:
        selected = _digest(issuance_id, "issuance_id")
        if selected != _digest(confirm_issuance_id, "confirm_issuance_id"):
            raise ValueError("timestamp issuance confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
                    (selected,),
                ).fetchone()
                if row is None:
                    raise KeyError(selected)
                current = self._attempt(row)
                if current.owner_id != owner or current.state != "failed":
                    raise RuntimeError("timestamp issuance is not retryable.")
                if current.attempt_count >= current.max_attempts:
                    raise RuntimeError("timestamp issuance exhausted its attempt ceiling.")
                connection.execute(
                    f"UPDATE {_TABLE} SET state='planned', failure_type=NULL, updated_at=? WHERE issuance_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def cancel(
        self,
        issuance_id: str,
        *,
        owner_id: str,
        confirm_issuance_id: str,
        now: float,
    ) -> CustodyTimestampIssuanceAttempt:
        selected = _digest(issuance_id, "issuance_id")
        if selected != _digest(confirm_issuance_id, "confirm_issuance_id"):
            raise ValueError("timestamp issuance confirmation differs.")
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(now, "now")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE issuance_id=?",
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
                        "timestamp issuance cannot be cancelled after output work."
                    )
                connection.execute(
                    f"UPDATE {_TABLE} SET state='cancelled', failure_type=NULL, updated_at=? WHERE issuance_id=?",
                    (timestamp, selected),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.get(selected)

    def next_claimable_id(self, *, owner_id: str, now: float) -> str | None:
        owner = normalize_owner_id(owner_id)
        timestamp = _timestamp(now, "now")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                f"SELECT issuance_id FROM {_TABLE} WHERE owner_id=? AND attempt_count < max_attempts AND (state='planned' OR (state='running' AND lease_expires_at<=?)) ORDER BY created_at, issuance_id LIMIT 1",
                (owner, timestamp),
            ).fetchone()
        return None if row is None else str(row["issuance_id"])


__all__ = ["CustodyTimestampIssuanceJournal"]
