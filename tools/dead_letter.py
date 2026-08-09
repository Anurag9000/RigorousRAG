"""Privacy-safe durable dead-letter journal with fenced replay authority."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.security import normalize_owner_id

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_STATES = frozenset({"queued", "leased", "replayed", "abandoned"})


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0) or 0) & _REPARSE
    )


def _safe_path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("dead-letter path must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("dead-letter path is invalid.")
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
            raise ValueError("dead-letter path may not contain redirects.")
    return absolute


@dataclass(frozen=True)
class DeadLetterRecord:
    dead_letter_id: str
    owner_id: str
    job_id: str
    job_type: str
    payload_digest: str
    failure_type: str
    state: str
    delivery_attempts: int
    replay_count: int
    fencing_token: int
    created_at: float
    updated_at: float
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    replay_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dead_letter_id", _digest(self.dead_letter_id, "dead_letter_id"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "job_id", _identifier(self.job_id, "job_id"))
        object.__setattr__(self, "job_type", _identifier(self.job_type, "job_type", 100))
        object.__setattr__(self, "payload_digest", _digest(self.payload_digest, "payload_digest"))
        object.__setattr__(self, "failure_type", _identifier(self.failure_type, "failure_type", 200))
        if self.state not in _STATES:
            raise ValueError("dead-letter state is invalid.")
        object.__setattr__(
            self,
            "delivery_attempts",
            _integer(self.delivery_attempts, "delivery_attempts", 1, 1_000_000),
        )
        object.__setattr__(self, "replay_count", _integer(self.replay_count, "replay_count", 0, 1_000_000))
        object.__setattr__(self, "fencing_token", _integer(self.fencing_token, "fencing_token", 0, 2**63 - 1))
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at may not precede created_at.")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "updated_at", updated)
        if self.lease_owner is not None:
            object.__setattr__(self, "lease_owner", _identifier(self.lease_owner, "lease_owner", 128))
        if self.lease_expires_at is not None:
            object.__setattr__(self, "lease_expires_at", _timestamp(self.lease_expires_at, "lease_expires_at"))
        if self.replay_receipt_digest is not None:
            object.__setattr__(
                self,
                "replay_receipt_digest",
                _digest(self.replay_receipt_digest, "replay_receipt_digest"),
            )
        if self.state == "leased" and (
            self.lease_owner is None
            or self.lease_expires_at is None
            or self.fencing_token < 1
        ):
            raise ValueError("leased dead-letter records require fenced authority.")
        if self.state != "leased" and (
            self.lease_owner is not None or self.lease_expires_at is not None
        ):
            raise ValueError("inactive dead-letter records may not retain leases.")
        if self.state == "replayed" and self.replay_receipt_digest is None:
            raise ValueError("replayed dead-letter records require a replay receipt digest.")
        if self.state != "replayed" and self.replay_receipt_digest is not None:
            raise ValueError("only replayed dead-letter records may carry replay receipts.")

    @property
    def audit_digest(self) -> str:
        return _sha256(
            {
                "dead_letter_id": self.dead_letter_id,
                "owner_id": self.owner_id,
                "job_id": self.job_id,
                "job_type": self.job_type,
                "payload_digest": self.payload_digest,
                "failure_type": self.failure_type,
                "state": self.state,
                "delivery_attempts": self.delivery_attempts,
                "replay_count": self.replay_count,
                "fencing_token": self.fencing_token,
                "replay_receipt_digest": self.replay_receipt_digest,
            }
        )


class DeadLetterStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = _safe_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if _redirecting(parent) or not stat.S_ISDIR(parent.st_mode):
            raise ValueError("dead-letter parent must be a regular directory.")
        self._parent_identity = (int(parent.st_dev), int(parent.st_ino))
        self._lock = threading.RLock()
        self._initialize()
        self._database_identity = self._file_identity()

    def _file_identity(self) -> tuple[int, int]:
        info = self.path.lstat()
        if _redirecting(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("dead-letter database must be a regular file.")
        return int(info.st_dev), int(info.st_ino)

    def _verify(self) -> None:
        parent = self.path.parent.lstat()
        if (
            _redirecting(parent)
            or (int(parent.st_dev), int(parent.st_ino)) != self._parent_identity
        ):
            raise RuntimeError("dead-letter parent identity changed.")
        if self._file_identity() != self._database_identity:
            raise RuntimeError("dead-letter database identity changed.")

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
                CREATE TABLE IF NOT EXISTS dead_letters (
                    dead_letter_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    payload_digest TEXT NOT NULL,
                    failure_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    delivery_attempts INTEGER NOT NULL,
                    replay_count INTEGER NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    replay_receipt_digest TEXT
                );
                CREATE INDEX IF NOT EXISTS dead_letter_claim
                    ON dead_letters(state, lease_expires_at, created_at, dead_letter_id);
                CREATE INDEX IF NOT EXISTS dead_letter_owner
                    ON dead_letters(owner_id, state, created_at, dead_letter_id);
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> DeadLetterRecord:
        try:
            return DeadLetterRecord(**dict(row))
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise RuntimeError("stored dead-letter record is corrupt.") from exc

    def enqueue(
        self,
        *,
        owner_id: str,
        job_id: str,
        job_type: str,
        payload_digest: str,
        failure_type: str,
        delivery_attempts: int,
        now: float | None = None,
    ) -> DeadLetterRecord:
        owner = normalize_owner_id(owner_id)
        selected_job = _identifier(job_id, "job_id")
        selected_type = _identifier(job_type, "job_type", 100)
        selected_payload = _digest(payload_digest, "payload_digest")
        selected_failure = _identifier(failure_type, "failure_type", 200)
        attempts = _integer(delivery_attempts, "delivery_attempts", 1, 1_000_000)
        current = _timestamp(time.time() if now is None else now, "now")
        dead_letter_id = _sha256(
            {
                "contract": "rigorousrag-dead-letter-v1",
                "owner_id": owner,
                "job_id": selected_job,
                "job_type": selected_type,
                "payload_digest": selected_payload,
                "failure_type": selected_failure,
            }
        )
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO dead_letters(
                    dead_letter_id, owner_id, job_id, job_type, payload_digest,
                    failure_type, state, delivery_attempts, replay_count,
                    fencing_token, created_at, updated_at, lease_owner,
                    lease_expires_at, replay_receipt_digest
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, 0, 0, ?, ?, NULL, NULL, NULL)
                """,
                (
                    dead_letter_id,
                    owner,
                    selected_job,
                    selected_type,
                    selected_payload,
                    selected_failure,
                    attempts,
                    current,
                    current,
                ),
            )
        result = self.get(dead_letter_id)
        if result is None:
            raise RuntimeError("dead-letter record could not be persisted.")
        return result

    def get(self, dead_letter_id: str) -> DeadLetterRecord | None:
        selected = _digest(dead_letter_id, "dead_letter_id")
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM dead_letters WHERE dead_letter_id=?",
                (selected,),
            ).fetchone()
        return None if row is None else self._record(row)

    def list(
        self,
        *,
        owner_id: str,
        state: str | None = None,
        limit: int = 100,
    ) -> tuple[DeadLetterRecord, ...]:
        owner = normalize_owner_id(owner_id)
        count = _integer(limit, "limit", 1, 10_000)
        query = "SELECT * FROM dead_letters WHERE owner_id=?"
        params: list[Any] = [owner]
        if state is not None:
            selected_state = _identifier(state, "state", 20)
            if selected_state not in _STATES:
                raise ValueError("dead-letter state is invalid.")
            query += " AND state=?"
            params.append(selected_state)
        query += " ORDER BY created_at, dead_letter_id LIMIT ?"
        params.append(count)
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._record(row) for row in rows)

    def claim(
        self,
        dead_letter_id: str,
        *,
        worker_id: str,
        lease_seconds: int = 300,
        now: float | None = None,
    ) -> DeadLetterRecord:
        selected = _digest(dead_letter_id, "dead_letter_id")
        worker = _identifier(worker_id, "worker_id", 128)
        lease = _integer(lease_seconds, "lease_seconds", 1, 86_400)
        current = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE dead_letters
                SET state='leased', lease_owner=?, lease_expires_at=?,
                    fencing_token=fencing_token+1, updated_at=?
                WHERE dead_letter_id=? AND (
                    state='queued' OR (state='leased' AND lease_expires_at <= ?)
                )
                """,
                (worker, current + lease, current, selected, current),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("dead-letter record is unavailable for replay.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("dead-letter record disappeared after claim.")
        return result

    def _leased_transition(
        self,
        dead_letter_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        state: str,
        replay_receipt_digest: str | None,
        replay_increment: int,
        now: float | None,
    ) -> DeadLetterRecord:
        selected = _digest(dead_letter_id, "dead_letter_id")
        worker = _identifier(worker_id, "worker_id", 128)
        fence = _integer(fencing_token, "fencing_token", 1, 2**63 - 1)
        current = _timestamp(time.time() if now is None else now, "now")
        receipt = None if replay_receipt_digest is None else _digest(
            replay_receipt_digest, "replay_receipt_digest"
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE dead_letters
                SET state=?, replay_count=replay_count+?, replay_receipt_digest=?,
                    lease_owner=NULL, lease_expires_at=NULL, updated_at=?
                WHERE dead_letter_id=? AND state='leased' AND lease_owner=?
                  AND fencing_token=? AND lease_expires_at > ?
                """,
                (
                    state,
                    replay_increment,
                    receipt,
                    current,
                    selected,
                    worker,
                    fence,
                    current,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("dead-letter replay lease, fence or state changed.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("dead-letter record disappeared after transition.")
        return result

    def mark_replayed(
        self,
        dead_letter_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        replay_receipt_digest: str,
        now: float | None = None,
    ) -> DeadLetterRecord:
        return self._leased_transition(
            dead_letter_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
            state="replayed",
            replay_receipt_digest=replay_receipt_digest,
            replay_increment=1,
            now=now,
        )

    def release(
        self,
        dead_letter_id: str,
        *,
        worker_id: str,
        fencing_token: int,
        now: float | None = None,
    ) -> DeadLetterRecord:
        return self._leased_transition(
            dead_letter_id,
            worker_id=worker_id,
            fencing_token=fencing_token,
            state="queued",
            replay_receipt_digest=None,
            replay_increment=0,
            now=now,
        )

    def abandon(
        self,
        dead_letter_id: str,
        *,
        confirm_dead_letter_id: str,
        now: float | None = None,
    ) -> DeadLetterRecord:
        selected = _digest(dead_letter_id, "dead_letter_id")
        confirmation = _digest(confirm_dead_letter_id, "confirm_dead_letter_id")
        if confirmation != selected:
            raise ValueError("confirmation must exactly match dead_letter_id.")
        current = _timestamp(time.time() if now is None else now, "now")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE dead_letters
                SET state='abandoned', updated_at=?
                WHERE dead_letter_id=? AND state='queued'
                """,
                (current, selected),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("only queued dead-letter records may be abandoned.")
        result = self.get(selected)
        if result is None:
            raise RuntimeError("dead-letter record disappeared after abandonment.")
        return result


__all__ = ["DeadLetterRecord", "DeadLetterStore"]
