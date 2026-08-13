"""SQLite-backed owner-scoped leases with monotonic fencing tokens.

This is a coordination primitive, not a distributed scheduler or a multi-store atomic
cutover transaction. Fencing tokens let downstream writers reject stale lease holders.
"""

from __future__ import annotations

import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.security import normalize_owner_id

_MAX_TEXT = 500
_MAX_TTL = 86_400.0


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > _MAX_TEXT or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _finite(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


@dataclass(frozen=True)
class LeaseRecord:
    owner_id: str
    resource_id: str
    holder_id: str
    fencing_token: int
    acquired_at: float
    expires_at: float

    @property
    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())


class LeaseCoordinator:
    """Replay-safe lease coordinator with one monotonic token stream per resource."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS leases (
                    owner_id TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    holder_id TEXT,
                    fencing_token INTEGER NOT NULL DEFAULT 0,
                    acquired_at REAL NOT NULL DEFAULT 0,
                    expires_at REAL NOT NULL DEFAULT 0,
                    PRIMARY KEY (owner_id, resource_id)
                );
                CREATE INDEX IF NOT EXISTS idx_leases_expiry ON leases (expires_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> LeaseRecord | None:
        holder = row["holder_id"]
        if holder is None:
            return None
        return LeaseRecord(
            owner_id=str(row["owner_id"]),
            resource_id=str(row["resource_id"]),
            holder_id=str(holder),
            fencing_token=int(row["fencing_token"]),
            acquired_at=float(row["acquired_at"]),
            expires_at=float(row["expires_at"]),
        )

    def acquire(
        self,
        *,
        owner_id: str,
        resource_id: str,
        holder_id: str,
        ttl_seconds: float = 60.0,
        now: float | None = None,
    ) -> LeaseRecord | None:
        owner = normalize_owner_id(owner_id)
        resource = _identifier(resource_id, "resource_id")
        holder = _identifier(holder_id, "holder_id")
        ttl = _finite(ttl_seconds, "ttl_seconds", 0.001, _MAX_TTL)
        selected_now = time.time() if now is None else _finite(now, "now", 0.0, 1e20)
        expires = selected_now + ttl
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM leases WHERE owner_id=? AND resource_id=?",
                (owner, resource),
            ).fetchone()
            if row is None:
                token = 1
                connection.execute(
                    "INSERT INTO leases(owner_id, resource_id, holder_id, fencing_token, acquired_at, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (owner, resource, holder, token, selected_now, expires),
                )
            else:
                current_holder = row["holder_id"]
                current_expiry = float(row["expires_at"])
                current_token = int(row["fencing_token"])
                if current_holder == holder and current_expiry > selected_now:
                    token = current_token
                    connection.execute(
                        "UPDATE leases SET expires_at=? WHERE owner_id=? AND resource_id=?",
                        (expires, owner, resource),
                    )
                elif current_holder is None or current_expiry <= selected_now:
                    token = current_token + 1
                    connection.execute(
                        "UPDATE leases SET holder_id=?, fencing_token=?, acquired_at=?, expires_at=? WHERE owner_id=? AND resource_id=?",
                        (holder, token, selected_now, expires, owner, resource),
                    )
                else:
                    connection.execute("ROLLBACK")
                    return None
            connection.execute("COMMIT")
            return LeaseRecord(owner, resource, holder, token, selected_now, expires)
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def renew(
        self,
        lease: LeaseRecord,
        *,
        ttl_seconds: float = 60.0,
        now: float | None = None,
    ) -> LeaseRecord | None:
        if not isinstance(lease, LeaseRecord):
            raise ValueError("lease must be LeaseRecord.")
        ttl = _finite(ttl_seconds, "ttl_seconds", 0.001, _MAX_TTL)
        selected_now = time.time() if now is None else _finite(now, "now", 0.0, 1e20)
        expires = selected_now + ttl
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """UPDATE leases SET expires_at=?
                   WHERE owner_id=? AND resource_id=? AND holder_id=? AND fencing_token=? AND expires_at>?""",
                (expires, lease.owner_id, lease.resource_id, lease.holder_id, lease.fencing_token, selected_now),
            ).rowcount
            connection.execute("COMMIT")
        if updated != 1:
            return None
        return LeaseRecord(
            lease.owner_id,
            lease.resource_id,
            lease.holder_id,
            lease.fencing_token,
            lease.acquired_at,
            expires,
        )

    def release(self, lease: LeaseRecord) -> bool:
        if not isinstance(lease, LeaseRecord):
            raise ValueError("lease must be LeaseRecord.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """UPDATE leases SET holder_id=NULL, acquired_at=0, expires_at=0
                   WHERE owner_id=? AND resource_id=? AND holder_id=? AND fencing_token=?""",
                (lease.owner_id, lease.resource_id, lease.holder_id, lease.fencing_token),
            ).rowcount
            connection.execute("COMMIT")
        return updated == 1

    def current(self, *, owner_id: str, resource_id: str, now: float | None = None) -> LeaseRecord | None:
        owner = normalize_owner_id(owner_id)
        resource = _identifier(resource_id, "resource_id")
        selected_now = time.time() if now is None else _finite(now, "now", 0.0, 1e20)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM leases WHERE owner_id=? AND resource_id=?",
                (owner, resource),
            ).fetchone()
        if row is None or row["holder_id"] is None or float(row["expires_at"]) <= selected_now:
            return None
        return self._record(row)

    def validate_fence(self, lease: LeaseRecord, *, now: float | None = None) -> bool:
        if not isinstance(lease, LeaseRecord):
            raise ValueError("lease must be LeaseRecord.")
        current = self.current(owner_id=lease.owner_id, resource_id=lease.resource_id, now=now)
        return current is not None and current.holder_id == lease.holder_id and current.fencing_token == lease.fencing_token

    def delete_owner(self, *, owner_id: str) -> int:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            return connection.execute("DELETE FROM leases WHERE owner_id=?", (owner,)).rowcount


__all__ = ["LeaseCoordinator", "LeaseRecord"]
