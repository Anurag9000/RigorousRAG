"""Persistent owner-scoped human-review queue with lease/fencing semantics."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from tools.review_routing import ReviewDecision
from tools.security import normalize_owner_id

_MAX_METADATA_BYTES = 64_000
_MAX_LIMIT = 1_000
_MAX_TTL = 86_400.0
_STATES = frozenset({"pending", "claimed", "resolved", "cancelled"})


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > 500 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _finite(value: Any, label: str, minimum: float = 0.0, maximum: float = 1e20) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} is invalid.")
    return parsed


def _metadata(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, Mapping) or len(value) > 100:
        raise ValueError("metadata must be a bounded mapping.")
    try:
        rendered = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("metadata must be JSON serializable.") from exc
    if len(rendered.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds the byte limit.")
    return rendered


def _query_hash(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 20_000:
        raise ValueError("query must be a bounded non-empty string.")
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReviewRecord:
    request_id: str
    owner_id: str
    state: Literal["pending", "claimed", "resolved", "cancelled"]
    priority: float
    reasons: tuple[str, ...]
    query_sha256: str | None
    metadata: Mapping[str, Any]
    created_at: float
    updated_at: float
    reviewer_id: str | None
    lease_token: int
    lease_expires_at: float
    resolution: str | None


class ReviewStore:
    """SQLite review queue with idempotent enqueue and stale-reviewer fencing."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reviews (
                    request_id TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority REAL NOT NULL,
                    reasons_json TEXT NOT NULL,
                    query_sha256 TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    reviewer_id TEXT,
                    lease_token INTEGER NOT NULL DEFAULT 0,
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    resolution TEXT,
                    PRIMARY KEY(owner_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_reviews_queue ON reviews(owner_id, state, priority DESC, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_reviews_lease ON reviews(state, lease_expires_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> ReviewRecord:
        return ReviewRecord(
            request_id=str(row["request_id"]),
            owner_id=str(row["owner_id"]),
            state=str(row["state"]),
            priority=float(row["priority"]),
            reasons=tuple(json.loads(str(row["reasons_json"]))),
            query_sha256=None if row["query_sha256"] is None else str(row["query_sha256"]),
            metadata=dict(json.loads(str(row["metadata_json"]))),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            reviewer_id=None if row["reviewer_id"] is None else str(row["reviewer_id"]),
            lease_token=int(row["lease_token"]),
            lease_expires_at=float(row["lease_expires_at"]),
            resolution=None if row["resolution"] is None else str(row["resolution"]),
        )

    def enqueue(
        self,
        *,
        owner_id: str,
        request_id: str,
        decision: ReviewDecision,
        query: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> ReviewRecord:
        owner = normalize_owner_id(owner_id)
        request = _identifier(request_id, "request_id")
        if not isinstance(decision, ReviewDecision) or decision.route != "human_review":
            raise ValueError("decision must be a human_review ReviewDecision.")
        selected_now = time.time() if now is None else _finite(now, "now")
        reasons_json = json.dumps(list(decision.reasons), separators=(",", ":"))
        metadata_json = _metadata(metadata)
        query_sha256 = _query_hash(query)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM reviews WHERE owner_id=? AND request_id=?", (owner, request)
            ).fetchone()
            if existing is not None:
                connection.execute("COMMIT")
                return self._record(existing)
            connection.execute(
                """INSERT INTO reviews(request_id, owner_id, state, priority, reasons_json, query_sha256, metadata_json,
                   created_at, updated_at, reviewer_id, lease_token, lease_expires_at, resolution)
                   VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, NULL, 0, 0, NULL)""",
                (request, owner, decision.priority, reasons_json, query_sha256, metadata_json, selected_now, selected_now),
            )
            row = connection.execute(
                "SELECT * FROM reviews WHERE owner_id=? AND request_id=?", (owner, request)
            ).fetchone()
            connection.execute("COMMIT")
            if row is None:
                raise RuntimeError("review enqueue failed.")
            return self._record(row)
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def claim_next(
        self,
        *,
        owner_id: str,
        reviewer_id: str,
        ttl_seconds: float = 300.0,
        now: float | None = None,
    ) -> ReviewRecord | None:
        owner = normalize_owner_id(owner_id)
        reviewer = _identifier(reviewer_id, "reviewer_id")
        ttl = _finite(ttl_seconds, "ttl_seconds", 0.001, _MAX_TTL)
        selected_now = time.time() if now is None else _finite(now, "now")
        expires = selected_now + ttl
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """UPDATE reviews SET state='pending', reviewer_id=NULL, lease_expires_at=0, updated_at=?
                   WHERE owner_id=? AND state='claimed' AND lease_expires_at<=?""",
                (selected_now, owner, selected_now),
            )
            row = connection.execute(
                """SELECT * FROM reviews WHERE owner_id=? AND state='pending'
                   ORDER BY priority DESC, created_at ASC, request_id ASC LIMIT 1""",
                (owner,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            token = int(row["lease_token"]) + 1
            connection.execute(
                """UPDATE reviews SET state='claimed', reviewer_id=?, lease_token=?, lease_expires_at=?, updated_at=?
                   WHERE owner_id=? AND request_id=? AND state='pending'""",
                (reviewer, token, expires, selected_now, owner, row["request_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM reviews WHERE owner_id=? AND request_id=?", (owner, row["request_id"])
            ).fetchone()
            connection.execute("COMMIT")
            return None if claimed is None else self._record(claimed)
        except Exception:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def renew(self, record: ReviewRecord, *, ttl_seconds: float = 300.0, now: float | None = None) -> ReviewRecord | None:
        if not isinstance(record, ReviewRecord) or record.state != "claimed" or record.reviewer_id is None:
            raise ValueError("record must be a claimed ReviewRecord.")
        selected_now = time.time() if now is None else _finite(now, "now")
        ttl = _finite(ttl_seconds, "ttl_seconds", 0.001, _MAX_TTL)
        expires = selected_now + ttl
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """UPDATE reviews SET lease_expires_at=?, updated_at=? WHERE owner_id=? AND request_id=?
                   AND state='claimed' AND reviewer_id=? AND lease_token=? AND lease_expires_at>?""",
                (expires, selected_now, record.owner_id, record.request_id, record.reviewer_id, record.lease_token, selected_now),
            ).rowcount
            row = connection.execute(
                "SELECT * FROM reviews WHERE owner_id=? AND request_id=?", (record.owner_id, record.request_id)
            ).fetchone()
            connection.execute("COMMIT")
        return self._record(row) if updated == 1 and row is not None else None

    def resolve(self, record: ReviewRecord, *, resolution: str, now: float | None = None) -> bool:
        if not isinstance(record, ReviewRecord) or record.state != "claimed" or record.reviewer_id is None:
            raise ValueError("record must be a claimed ReviewRecord.")
        selected_resolution = _identifier(resolution, "resolution")
        selected_now = time.time() if now is None else _finite(now, "now")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """UPDATE reviews SET state='resolved', resolution=?, lease_expires_at=0, updated_at=?
                   WHERE owner_id=? AND request_id=? AND state='claimed' AND reviewer_id=? AND lease_token=? AND lease_expires_at>?""",
                (selected_resolution, selected_now, record.owner_id, record.request_id, record.reviewer_id, record.lease_token, selected_now),
            ).rowcount
            connection.execute("COMMIT")
        return updated == 1

    def cancel(self, *, owner_id: str, request_id: str, now: float | None = None) -> bool:
        owner = normalize_owner_id(owner_id)
        request = _identifier(request_id, "request_id")
        selected_now = time.time() if now is None else _finite(now, "now")
        with self._connect() as connection:
            updated = connection.execute(
                """UPDATE reviews SET state='cancelled', lease_expires_at=0, updated_at=?
                   WHERE owner_id=? AND request_id=? AND state IN ('pending','claimed')""",
                (selected_now, owner, request),
            ).rowcount
        return updated == 1

    def get(self, *, owner_id: str, request_id: str) -> ReviewRecord | None:
        owner = normalize_owner_id(owner_id)
        request = _identifier(request_id, "request_id")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM reviews WHERE owner_id=? AND request_id=?", (owner, request)
            ).fetchone()
        return None if row is None else self._record(row)

    def list(self, *, owner_id: str, state: str | None = None, limit: int = 100) -> tuple[ReviewRecord, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
            raise ValueError("limit is invalid.")
        params: list[Any] = [owner]
        where = "owner_id=?"
        if state is not None:
            if state not in _STATES:
                raise ValueError("state is invalid.")
            where += " AND state=?"
            params.append(state)
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM reviews WHERE {where} ORDER BY priority DESC, created_at ASC LIMIT ?", params
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def delete_owner(self, *, owner_id: str) -> int:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            return connection.execute("DELETE FROM reviews WHERE owner_id=?", (owner,)).rowcount


__all__ = ["ReviewRecord", "ReviewStore"]
