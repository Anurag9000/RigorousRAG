"""Owner-scoped feedback events and bounded active-learning exports."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from tools.security import normalize_owner_id

FeedbackKind = Literal[
    "answer_correct",
    "answer_incorrect",
    "citation_valid",
    "citation_invalid",
    "route_preference",
    "abstention_good",
    "abstention_bad",
]
_KINDS = frozenset(
    {
        "answer_correct",
        "answer_incorrect",
        "citation_valid",
        "citation_invalid",
        "route_preference",
        "abstention_good",
        "abstention_bad",
    }
)
_MAX_METADATA_BYTES = 64_000
_MAX_LIMIT = 10_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _finite(value: Any, label: str, minimum: float = 0.0, maximum: float = 1_000.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} is invalid.")
    return parsed


def _json(value: Mapping[str, Any] | None) -> str:
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


def _digest(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    text = _identifier(value, label, 20_000)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeedbackEvent:
    event_id: str
    owner_id: str
    kind: FeedbackKind
    subject_id: str
    query_sha256: str | None
    evidence_sha256: str | None
    weight: float
    metadata: Mapping[str, Any]
    created_at: float


@dataclass(frozen=True)
class ActiveLearningExample:
    kind: FeedbackKind
    subject_id: str
    weight: float
    metadata: Mapping[str, Any]
    query_sha256: str | None
    evidence_sha256: str | None


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    owner_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    query_sha256 TEXT,
                    evidence_sha256 TEXT,
                    weight REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_kind ON feedback(owner_id, kind, created_at DESC);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> FeedbackEvent:
        return FeedbackEvent(
            event_id=str(row["event_id"]),
            owner_id=str(row["owner_id"]),
            kind=str(row["kind"]),
            subject_id=str(row["subject_id"]),
            query_sha256=None if row["query_sha256"] is None else str(row["query_sha256"]),
            evidence_sha256=None if row["evidence_sha256"] is None else str(row["evidence_sha256"]),
            weight=float(row["weight"]),
            metadata=dict(json.loads(str(row["metadata_json"]))),
            created_at=float(row["created_at"]),
        )

    def put(
        self,
        *,
        owner_id: str,
        event_id: str,
        kind: FeedbackKind,
        subject_id: str,
        query: str | None = None,
        evidence: str | None = None,
        weight: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        created_at: float | None = None,
    ) -> FeedbackEvent:
        owner = normalize_owner_id(owner_id)
        event = _identifier(event_id, "event_id")
        if kind not in _KINDS:
            raise ValueError("kind is unsupported.")
        subject = _identifier(subject_id, "subject_id")
        selected_weight = _finite(weight, "weight", 0.000001, 1_000.0)
        timestamp = time.time() if created_at is None else _finite(created_at, "created_at", 0.0, 1e20)
        query_hash = _digest(query, "query")
        evidence_hash = _digest(evidence, "evidence")
        metadata_json = _json(metadata)
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO feedback(owner_id,event_id,kind,subject_id,query_sha256,evidence_sha256,weight,metadata_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (owner, event, kind, subject, query_hash, evidence_hash, selected_weight, metadata_json, timestamp),
            )
            row = connection.execute(
                "SELECT * FROM feedback WHERE owner_id=? AND event_id=?", (owner, event)
            ).fetchone()
        if row is None:
            raise RuntimeError("feedback write failed.")
        return self._record(row)

    def list(self, *, owner_id: str, kind: FeedbackKind | None = None, limit: int = 100) -> tuple[FeedbackEvent, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
            raise ValueError("limit is invalid.")
        if kind is not None and kind not in _KINDS:
            raise ValueError("kind is unsupported.")
        with self._connect() as connection:
            if kind is None:
                rows = connection.execute(
                    "SELECT * FROM feedback WHERE owner_id=? ORDER BY created_at DESC,event_id DESC LIMIT ?",
                    (owner, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM feedback WHERE owner_id=? AND kind=? ORDER BY created_at DESC,event_id DESC LIMIT ?",
                    (owner, kind, limit),
                ).fetchall()
        return tuple(self._record(row) for row in rows)

    def export_active_learning(self, *, owner_id: str, limit: int = 1_000) -> tuple[ActiveLearningExample, ...]:
        return tuple(
            ActiveLearningExample(
                kind=row.kind,
                subject_id=row.subject_id,
                weight=row.weight,
                metadata=row.metadata,
                query_sha256=row.query_sha256,
                evidence_sha256=row.evidence_sha256,
            )
            for row in self.list(owner_id=owner_id, limit=limit)
        )

    def delete_owner(self, *, owner_id: str) -> int:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM feedback WHERE owner_id=?", (owner,))
            return cursor.rowcount


__all__ = ["ActiveLearningExample", "FeedbackEvent", "FeedbackKind", "FeedbackStore"]
