"""Durable owner-scoped reviewed source-trust feature registry.

Trust records are explicit review/governance inputs, not inferred truth labels. Revisions
are immutable and content addressed; a separate owner/source head records which reviewed
revision is currently active. Every head transition is written to a durable activation
outbox in the same transaction so downstream invalidation can be retried idempotently
without mutating historical review content.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.security import normalize_owner_id
from tools.source_trust import SourceTrustFeatures

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        attrs = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attrs & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("source trust registry path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _bounded_source_id(value: str) -> str:
    source = str(value or "").strip()
    if not source or len(source) > 1000 or any(ord(ch) < 32 or ord(ch) == 127 for ch in source):
        raise ValueError("source_id is invalid")
    return source


@dataclass(frozen=True)
class SourceTrustRevision:
    owner_id: str
    revision_id: str
    features: SourceTrustFeatures
    reviewer_id: str
    review_basis: str
    created_at: float

    @property
    def fingerprint(self) -> str:
        return self.revision_id


@dataclass(frozen=True)
class SourceTrustActivation:
    owner_id: str
    activation_id: str
    source_id: str
    previous_revision_id: str
    revision_id: str
    activated_at: float
    invalidation_completed_at: float | None = None
    last_error: str = ""

    @property
    def pending(self) -> bool:
        return self.invalidation_completed_at is None


class SourceTrustStore:
    def __init__(self, path: str | Path) -> None:
        self.path = _safe_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_trust_revisions (
                    owner_id TEXT NOT NULL,
                    revision_id CHAR(64) NOT NULL,
                    source_id TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    review_basis TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, revision_id)
                );
                CREATE INDEX IF NOT EXISTS source_trust_source_idx
                  ON source_trust_revisions(owner_id, source_id, created_at DESC, revision_id DESC);

                CREATE TABLE IF NOT EXISTS source_trust_heads (
                    owner_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    revision_id CHAR(64) NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, source_id)
                );
                CREATE INDEX IF NOT EXISTS source_trust_heads_updated_idx
                  ON source_trust_heads(owner_id, updated_at DESC, source_id);

                CREATE TABLE IF NOT EXISTS source_trust_activations (
                    owner_id TEXT NOT NULL,
                    activation_id CHAR(64) NOT NULL,
                    source_id TEXT NOT NULL,
                    previous_revision_id TEXT NOT NULL DEFAULT '',
                    revision_id CHAR(64) NOT NULL,
                    activated_at REAL NOT NULL,
                    invalidation_completed_at REAL,
                    last_error TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY(owner_id, activation_id)
                );
                CREATE INDEX IF NOT EXISTS source_trust_activations_pending_idx
                  ON source_trust_activations(owner_id, invalidation_completed_at, activated_at, activation_id);
                CREATE INDEX IF NOT EXISTS source_trust_activations_source_idx
                  ON source_trust_activations(owner_id, source_id, activated_at, activation_id);
                """
            )
            # Backfill heads for stores created before explicit head tracking. Correlated
            # NOT EXISTS deterministically selects the newest immutable revision.
            connection.execute(
                """INSERT OR IGNORE INTO source_trust_heads(owner_id,source_id,revision_id,updated_at)
                   SELECT r.owner_id,r.source_id,r.revision_id,r.created_at
                   FROM source_trust_revisions r
                   WHERE NOT EXISTS (
                     SELECT 1 FROM source_trust_revisions newer
                     WHERE newer.owner_id=r.owner_id AND newer.source_id=r.source_id
                       AND (newer.created_at>r.created_at OR
                            (newer.created_at=r.created_at AND newer.revision_id>r.revision_id))
                   )"""
            )

    @staticmethod
    def _activation_id(
        owner_id: str,
        source_id: str,
        previous_revision_id: str,
        revision_id: str,
    ) -> str:
        payload = {
            "owner_id": owner_id,
            "source_id": source_id,
            "previous_revision_id": previous_revision_id,
            "revision_id": revision_id,
            # Activation is an occurrence, not a content identity. A nonce allows a later
            # legitimate A->B transition to differ from an earlier A->B transition.
            "nonce": uuid.uuid4().hex,
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def put(
        self,
        owner_id: str,
        features: SourceTrustFeatures,
        *,
        reviewer_id: str,
        review_basis: str,
    ) -> SourceTrustRevision:
        owner = normalize_owner_id(owner_id)
        if not isinstance(features, SourceTrustFeatures):
            raise TypeError("features must be SourceTrustFeatures")
        reviewer = str(reviewer_id or "").strip()
        basis = str(review_basis or "").strip()
        if not reviewer or len(reviewer) > 256 or not basis or len(basis) > 5000:
            raise ValueError("reviewer_id and review_basis are required and bounded")
        payload = {
            "owner_id": owner,
            "features": asdict(features),
            "reviewer_id": reviewer,
            "review_basis": basis,
        }
        revision_id = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                head = connection.execute(
                    "SELECT revision_id FROM source_trust_heads WHERE owner_id=? AND source_id=?",
                    (owner, features.source_id),
                ).fetchone()
                previous_revision_id = str(head["revision_id"]) if head is not None else ""
                connection.execute(
                    """INSERT OR IGNORE INTO source_trust_revisions
                       (owner_id,revision_id,source_id,features_json,reviewer_id,review_basis,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (
                        owner,
                        revision_id,
                        features.source_id,
                        _canonical(asdict(features)),
                        reviewer,
                        basis,
                        created_at,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM source_trust_revisions WHERE owner_id=? AND revision_id=?",
                    (owner, revision_id),
                ).fetchone()
                if row is None:
                    raise RuntimeError("source trust revision persistence failed")
                if previous_revision_id != revision_id:
                    activation_id = self._activation_id(
                        owner,
                        features.source_id,
                        previous_revision_id,
                        revision_id,
                    )
                    activated_at = time.time()
                    connection.execute(
                        """INSERT INTO source_trust_activations
                           (owner_id,activation_id,source_id,previous_revision_id,revision_id,activated_at)
                           VALUES(?,?,?,?,?,?)""",
                        (
                            owner,
                            activation_id,
                            features.source_id,
                            previous_revision_id,
                            revision_id,
                            activated_at,
                        ),
                    )
                    connection.execute(
                        """INSERT INTO source_trust_heads(owner_id,source_id,revision_id,updated_at)
                           VALUES(?,?,?,?)
                           ON CONFLICT(owner_id,source_id) DO UPDATE SET
                             revision_id=excluded.revision_id,updated_at=excluded.updated_at""",
                        (owner, features.source_id, revision_id, activated_at),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._from_row(row)

    def latest(self, owner_id: str, source_id: str) -> SourceTrustRevision | None:
        owner = normalize_owner_id(owner_id)
        source = _bounded_source_id(source_id)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT r.* FROM source_trust_heads h
                   JOIN source_trust_revisions r
                     ON r.owner_id=h.owner_id AND r.revision_id=h.revision_id
                   WHERE h.owner_id=? AND h.source_id=? LIMIT 1""",
                (owner, source),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def history(self, owner_id: str, source_id: str, *, limit: int = 100) -> tuple[SourceTrustRevision, ...]:
        owner = normalize_owner_id(owner_id)
        source = _bounded_source_id(source_id)
        if not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM source_trust_revisions
                   WHERE owner_id=? AND source_id=?
                   ORDER BY created_at DESC,revision_id DESC LIMIT ?""",
                (owner, source, limit),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def list_latest(self, owner_id: str, *, limit: int = 500) -> tuple[SourceTrustRevision, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 5000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT r.* FROM source_trust_heads h
                   JOIN source_trust_revisions r
                     ON r.owner_id=h.owner_id AND r.revision_id=h.revision_id
                   WHERE h.owner_id=? ORDER BY h.updated_at DESC,h.source_id LIMIT ?""",
                (owner, limit),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    def pending_activations(
        self,
        owner_id: str,
        *,
        source_id: str | None = None,
        limit: int = 1000,
    ) -> tuple[SourceTrustActivation, ...]:
        owner = normalize_owner_id(owner_id)
        if not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        clauses = ["owner_id=?", "invalidation_completed_at IS NULL"]
        params: list[Any] = [owner]
        if source_id is not None:
            clauses.append("source_id=?")
            params.append(_bounded_source_id(source_id))
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM source_trust_activations WHERE "
                + " AND ".join(clauses)
                + " ORDER BY activated_at,activation_id LIMIT ?",
                tuple(params),
            ).fetchall()
        return tuple(self._activation_from_row(row) for row in rows)

    def mark_activation_completed(self, owner_id: str, activation_id: str) -> None:
        owner = normalize_owner_id(owner_id)
        activation = str(activation_id or "").strip().lower()
        if len(activation) != 64 or any(ch not in "0123456789abcdef" for ch in activation):
            raise ValueError("activation_id must be SHA-256")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """UPDATE source_trust_activations
                   SET invalidation_completed_at=?,last_error=''
                   WHERE owner_id=? AND activation_id=? AND invalidation_completed_at IS NULL""",
                (time.time(), owner, activation),
            )
            if cursor.rowcount not in {0, 1}:
                raise RuntimeError("source trust activation completion was ambiguous")

    def mark_activation_failed(self, owner_id: str, activation_id: str, error_type: str) -> None:
        owner = normalize_owner_id(owner_id)
        activation = str(activation_id or "").strip().lower()
        if len(activation) != 64 or any(ch not in "0123456789abcdef" for ch in activation):
            raise ValueError("activation_id must be SHA-256")
        error = str(error_type or "unknown")[:200]
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE source_trust_activations SET last_error=?
                   WHERE owner_id=? AND activation_id=? AND invalidation_completed_at IS NULL""",
                (error, owner, activation),
            )

    def activation_history(
        self,
        owner_id: str,
        source_id: str,
        *,
        limit: int = 100,
    ) -> tuple[SourceTrustActivation, ...]:
        owner = normalize_owner_id(owner_id)
        source = _bounded_source_id(source_id)
        if not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM source_trust_activations
                   WHERE owner_id=? AND source_id=?
                   ORDER BY activated_at DESC,activation_id DESC LIMIT ?""",
                (owner, source, limit),
            ).fetchall()
        return tuple(self._activation_from_row(row) for row in rows)

    @staticmethod
    def _from_row(row: sqlite3.Row) -> SourceTrustRevision:
        return SourceTrustRevision(
            owner_id=str(row["owner_id"]),
            revision_id=str(row["revision_id"]),
            features=SourceTrustFeatures(**json.loads(str(row["features_json"]))),
            reviewer_id=str(row["reviewer_id"]),
            review_basis=str(row["review_basis"]),
            created_at=float(row["created_at"]),
        )

    @staticmethod
    def _activation_from_row(row: sqlite3.Row) -> SourceTrustActivation:
        completed = row["invalidation_completed_at"]
        return SourceTrustActivation(
            owner_id=str(row["owner_id"]),
            activation_id=str(row["activation_id"]),
            source_id=str(row["source_id"]),
            previous_revision_id=str(row["previous_revision_id"]),
            revision_id=str(row["revision_id"]),
            activated_at=float(row["activated_at"]),
            invalidation_completed_at=float(completed) if completed is not None else None,
            last_error=str(row["last_error"]),
        )


__all__ = ["SourceTrustActivation", "SourceTrustRevision", "SourceTrustStore"]
