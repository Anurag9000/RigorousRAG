"""Durable owner-scoped reviewed source-trust feature registry.

Trust records are explicit review/governance inputs, not inferred truth labels. Revisions
are immutable and content addressed; a separate owner/source head records which reviewed
revision is currently active. Re-activating an older revision therefore changes the head
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
                connection.execute(
                    """INSERT INTO source_trust_heads(owner_id,source_id,revision_id,updated_at)
                       VALUES(?,?,?,?)
                       ON CONFLICT(owner_id,source_id) DO UPDATE SET
                         revision_id=excluded.revision_id,updated_at=excluded.updated_at""",
                    (owner, features.source_id, revision_id, time.time()),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return self._from_row(row)

    def latest(self, owner_id: str, source_id: str) -> SourceTrustRevision | None:
        owner = normalize_owner_id(owner_id)
        source = str(source_id or "").strip()
        if not source or len(source) > 1000:
            raise ValueError("source_id is invalid")
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
        source = str(source_id or "").strip()
        if not source or len(source) > 1000 or not 1 <= limit <= 1000:
            raise ValueError("source_id or limit is invalid")
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


__all__ = ["SourceTrustRevision", "SourceTrustStore"]
