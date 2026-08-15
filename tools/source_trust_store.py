"""Durable owner-scoped reviewed source-trust feature registry.

Trust records are explicit review/governance inputs, not inferred truth labels. Updates
are append-only revisions so downstream policy decisions can cite the exact feature
fingerprint that was active when an answer was produced.
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
                """
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
        return SourceTrustRevision(owner, revision_id, features, reviewer, basis, created_at)

    def latest(self, owner_id: str, source_id: str) -> SourceTrustRevision | None:
        owner = normalize_owner_id(owner_id)
        source = str(source_id or "").strip()
        if not source or len(source) > 1000:
            raise ValueError("source_id is invalid")
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM source_trust_revisions
                   WHERE owner_id=? AND source_id=?
                   ORDER BY created_at DESC,revision_id DESC LIMIT 1""",
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
                """SELECT r.* FROM source_trust_revisions r
                   JOIN (
                     SELECT source_id, MAX(created_at) AS max_created
                     FROM source_trust_revisions WHERE owner_id=? GROUP BY source_id
                   ) latest ON latest.source_id=r.source_id AND latest.max_created=r.created_at
                   WHERE r.owner_id=? ORDER BY r.created_at DESC,r.source_id LIMIT ?""",
                (owner, owner, limit),
            ).fetchall()
        # Same timestamp collisions are rare but possible; deduplicate deterministically.
        output: list[SourceTrustRevision] = []
        seen: set[str] = set()
        for row in rows:
            revision = self._from_row(row)
            if revision.features.source_id in seen:
                continue
            seen.add(revision.features.source_id)
            output.append(revision)
        return tuple(output)

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
