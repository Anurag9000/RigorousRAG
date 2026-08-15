"""Durable owner-scoped old→new artifact replacement lineage."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.dependency_invalidation import DependencyRef
from tools.security import normalize_owner_id

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
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("replacement ledger path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class ArtifactReplacement:
    old: DependencyRef
    new: DependencyRef
    reason: str
    triggering_event_sha256: str
    replacement_sha256: str
    created_at: float


class ArtifactReplacementStore:
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
                CREATE TABLE IF NOT EXISTS artifact_replacements (
                    owner_id TEXT NOT NULL,
                    old_kind TEXT NOT NULL,
                    old_id TEXT NOT NULL,
                    old_key CHAR(64) NOT NULL,
                    new_kind TEXT NOT NULL,
                    new_id TEXT NOT NULL,
                    new_key CHAR(64) NOT NULL,
                    reason TEXT NOT NULL,
                    event_sha256 CHAR(64) NOT NULL,
                    replacement_sha256 CHAR(64) NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, replacement_sha256)
                );
                CREATE INDEX IF NOT EXISTS artifact_replacements_old_idx
                  ON artifact_replacements(owner_id, old_key, created_at DESC, replacement_sha256);
                CREATE INDEX IF NOT EXISTS artifact_replacements_new_idx
                  ON artifact_replacements(owner_id, new_key, created_at DESC, replacement_sha256);
                """
            )

    def put(
        self,
        owner_id: str,
        *,
        old: DependencyRef,
        new: DependencyRef,
        reason: str,
        triggering_event_sha256: str,
    ) -> ArtifactReplacement:
        owner = normalize_owner_id(owner_id)
        if not isinstance(old, DependencyRef) or not isinstance(new, DependencyRef):
            raise TypeError("old/new must be DependencyRef")
        if old.kind != new.kind:
            raise ValueError("replacement artifacts must preserve artifact kind")
        if old == new:
            raise ValueError("replacement must identify a new artifact")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 5000:
            raise ValueError("reason is invalid")
        event = triggering_event_sha256.strip().lower()
        if len(event) != 64 or any(ch not in "0123456789abcdef" for ch in event):
            raise ValueError("triggering_event_sha256 must be SHA-256")
        payload = {
            "owner_id": owner,
            "old": {"kind": old.kind, "resource_id": old.resource_id},
            "new": {"kind": new.kind, "resource_id": new.resource_id},
            "reason": reason.strip(),
            "event_sha256": event,
        }
        replacement_sha = hashlib.sha256(_canonical(payload)).hexdigest()
        created_at = time.time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO artifact_replacements
                   (owner_id,old_kind,old_id,old_key,new_kind,new_id,new_key,reason,event_sha256,replacement_sha256,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (owner, old.kind, old.resource_id, old.key, new.kind, new.resource_id, new.key, reason.strip(), event, replacement_sha, created_at),
            )
        return ArtifactReplacement(old, new, reason.strip(), event, replacement_sha, created_at)

    def latest(self, owner_id: str, old: DependencyRef) -> ArtifactReplacement | None:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM artifact_replacements WHERE owner_id=? AND old_key=?
                   ORDER BY created_at DESC,replacement_sha256 DESC LIMIT 1""",
                (owner, old.key),
            ).fetchone()
        if row is None:
            return None
        return self._from_row(row)

    def chain(self, owner_id: str, start: DependencyRef, *, max_depth: int = 64) -> tuple[ArtifactReplacement, ...]:
        if not 1 <= max_depth <= 256:
            raise ValueError("max_depth is invalid")
        output: list[ArtifactReplacement] = []
        current = start
        seen = {current.key}
        for _ in range(max_depth):
            replacement = self.latest(owner_id, current)
            if replacement is None:
                break
            if replacement.new.key in seen:
                raise RuntimeError("replacement lineage contains a cycle")
            output.append(replacement)
            seen.add(replacement.new.key)
            current = replacement.new
        return tuple(output)

    def current(self, owner_id: str, start: DependencyRef) -> DependencyRef:
        chain = self.chain(owner_id, start)
        return chain[-1].new if chain else start

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ArtifactReplacement:
        return ArtifactReplacement(
            old=DependencyRef(str(row["old_kind"]), str(row["old_id"])),
            new=DependencyRef(str(row["new_kind"]), str(row["new_id"])),
            reason=str(row["reason"]),
            triggering_event_sha256=str(row["event_sha256"]),
            replacement_sha256=str(row["replacement_sha256"]),
            created_at=float(row["created_at"]),
        )


__all__ = ["ArtifactReplacement", "ArtifactReplacementStore"]
