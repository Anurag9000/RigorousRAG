"""Durable immutable-version SQLite store for owner/project-scoped hydrology artifacts."""
from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any

from tools.hydrology_store import HydrologyArtifactEnvelope, HydrologyArtifactSummary, strict_json
from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_KINDS = frozenset({"topology", "engineering_package", "retrieval_plan", "evidence_projection", "evidence_report"})


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _digest(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


def _kind(value: str) -> str:
    cleaned = _text(value, "kind", 64).lower()
    if cleaned not in _KINDS:
        raise ValueError("unsupported hydrology artifact kind")
    return cleaned


def _safe_database_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("hydrology database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError("hydrology database path could not be inspected") from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("hydrology database path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


class SQLiteHydrologyArtifactStore:
    """Append-only artifact versions plus an optimistic-concurrency current pointer."""

    def __init__(self, path: str | Path) -> None:
        self.path = _safe_database_path(path)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS hydrology_artifact_versions (
                    owner_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    logical_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, project_id, kind, logical_id, version)
                );
                CREATE INDEX IF NOT EXISTS hydrology_versions_fingerprint_idx
                  ON hydrology_artifact_versions(owner_id, project_id, fingerprint);
                CREATE INDEX IF NOT EXISTS hydrology_versions_project_time_idx
                  ON hydrology_artifact_versions(owner_id, project_id, created_at DESC, kind, logical_id, version DESC);
                CREATE TABLE IF NOT EXISTS hydrology_artifact_current (
                    owner_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    logical_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, project_id, kind, logical_id),
                    FOREIGN KEY(owner_id, project_id, kind, logical_id, version)
                      REFERENCES hydrology_artifact_versions(owner_id, project_id, kind, logical_id, version)
                );
                """
            )

    @staticmethod
    def _summary(row: sqlite3.Row, *, is_current: bool) -> HydrologyArtifactSummary:
        return HydrologyArtifactSummary(
            str(row["owner_id"]), str(row["project_id"]), str(row["kind"]), str(row["logical_id"]),
            str(row["fingerprint"]), int(row["version"]), float(row["created_at"]), is_current,
        )

    def put(self, envelope: HydrologyArtifactEnvelope, *, expected_current_fingerprint: str | None = None) -> HydrologyArtifactSummary:
        if not isinstance(envelope, HydrologyArtifactEnvelope):
            raise TypeError("envelope must be HydrologyArtifactEnvelope")
        expected = _digest(expected_current_fingerprint, "expected_current_fingerprint")
        payload = strict_json(envelope.payload)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = connection.execute(
                    "SELECT version,fingerprint FROM hydrology_artifact_current WHERE owner_id=? AND project_id=? AND kind=? AND logical_id=?",
                    (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id),
                ).fetchone()
                if current is None:
                    if expected is not None:
                        raise RuntimeError("hydrology artifact optimistic concurrency check failed")
                    next_version = 1
                else:
                    current_version, current_fingerprint = int(current["version"]), str(current["fingerprint"])
                    if expected is not None and current_fingerprint != expected:
                        raise RuntimeError("hydrology artifact optimistic concurrency check failed")
                    if current_fingerprint == envelope.fingerprint:
                        row = connection.execute(
                            "SELECT owner_id,project_id,kind,logical_id,version,fingerprint,created_at FROM hydrology_artifact_versions WHERE owner_id=? AND project_id=? AND kind=? AND logical_id=? AND version=?",
                            (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, current_version),
                        ).fetchone()
                        if row is None:
                            raise RuntimeError("hydrology current pointer references a missing version")
                        connection.commit()
                        return self._summary(row, is_current=True)
                    next_version = current_version + 1
                connection.execute(
                    "INSERT INTO hydrology_artifact_versions (owner_id,project_id,kind,logical_id,version,fingerprint,schema_version,payload,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, next_version, envelope.fingerprint, envelope.schema_version, payload, envelope.created_at),
                )
                now = time.time()
                if current is None:
                    connection.execute(
                        "INSERT INTO hydrology_artifact_current (owner_id,project_id,kind,logical_id,version,fingerprint,updated_at) VALUES(?,?,?,?,?,?,?)",
                        (envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, next_version, envelope.fingerprint, now),
                    )
                else:
                    cursor = connection.execute(
                        "UPDATE hydrology_artifact_current SET version=?,fingerprint=?,updated_at=? WHERE owner_id=? AND project_id=? AND kind=? AND logical_id=? AND version=?",
                        (next_version, envelope.fingerprint, now, envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, int(current["version"])),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("hydrology artifact concurrent update detected")
                connection.commit()
                return HydrologyArtifactSummary(envelope.owner_id, envelope.project_id, envelope.kind, envelope.logical_id, envelope.fingerprint, next_version, envelope.created_at, True)
            except Exception:
                connection.rollback()
                raise

    def get(self, owner_id: str, project_id: str, kind: str, logical_id: str, *, fingerprint: str | None = None) -> HydrologyArtifactEnvelope:
        owner = normalize_owner_id(owner_id)
        project, artifact_kind, logical = _text(project_id, "project_id", 256), _kind(kind), _text(logical_id, "logical_id", 500)
        requested = _digest(fingerprint, "fingerprint")
        with self._connect() as connection:
            if requested is None:
                row = connection.execute(
                    """SELECT v.fingerprint,v.schema_version,v.payload,v.created_at FROM hydrology_artifact_current c
                       JOIN hydrology_artifact_versions v ON v.owner_id=c.owner_id AND v.project_id=c.project_id AND v.kind=c.kind AND v.logical_id=c.logical_id AND v.version=c.version
                       WHERE c.owner_id=? AND c.project_id=? AND c.kind=? AND c.logical_id=?""",
                    (owner, project, artifact_kind, logical),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT fingerprint,schema_version,payload,created_at FROM hydrology_artifact_versions WHERE owner_id=? AND project_id=? AND kind=? AND logical_id=? AND fingerprint=? ORDER BY version DESC LIMIT 1",
                    (owner, project, artifact_kind, logical, requested),
                ).fetchone()
        if row is None:
            raise KeyError(logical)
        payload = json.loads(str(row["payload"]))
        if not isinstance(payload, dict):
            raise RuntimeError("stored hydrology artifact payload is not an object")
        return HydrologyArtifactEnvelope(owner, project, artifact_kind, logical, str(row["fingerprint"]), payload, int(row["schema_version"]), float(row["created_at"]))

    def list(self, owner_id: str, project_id: str, *, kind: str | None = None, include_history: bool = False, limit: int = 200) -> tuple[HydrologyArtifactSummary, ...]:
        owner, project = normalize_owner_id(owner_id), _text(project_id, "project_id", 256)
        artifact_kind = _kind(kind) if kind is not None else None
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit is invalid")
        params: list[Any] = [owner, project]
        kind_clause = ""
        if artifact_kind is not None:
            kind_clause = " AND v.kind=?"
            params.append(artifact_kind)
        with self._connect() as connection:
            if include_history:
                rows = connection.execute(
                    f"""SELECT v.owner_id,v.project_id,v.kind,v.logical_id,v.version,v.fingerprint,v.created_at,CASE WHEN c.version=v.version THEN 1 ELSE 0 END AS is_current
                        FROM hydrology_artifact_versions v LEFT JOIN hydrology_artifact_current c ON c.owner_id=v.owner_id AND c.project_id=v.project_id AND c.kind=v.kind AND c.logical_id=v.logical_id
                        WHERE v.owner_id=? AND v.project_id=?{kind_clause} ORDER BY v.created_at DESC,v.kind,v.logical_id,v.version DESC LIMIT ?""",
                    (*params, limit),
                ).fetchall()
                return tuple(self._summary(row, is_current=bool(row["is_current"])) for row in rows)
            rows = connection.execute(
                f"""SELECT v.owner_id,v.project_id,v.kind,v.logical_id,v.version,v.fingerprint,v.created_at FROM hydrology_artifact_current c
                    JOIN hydrology_artifact_versions v ON v.owner_id=c.owner_id AND v.project_id=c.project_id AND v.kind=c.kind AND v.logical_id=c.logical_id AND v.version=c.version
                    WHERE v.owner_id=? AND v.project_id=?{kind_clause} ORDER BY v.created_at DESC,v.kind,v.logical_id LIMIT ?""",
                (*params, limit),
            ).fetchall()
        return tuple(self._summary(row, is_current=True) for row in rows)


__all__ = ["SQLiteHydrologyArtifactStore"]
