"""Durable owner-scoped SQLite backend for research projects and sessions."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from tools.research_workspace import ResearchProject, ResearchSession, ResearchTurn, CorpusBinding
from tools.security import normalize_owner_id

_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _safe_database_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    if len(str(absolute)) > 4096:
        raise ValueError("workspace database path is too long")
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError("workspace database path could not be inspected") from exc
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or bool(attributes & _WINDOWS_REPARSE_POINT):
            raise RuntimeError("workspace database path may not traverse symlinks/reparse points")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    return absolute


def _strict_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _project_from_payload(payload: Mapping[str, Any]) -> ResearchProject:
    corpora = tuple(CorpusBinding(**item) for item in payload.get("corpora", ()))
    return ResearchProject(
        owner_id=payload["owner_id"],
        project_id=payload["project_id"],
        title=payload["title"],
        research_question=payload["research_question"],
        corpora=corpora,
        tags=tuple(payload.get("tags", ())),
        created_at=float(payload.get("created_at", time.time())),
        archived=bool(payload.get("archived", False)),
    )


def _session_from_payload(payload: Mapping[str, Any]) -> ResearchSession:
    turns = tuple(ResearchTurn(**item) for item in payload.get("turns", ()))
    closed_raw = payload.get("closed_at")
    return ResearchSession(
        owner_id=payload["owner_id"],
        project_id=payload["project_id"],
        session_id=payload["session_id"],
        turns=turns,
        created_at=float(payload.get("created_at", time.time())),
        closed_at=float(closed_raw) if closed_raw is not None else None,
    )


class SQLiteResearchWorkspaceStore:
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
                CREATE TABLE IF NOT EXISTS research_projects (
                    owner_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, project_id)
                );
                CREATE INDEX IF NOT EXISTS research_projects_owner_updated_idx
                  ON research_projects(owner_id, updated_at DESC, project_id);
                CREATE TABLE IF NOT EXISTS research_sessions (
                    owner_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    version INTEGER NOT NULL,
                    PRIMARY KEY(owner_id, session_id),
                    FOREIGN KEY(owner_id, project_id)
                      REFERENCES research_projects(owner_id, project_id)
                      ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS research_sessions_project_updated_idx
                  ON research_sessions(owner_id, project_id, updated_at DESC, session_id);
                """
            )

    def create_project(self, project: ResearchProject) -> None:
        if not isinstance(project, ResearchProject):
            raise TypeError("project must be ResearchProject")
        payload = _strict_json({
            "owner_id": project.owner_id,
            "project_id": project.project_id,
            "title": project.title,
            "research_question": project.research_question,
            "corpora": [vars(item) for item in project.corpora],
            "tags": list(project.tags),
            "created_at": project.created_at,
            "archived": project.archived,
        })
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT fingerprint FROM research_projects WHERE owner_id=? AND project_id=?",
                    (project.owner_id, project.project_id),
                ).fetchone()
                if existing is not None:
                    if str(existing["fingerprint"]) != project.fingerprint:
                        raise ValueError("project already exists with different content")
                    connection.commit()
                    return
                connection.execute(
                    """INSERT INTO research_projects
                       (owner_id,project_id,fingerprint,payload,created_at,updated_at,version)
                       VALUES(?,?,?,?,?,?,1)""",
                    (project.owner_id, project.project_id, project.fingerprint, payload, project.created_at, now),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_project(self, owner_id: str, project_id: str) -> ResearchProject:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM research_projects WHERE owner_id=? AND project_id=?",
                (owner, project_id),
            ).fetchone()
        if row is None:
            raise KeyError(project_id)
        payload = json.loads(str(row["payload"]))
        project = _project_from_payload(payload)
        if project.owner_id != owner:
            raise PermissionError("workspace project owner mismatch")
        return project

    def list_projects(self, owner_id: str, *, limit: int = 200) -> tuple[ResearchProject, ...]:
        owner = normalize_owner_id(owner_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT payload FROM research_projects WHERE owner_id=?
                   ORDER BY updated_at DESC, project_id LIMIT ?""",
                (owner, limit),
            ).fetchall()
        return tuple(_project_from_payload(json.loads(str(row["payload"]))) for row in rows)

    def put_session(self, session: ResearchSession, *, expected_fingerprint: str | None = None) -> None:
        if not isinstance(session, ResearchSession):
            raise TypeError("session must be ResearchSession")
        # Parent lookup also enforces owner-scoped project membership.
        self.get_project(session.owner_id, session.project_id)
        payload = _strict_json({
            "owner_id": session.owner_id,
            "project_id": session.project_id,
            "session_id": session.session_id,
            "turns": [vars(item) for item in session.turns],
            "created_at": session.created_at,
            "closed_at": session.closed_at,
        })
        now = time.time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT fingerprint,version FROM research_sessions WHERE owner_id=? AND session_id=?",
                    (session.owner_id, session.session_id),
                ).fetchone()
                if row is None:
                    if expected_fingerprint is not None:
                        raise RuntimeError("research session optimistic concurrency check failed")
                    connection.execute(
                        """INSERT INTO research_sessions
                           (owner_id,session_id,project_id,fingerprint,payload,created_at,updated_at,version)
                           VALUES(?,?,?,?,?,?,?,1)""",
                        (session.owner_id, session.session_id, session.project_id, session.fingerprint, payload, session.created_at, now),
                    )
                else:
                    current = str(row["fingerprint"])
                    version = int(row["version"])
                    if expected_fingerprint is not None and current != expected_fingerprint:
                        raise RuntimeError("research session optimistic concurrency check failed")
                    cursor = connection.execute(
                        """UPDATE research_sessions SET project_id=?,fingerprint=?,payload=?,updated_at=?,version=?
                           WHERE owner_id=? AND session_id=? AND version=?""",
                        (session.project_id, session.fingerprint, payload, now, version + 1, session.owner_id, session.session_id, version),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("research session concurrent update detected")
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def get_session(self, owner_id: str, session_id: str) -> ResearchSession:
        owner = normalize_owner_id(owner_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM research_sessions WHERE owner_id=? AND session_id=?",
                (owner, session_id),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        session = _session_from_payload(json.loads(str(row["payload"])))
        if session.owner_id != owner:
            raise PermissionError("workspace session owner mismatch")
        return session

    def list_sessions(self, owner_id: str, project_id: str, *, limit: int = 200) -> tuple[ResearchSession, ...]:
        owner = normalize_owner_id(owner_id)
        self.get_project(owner, project_id)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit is invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT payload FROM research_sessions WHERE owner_id=? AND project_id=?
                   ORDER BY updated_at DESC, session_id LIMIT ?""",
                (owner, project_id, limit),
            ).fetchall()
        return tuple(_session_from_payload(json.loads(str(row["payload"]))) for row in rows)


__all__ = ["SQLiteResearchWorkspaceStore"]
