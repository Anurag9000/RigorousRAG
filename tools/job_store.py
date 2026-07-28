"""Crash-persistent, owner-scoped ingestion job storage."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.privacy import mask_metadata_text
from tools.security import normalize_owner_id


class JobStore:
    """Durable ingestion queue and public job-status registry."""

    def __init__(self, path: str | Path | None = None, ttl_seconds: int = 86_400) -> None:
        self.path = Path(path or os.getenv("JOB_DB_PATH", "data/jobs.sqlite3")).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(int(ttl_seconds), 60)
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialise(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    message TEXT,
                    doc_id TEXT,
                    source_path TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            existing_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "source_path" not in existing_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN source_path TEXT")
            if "attempts" not in existing_columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_owner_updated "
                "ON jobs(owner_id, updated_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_recovery "
                "ON jobs(status, updated_at)"
            )

    def prune(self, now: Optional[float] = None) -> int:
        cutoff = (now or time.time()) - self.ttl_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE updated_at < ? AND status IN ('success', 'failed')",
                (cutoff,),
            )
            return max(cursor.rowcount, 0)

    def update(self, job_id: str, owner_id: str, **fields: Any) -> None:
        """Create or update a job without allowing cross-owner ID reuse."""

        owner = normalize_owner_id(owner_id)
        identifier = (job_id or "").strip()
        if not identifier or len(identifier) > 200:
            raise ValueError("job_id must contain 1-200 characters.")
        now = time.time()
        status = str(fields.get("status") or "queued")[:64]
        filename = mask_metadata_text(str(fields.get("filename") or "upload"))[:500]
        raw_message = fields.get("message")
        message = (
            mask_metadata_text(str(raw_message))[:2000]
            if raw_message not in (None, "")
            else None
        )
        doc_id = str(fields.get("doc_id"))[:200] if fields.get("doc_id") else None
        source_path_value = fields.get("source_path")
        source_path = (
            str(Path(source_path_value).resolve())[:4000]
            if source_path_value not in (None, "")
            else None
        )
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT owner_id, attempts, source_path FROM jobs WHERE job_id=?",
                (identifier,),
            ).fetchone()
            if existing is not None and str(existing["owner_id"]) != owner:
                raise PermissionError("A job ID cannot be reassigned to a different owner.")
            attempts = int(existing["attempts"] or 0) if existing else 0
            if source_path_value is None and existing is not None:
                source_path = str(existing["source_path"] or "") or None
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, owner_id, status, filename, message, doc_id,
                    source_path, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    filename=excluded.filename,
                    message=excluded.message,
                    doc_id=COALESCE(excluded.doc_id, jobs.doc_id),
                    source_path=excluded.source_path,
                    attempts=excluded.attempts,
                    updated_at=excluded.updated_at
                """,
                (
                    identifier,
                    owner,
                    status,
                    filename,
                    message,
                    doc_id,
                    source_path,
                    attempts,
                    now,
                    now,
                ),
            )
        self.prune(now)

    def claim(self, job_id: str, owner_id: str, max_attempts: int) -> bool:
        """Atomically claim one queued job; safe across threads and processes."""

        owner = normalize_owner_id(owner_id)
        limit = max(1, int(max_attempts))
        now = time.time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status='processing', attempts=attempts + 1,
                    message=NULL, updated_at=?
                WHERE job_id=? AND owner_id=? AND status='queued' AND attempts < ?
                """,
                (now, job_id, owner, limit),
            )
            return cursor.rowcount == 1

    def get(self, job_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        """Return only fields safe for the owner-facing API."""

        owner = normalize_owner_id(owner_id)
        self.prune()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, status, filename, message, doc_id
                FROM jobs WHERE job_id=? AND owner_id=?
                """,
                (job_id, owner),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_internal(self, job_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        owner = normalize_owner_id(owner_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, message, doc_id,
                       source_path, attempts, created_at, updated_at
                FROM jobs WHERE job_id=? AND owner_id=?
                """,
                (job_id, owner),
            ).fetchone()
        return dict(row) if row is not None else None

    def recoverable(self) -> List[Dict[str, Any]]:
        """Return every queued/interrupted job for startup reconciliation."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, source_path, attempts
                FROM jobs
                WHERE status IN ('queued', 'processing')
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]
