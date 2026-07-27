"""Crash-persistent, owner-scoped ingestion job storage."""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tools.privacy import mask_metadata_text


class JobStore:
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
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_owner_updated ON jobs(owner_id, updated_at)"
            )

    def prune(self, now: Optional[float] = None) -> int:
        cutoff = (now or time.time()) - self.ttl_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff,))
            return max(cursor.rowcount, 0)

    def update(self, job_id: str, owner_id: str, **fields: Any) -> None:
        now = time.time()
        status = str(fields.get("status") or "processing")[:64]
        filename = mask_metadata_text(str(fields.get("filename") or "upload"))[:500]
        raw_message = fields.get("message")
        message = (
            mask_metadata_text(str(raw_message))[:2000]
            if raw_message not in (None, "")
            else None
        )
        doc_id = str(fields.get("doc_id"))[:200] if fields.get("doc_id") else None
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(job_id, owner_id, status, filename, message, doc_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    filename=excluded.filename,
                    message=excluded.message,
                    doc_id=excluded.doc_id,
                    updated_at=excluded.updated_at
                WHERE jobs.owner_id=excluded.owner_id
                """,
                (job_id, owner_id, status, filename, message, doc_id, now, now),
            )
        self.prune(now)

    def get(self, job_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        self.prune()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, status, filename, message, doc_id
                FROM jobs WHERE job_id=? AND owner_id=?
                """,
                (job_id, owner_id),
            ).fetchone()
        return dict(row) if row is not None else None
