"""Crash-persistent, owner-scoped ingestion job storage."""

from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools.privacy import mask_metadata_text
from tools.security import normalize_owner_id

_ALLOWED_STATUSES = frozenset({"queued", "processing", "finalizing", "success", "failed"})
_ALLOWED_TRANSITIONS = {
    "queued": frozenset({"queued", "failed"}),
    "processing": frozenset({"processing", "queued", "finalizing", "failed"}),
    "finalizing": frozenset({"finalizing", "queued", "failed", "success"}),
    "success": frozenset({"success"}),
    "failed": frozenset({"failed"}),
}


def _finite_positive_env(name: str, default: str, *, minimum: float) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(value, minimum)


def _lexical_absolute(path: str | Path) -> Path:
    """Make a path absolute without following a symlink or resolving its target."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _job_identifier(value: str) -> str:
    identifier = str(value or "").strip()
    if not identifier or len(identifier) > 200:
        raise ValueError("job_id must contain 1-200 characters.")
    return identifier


class JobStore:
    """Durable ingestion queue and public job-status registry."""

    def __init__(self, path: str | Path | None = None, ttl_seconds: int = 86_400) -> None:
        raw_path = Path(path or os.getenv("JOB_DB_PATH", "data/jobs.sqlite3"))
        if raw_path.is_symlink():
            raise ValueError("JOB_DB_PATH may not be a symbolic link.")
        self.path = _lexical_absolute(raw_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("JOB_DB_PATH may not be a symbolic link.")
        self.ttl_seconds = max(int(ttl_seconds), 60)
        self.retry_base_seconds = _finite_positive_env(
            "INGEST_RETRY_BASE_SECONDS",
            "2",
            minimum=0.1,
        )
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            _finite_positive_env(
                "INGEST_RETRY_MAX_SECONDS",
                "30",
                minimum=0.1,
            ),
        )
        self._lock = threading.RLock()
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        if self.path.is_symlink():
            raise ValueError("JOB_DB_PATH became a symbolic link.")
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
                    next_attempt_at REAL NOT NULL DEFAULT 0,
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
            if "next_attempt_at" not in existing_columns:
                connection.execute(
                    "ALTER TABLE jobs ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_owner_updated "
                "ON jobs(owner_id, updated_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_recovery "
                "ON jobs(status, next_attempt_at, updated_at)"
            )

    def ping(self) -> bool:
        """Return whether the queue database can complete a trivial read."""

        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and int(row["ok"]) == 1)
        except (sqlite3.Error, OSError, ValueError):
            return False

    def prune(self, now: Optional[float] = None) -> int:
        current_time = time.time() if now is None else float(now)
        if not math.isfinite(current_time):
            raise ValueError("now must be a finite timestamp.")
        cutoff = current_time - self.ttl_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE updated_at < ? AND status IN ('success', 'failed')",
                (cutoff,),
            )
            return max(cursor.rowcount, 0)

    def _retry_delay(self, attempts: int) -> float:
        exponent = min(max(int(attempts) - 1, 0), 60)
        return min(self.retry_max_seconds, self.retry_base_seconds * (2**exponent))

    @staticmethod
    def _validate_transition(previous_status: str, status: str) -> None:
        if status not in _ALLOWED_STATUSES:
            raise ValueError(
                "status must be one of queued, processing, finalizing, success, or failed."
            )
        if previous_status and status not in _ALLOWED_TRANSITIONS.get(previous_status, frozenset()):
            raise ValueError(
                f"Invalid ingestion job transition from {previous_status} to {status}."
            )

    def update(self, job_id: str, owner_id: str, **fields: Any) -> None:
        """Create or update a job without owner reassignment or inferred completion."""

        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        now = time.time()
        status = str(fields.get("status") or "queued").strip().lower()
        filename = mask_metadata_text(str(fields.get("filename") or "upload")).strip()[:500]
        filename = filename or "upload"
        raw_message = fields.get("message")
        message = (
            mask_metadata_text(str(raw_message)).strip()[:2000]
            if raw_message not in (None, "")
            else None
        )
        message = message or None
        raw_doc_id = str(fields.get("doc_id") or "").strip()
        if len(raw_doc_id) > 200:
            raise ValueError("doc_id must contain at most 200 characters.")
        requested_doc_id = raw_doc_id or None
        source_path_value = fields.get("source_path")
        source_path = None
        if source_path_value not in (None, ""):
            source_path = str(_lexical_absolute(str(source_path_value)))
            if len(source_path) > 4000:
                raise ValueError("source_path exceeds the 4,000-character limit.")
        next_attempt_value = fields.get("next_attempt_at")
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT owner_id, status, attempts, source_path, next_attempt_at, doc_id "
                "FROM jobs WHERE job_id=?",
                (identifier,),
            ).fetchone()
            if existing is not None and str(existing["owner_id"]) != owner:
                raise PermissionError("A job ID cannot be reassigned to a different owner.")
            attempts = max(int(existing["attempts"] or 0), 0) if existing else 0
            previous_status = str(existing["status"] or "") if existing else ""
            self._validate_transition(previous_status, status)
            if source_path_value is None and existing is not None:
                source_path = str(existing["source_path"] or "") or None
            if next_attempt_value is None and existing is not None:
                next_attempt_at = float(existing["next_attempt_at"] or 0.0)
            else:
                next_attempt_at = max(0.0, float(next_attempt_value or 0.0))
            if not math.isfinite(next_attempt_at):
                raise ValueError("next_attempt_at must be a finite timestamp.")
            if (
                status == "queued"
                and next_attempt_value is None
                and previous_status in {"processing", "finalizing"}
            ):
                next_attempt_at = now + self._retry_delay(attempts)
            elif status != "queued":
                next_attempt_at = 0.0
            stored_doc_id = (
                requested_doc_id
                if status in {"finalizing", "success"}
                else None
            )
            if status in {"finalizing", "success"} and stored_doc_id is None and existing:
                existing_doc_id = str(existing["doc_id"] or "").strip()
                stored_doc_id = existing_doc_id or None
            if status in {"finalizing", "success"} and stored_doc_id is None:
                raise ValueError(f"doc_id is required when status is {status}.")
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, owner_id, status, filename, message, doc_id,
                    source_path, attempts, next_attempt_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    filename=excluded.filename,
                    message=excluded.message,
                    doc_id=excluded.doc_id,
                    source_path=excluded.source_path,
                    attempts=excluded.attempts,
                    next_attempt_at=excluded.next_attempt_at,
                    updated_at=excluded.updated_at
                """,
                (
                    identifier,
                    owner,
                    status,
                    filename,
                    message,
                    stored_doc_id,
                    source_path,
                    attempts,
                    next_attempt_at,
                    now,
                    now,
                ),
            )
        self.prune(now)

    def claim(
        self,
        job_id: str,
        owner_id: str,
        max_attempts: int,
        *,
        now: Optional[float] = None,
    ) -> bool:
        """Atomically claim one due queued job without occupying a worker while delayed."""

        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        limit = max(1, min(int(max_attempts), 1_000_000))
        current_time = time.time() if now is None else float(now)
        if not math.isfinite(current_time):
            raise ValueError("now must be a finite timestamp.")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status='processing', attempts=attempts + 1,
                    next_attempt_at=0, message=NULL, doc_id=NULL, updated_at=?
                WHERE job_id=? AND owner_id=? AND status='queued'
                  AND attempts >= 0 AND attempts < ? AND next_attempt_at <= ?
                """,
                (current_time, identifier, owner, limit, current_time),
            )
            return cursor.rowcount == 1

    def get(self, job_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        """Return only fields safe for the owner-facing API."""

        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        self.prune()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, status, filename, message, doc_id
                FROM jobs WHERE job_id=? AND owner_id=?
                """,
                (identifier, owner),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_internal(self, job_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, message, doc_id,
                       source_path, attempts, next_attempt_at, created_at, updated_at
                FROM jobs WHERE job_id=? AND owner_id=?
                """,
                (identifier, owner),
            ).fetchone()
        return dict(row) if row is not None else None

    def recoverable(self) -> List[Dict[str, Any]]:
        """Return queued, interrupted, and finalizing jobs for reconciliation."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, doc_id,
                       source_path, attempts, next_attempt_at
                FROM jobs
                WHERE status IN ('queued', 'processing', 'finalizing')
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def active_source_paths(self) -> Set[Path]:
        """Return lexical source paths referenced by unfinished jobs."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_path FROM jobs
                WHERE status IN ('queued', 'processing', 'finalizing')
                  AND source_path IS NOT NULL
                """
            ).fetchall()
        paths: Set[Path] = set()
        for row in rows:
            raw_path = str(row["source_path"] or "")
            if raw_path:
                paths.add(_lexical_absolute(raw_path))
        return paths
