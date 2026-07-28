"""Crash-persistent, owner-scoped ingestion job storage."""

from __future__ import annotations

import math
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools.config import bounded_float_env
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
_MAX_RECOVERABLE_JOBS = 100_000
_MAX_JOB_ID_CHARS = 200
_MAX_OWNER_SAFE_PATH_CHARS = 4000


def _absolute_without_resolution(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if not rendered or len(rendered) > 4096 or "\x00" in rendered:
        raise ValueError(f"{label} is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _safe_database_path(value: str | os.PathLike[str]) -> Path:
    absolute = _absolute_without_resolution(value, "JOB_DB_PATH")
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(
                "JOB_DB_PATH may not contain symbolic-link components."
            )
    return absolute


def _job_identifier(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("job_id must be a string.")
    identifier = value.strip()
    if (
        not identifier
        or len(identifier) > _MAX_JOB_ID_CHARS
        or "\x00" in identifier
    ):
        raise ValueError("job_id must contain 1-200 valid characters.")
    return identifier


def _bounded_integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _saturating_nonnegative_integer(value: Any, maximum: int) -> int:
    if isinstance(value, bool):
        return 0
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(numeric, maximum))


def _finite_timestamp(value: Any, label: str, *, minimum: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}.")
    return numeric


def _safe_public_text(value: Any, *, limit: int, default: str = "") -> str:
    if value is None:
        rendered = default
    elif not isinstance(value, str):
        try:
            rendered = str(value)
        except Exception:
            rendered = default
    else:
        rendered = value
    return mask_metadata_text(rendered).strip()[:limit]


class JobStore:
    """Durable ingestion queue and public job-status registry."""

    def __init__(
        self,
        path: str | Path | None = None,
        ttl_seconds: int = 86_400,
    ) -> None:
        selected = path if path is not None else os.getenv(
            "JOB_DB_PATH", "data/jobs.sqlite3"
        )
        self.path = _safe_database_path(selected)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(ttl_seconds, bool):
            raise ValueError("ttl_seconds must be an integer.")
        try:
            raw_ttl = int(ttl_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("ttl_seconds must be an integer.") from exc
        self.ttl_seconds = max(60, min(raw_ttl, 31_536_000))
        self.retry_base_seconds = bounded_float_env(
            "INGEST_RETRY_BASE_SECONDS",
            2.0,
            minimum=0.1,
            maximum=86_400.0,
        )
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            bounded_float_env(
                "INGEST_RETRY_MAX_SECONDS",
                30.0,
                minimum=0.1,
                maximum=604_800.0,
            ),
        )
        self._lock = threading.RLock()
        self._initialise()

    def _ensure_database_path(self) -> None:
        _safe_database_path(self.path)
        if not self.path.parent.exists() or not self.path.parent.is_dir():
            raise OSError("JOB_DB_PATH parent must remain a directory.")
        if self.path.exists():
            mode = self.path.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise OSError("JOB_DB_PATH must remain a regular file.")

    def _connect(self) -> sqlite3.Connection:
        self._ensure_database_path()
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            self._ensure_database_path()
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            return connection
        except Exception:
            connection.close()
            raise

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
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and int(row["ok"]) == 1)
        except (sqlite3.Error, OSError, ValueError):
            return False

    def prune(self, now: Optional[float] = None) -> int:
        current_time = time.time() if now is None else _finite_timestamp(now, "now")
        cutoff = current_time - self.ttl_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE updated_at < ? "
                "AND status IN ('success', 'failed')",
                (cutoff,),
            )
            return max(cursor.rowcount, 0)

    def _retry_delay(self, attempts: int) -> float:
        count = _saturating_nonnegative_integer(attempts, 1_000_000)
        exponent = min(max(count - 1, 0), 60)
        return min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2**exponent),
        )

    @staticmethod
    def _validate_transition(previous_status: str, status: str) -> None:
        if status not in _ALLOWED_STATUSES:
            raise ValueError(
                "status must be one of queued, processing, finalizing, success, or failed."
            )
        if previous_status and status not in _ALLOWED_TRANSITIONS.get(
            previous_status, frozenset()
        ):
            raise ValueError(
                f"Invalid ingestion job transition from {previous_status} to {status}."
            )

    def update(self, job_id: str, owner_id: str, **fields: Any) -> None:
        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        now = time.time()
        raw_status = fields.get("status", "queued")
        if not isinstance(raw_status, str):
            raise ValueError("status must be a string.")
        status = raw_status.strip().lower()
        filename = _safe_public_text(
            fields.get("filename"), limit=500, default="upload"
        ) or "upload"
        raw_message = fields.get("message")
        message = (
            _safe_public_text(raw_message, limit=2000)
            if raw_message not in (None, "")
            else None
        )
        message = message or None
        raw_doc_id = fields.get("doc_id")
        if raw_doc_id in (None, ""):
            requested_doc_id = None
        else:
            if not isinstance(raw_doc_id, str):
                raise ValueError("doc_id must be a string.")
            requested_doc_id = raw_doc_id.strip()
            if (
                not requested_doc_id
                or len(requested_doc_id) > 200
                or "\x00" in requested_doc_id
            ):
                raise ValueError("doc_id must contain 1-200 valid characters.")

        source_path_value = fields.get("source_path")
        source_path = None
        if source_path_value not in (None, ""):
            source_path = str(
                _absolute_without_resolution(source_path_value, "source_path")
            )
            if len(source_path) > _MAX_OWNER_SAFE_PATH_CHARS:
                raise ValueError("source_path exceeds the 4,000-character limit.")
        next_attempt_value = fields.get("next_attempt_at")

        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT owner_id, status, attempts, source_path, "
                "next_attempt_at, doc_id FROM jobs WHERE job_id=?",
                (identifier,),
            ).fetchone()
            if existing is not None and str(existing["owner_id"]) != owner:
                raise PermissionError(
                    "A job ID cannot be reassigned to a different owner."
                )
            attempts = (
                _saturating_nonnegative_integer(existing["attempts"] or 0, 1_000_000)
                if existing
                else 0
            )
            previous_status = str(existing["status"] or "") if existing else ""
            self._validate_transition(previous_status, status)
            if source_path_value is None and existing is not None:
                source_path = str(existing["source_path"] or "") or None
            if next_attempt_value is None and existing is not None:
                next_attempt_at = _finite_timestamp(
                    existing["next_attempt_at"] or 0.0,
                    "stored next_attempt_at",
                )
            else:
                next_attempt_at = _finite_timestamp(
                    next_attempt_value or 0.0,
                    "next_attempt_at",
                )
            if (
                status == "queued"
                and next_attempt_value is None
                and previous_status in {"processing", "finalizing"}
            ):
                next_attempt_at = now + self._retry_delay(attempts)
            elif status != "queued":
                next_attempt_at = 0.0

            stored_doc_id = requested_doc_id if status in {"finalizing", "success"} else None
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
        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        limit = _bounded_integer(
            max_attempts,
            "max_attempts",
            minimum=1,
            maximum=1_000_000,
        )
        current_time = time.time() if now is None else _finite_timestamp(now, "now")
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
        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        self.prune()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT job_id, status, filename, message, doc_id "
                "FROM jobs WHERE job_id=? AND owner_id=?",
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
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, doc_id,
                       source_path, attempts, next_attempt_at
                FROM jobs
                WHERE status IN ('queued', 'processing', 'finalizing')
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (_MAX_RECOVERABLE_JOBS,),
            ).fetchall()
        return [dict(row) for row in rows]

    def active_source_paths(self) -> Set[Path]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT source_path FROM jobs
                WHERE status IN ('queued', 'processing', 'finalizing')
                  AND source_path IS NOT NULL
                LIMIT ?
                """,
                (_MAX_RECOVERABLE_JOBS,),
            ).fetchall()
        paths: Set[Path] = set()
        for row in rows:
            raw_path = row["source_path"]
            if not isinstance(raw_path, str) or not raw_path:
                continue
            try:
                paths.add(_absolute_without_resolution(raw_path, "source_path"))
            except ValueError:
                continue
        return paths
