"""Crash-persistent, owner-scoped ingestion job storage.

The store deliberately exposes a small state-machine API. Every public write validates
identifiers, clocks, retry values, transition legality, and public text before opening a
transaction. Every read treats durable rows as untrusted and fails closed on malformed
records rather than replaying or returning them.
"""

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
_ACTIVE_STATUSES = frozenset({"queued", "processing", "finalizing"})
_ALLOWED_TRANSITIONS = {
    "queued": frozenset({"queued", "processing", "failed"}),
    "processing": frozenset({"processing", "queued", "finalizing", "failed"}),
    "finalizing": frozenset({"finalizing", "queued", "failed", "success"}),
    "success": frozenset({"success"}),
    "failed": frozenset({"failed"}),
}
_ALLOWED_UPDATE_FIELDS = frozenset(
    {"status", "filename", "message", "doc_id", "source_path", "next_attempt_at"}
)
_MAX_RECOVERABLE_JOBS = 100_000
_MAX_JOB_ID_CHARS = 200
_MAX_DOCUMENT_ID_CHARS = 200
_MAX_FILENAME_CHARS = 500
_MAX_MESSAGE_CHARS = 2000
_MAX_OWNER_SAFE_PATH_CHARS = 4000
_MAX_ATTEMPTS = 1_000_000
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def _absolute_without_resolution(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    try:
        rendered = os.fspath(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be a filesystem path.") from exc
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_OWNER_SAFE_PATH_CHARS
        or _contains_ascii_control(rendered)
    ):
        raise ValueError(f"{label} is invalid or too long.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _safe_database_path(value: str | os.PathLike[str]) -> Path:
    absolute = _absolute_without_resolution(value, "JOB_DB_PATH")
    for component in (absolute, *absolute.parents):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("JOB_DB_PATH could not be inspected safely.") from exc
        if _is_link_or_reparse(info):
            raise ValueError(
                "JOB_DB_PATH may not contain symbolic-link or reparse-point components."
            )
    return absolute


def _bounded_identifier(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    identifier = value.strip()
    if (
        not identifier
        or len(identifier) > maximum
        or _contains_ascii_control(identifier)
    ):
        raise ValueError(f"{label} must contain 1-{maximum} valid characters.")
    return identifier


def _job_identifier(value: Any) -> str:
    return _bounded_identifier(value, "job_id", _MAX_JOB_ID_CHARS)


def _document_identifier(value: Any) -> str:
    return _bounded_identifier(value, "doc_id", _MAX_DOCUMENT_ID_CHARS)


def _strict_integer(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _finite_timestamp(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}.")
    return numeric


def _current_time() -> float:
    return _finite_timestamp(time.time(), "current time")


def _safe_public_text(value: Any, *, limit: int, default: str = "") -> str:
    """Mask and normalize one public durable string to a bounded single line."""

    if value is None:
        rendered = default
    elif isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = str(value)
        except Exception:
            rendered = default
    masked = mask_metadata_text(rendered)
    without_controls = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in masked
    )
    return without_controls.strip()[:limit]


def _validate_stored_attempts(value: Any) -> int:
    return _strict_integer(value, "stored attempts", minimum=0, maximum=_MAX_ATTEMPTS)


class JobStore:
    """Durable ingestion queue and owner-scoped public job-status registry."""

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
        self.path = _safe_database_path(self.path)
        parent_info = os.stat(self.path.parent, follow_symlinks=False)
        if not stat.S_ISDIR(parent_info.st_mode) or _is_link_or_reparse(parent_info):
            raise ValueError("JOB_DB_PATH parent must be a safe directory.")
        self._parent_identity = (int(parent_info.st_dev), int(parent_info.st_ino))

        if isinstance(ttl_seconds, bool) or not isinstance(ttl_seconds, int):
            raise ValueError("ttl_seconds must be an integer.")
        if ttl_seconds <= 0 or ttl_seconds > 31_536_000:
            raise ValueError("ttl_seconds must be between 1 and 31,536,000.")
        self.ttl_seconds = max(60, ttl_seconds)
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
        try:
            parent_info = os.stat(self.path.parent, follow_symlinks=False)
        except OSError as exc:
            raise OSError("JOB_DB_PATH parent is unavailable.") from exc
        parent_identity = (int(parent_info.st_dev), int(parent_info.st_ino))
        if (
            not stat.S_ISDIR(parent_info.st_mode)
            or _is_link_or_reparse(parent_info)
            or parent_identity != self._parent_identity
        ):
            raise OSError("JOB_DB_PATH parent identity changed after initialization.")
        if self.path.exists():
            info = os.stat(self.path, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or _is_link_or_reparse(info):
                raise OSError("JOB_DB_PATH must remain a safe regular file.")

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
        except (sqlite3.Error, OSError, TypeError, ValueError):
            return False

    def prune(self, now: Optional[float] = None) -> int:
        current = _current_time() if now is None else _finite_timestamp(now, "current time")
        cutoff = current - self.ttl_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM jobs WHERE updated_at < ? "
                "AND status IN ('success', 'failed')",
                (cutoff,),
            )
            return max(int(cursor.rowcount), 0)

    def _retry_delay(self, attempts: Any) -> float:
        if isinstance(attempts, bool):
            count = 0
        else:
            try:
                count = int(attempts)
            except (TypeError, ValueError, OverflowError):
                count = 0
        count = max(0, min(count, _MAX_ATTEMPTS))
        exponent = min(max(count - 1, 0), 60)
        return min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2**exponent),
        )

    def retry_delay_seconds(self, attempts: int) -> float:
        count = _strict_integer(
            attempts,
            "attempts",
            minimum=0,
            maximum=_MAX_ATTEMPTS,
        )
        return self._retry_delay(count)

    def retry_deadline(self, attempts: int, *, now: Optional[float] = None) -> float:
        current = _current_time() if now is None else _finite_timestamp(now, "current time")
        deadline = current + self.retry_delay_seconds(attempts)
        if not math.isfinite(deadline):
            raise ValueError("retry deadline must remain finite.")
        return deadline

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
        unknown = sorted(set(fields) - _ALLOWED_UPDATE_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported job update field: {unknown[0]}.")
        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        now = _current_time()

        raw_status = fields.get("status", "queued")
        if not isinstance(raw_status, str):
            raise ValueError("status must be a string.")
        status = raw_status.strip().lower()
        if status not in _ALLOWED_STATUSES:
            raise ValueError(
                "status must be one of queued, processing, finalizing, success, or failed."
            )

        raw_message = fields.get("message")
        message = (
            _safe_public_text(raw_message, limit=_MAX_MESSAGE_CHARS)
            if raw_message not in (None, "")
            else None
        ) or None
        raw_doc_id = fields.get("doc_id")
        requested_doc_id = (
            None
            if raw_doc_id in (None, "")
            else _document_identifier(raw_doc_id)
        )

        source_path_value = fields.get("source_path")
        explicit_source = "source_path" in fields
        source_path: Optional[str] = None
        if source_path_value not in (None, ""):
            source_path = str(
                _absolute_without_resolution(source_path_value, "source_path")
            )

        explicit_deadline = "next_attempt_at" in fields
        deadline_value = fields.get("next_attempt_at")
        if explicit_deadline:
            explicit_next_attempt_at = _finite_timestamp(
                deadline_value,
                "next_attempt_at",
            )
        else:
            explicit_next_attempt_at = 0.0

        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT owner_id, status, filename, attempts, source_path, "
                "next_attempt_at, doc_id, created_at FROM jobs WHERE job_id=?",
                (identifier,),
            ).fetchone()
            if existing is not None and str(existing["owner_id"]) != owner:
                raise PermissionError(
                    "A job ID cannot be reassigned to a different owner."
                )

            previous_status = str(existing["status"] or "") if existing else ""
            self._validate_transition(previous_status, status)
            attempts = (
                _validate_stored_attempts(existing["attempts"])
                if existing is not None
                else 0
            )
            created_at = (
                _finite_timestamp(existing["created_at"], "stored created_at")
                if existing is not None
                else now
            )

            if "filename" in fields:
                filename = _safe_public_text(
                    fields.get("filename"),
                    limit=_MAX_FILENAME_CHARS,
                    default="upload",
                ) or "upload"
            elif existing is not None:
                filename = _safe_public_text(
                    existing["filename"],
                    limit=_MAX_FILENAME_CHARS,
                    default="upload",
                ) or "upload"
            else:
                filename = "upload"

            if not explicit_source and existing is not None:
                stored_source = existing["source_path"]
                source_path = (
                    str(_absolute_without_resolution(stored_source, "stored source_path"))
                    if stored_source not in (None, "")
                    else None
                )

            if explicit_deadline:
                next_attempt_at = explicit_next_attempt_at
            elif existing is not None:
                next_attempt_at = _finite_timestamp(
                    existing["next_attempt_at"],
                    "stored next_attempt_at",
                )
            else:
                next_attempt_at = 0.0

            if (
                status == "queued"
                and not explicit_deadline
                and previous_status in {"processing", "finalizing"}
            ):
                next_attempt_at = self.retry_deadline(attempts, now=now)
            elif status != "queued":
                next_attempt_at = 0.0

            stored_doc_id = requested_doc_id if status in {"finalizing", "success"} else None
            if status in {"finalizing", "success"} and stored_doc_id is None and existing:
                prior_doc_id = existing["doc_id"]
                if prior_doc_id not in (None, ""):
                    stored_doc_id = _document_identifier(prior_doc_id)
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
                    created_at,
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
        limit = _strict_integer(
            max_attempts,
            "max_attempts",
            minimum=1,
            maximum=_MAX_ATTEMPTS,
        )
        current = _current_time() if now is None else _finite_timestamp(now, "current time")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status='processing', attempts=attempts + 1,
                    next_attempt_at=0, message=NULL, doc_id=NULL, updated_at=?
                WHERE job_id=? AND owner_id=? AND status='queued'
                  AND typeof(attempts)='integer'
                  AND attempts >= 0 AND attempts < ?
                  AND typeof(next_attempt_at) IN ('integer', 'real')
                  AND next_attempt_at >= 0 AND next_attempt_at <= ?
                """,
                (current, identifier, owner, limit, current),
            )
            return int(cursor.rowcount) == 1

    @staticmethod
    def _sanitize_internal_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate one complete durable row without silently repairing corruption."""

        try:
            job_id = _job_identifier(raw.get("job_id"))
            owner_id = normalize_owner_id(raw.get("owner_id"))
            status = raw.get("status")
            if not isinstance(status, str) or status not in _ALLOWED_STATUSES:
                return None
            filename = raw.get("filename")
            if (
                not isinstance(filename, str)
                or not filename
                or len(filename) > _MAX_FILENAME_CHARS
                or _contains_ascii_control(filename)
            ):
                return None
            message = raw.get("message")
            if message is not None and (
                not isinstance(message, str)
                or len(message) > _MAX_MESSAGE_CHARS
                or _contains_ascii_control(message)
            ):
                return None
            doc_value = raw.get("doc_id")
            doc_id = (
                None
                if doc_value in (None, "")
                else _document_identifier(doc_value)
            )
            if status in {"finalizing", "success"} and doc_id is None:
                return None
            if status not in {"finalizing", "success"} and doc_id is not None:
                return None
            source_value = raw.get("source_path")
            source_path = (
                None
                if source_value in (None, "")
                else str(_absolute_without_resolution(source_value, "source_path"))
            )
            attempts = _strict_integer(
                raw.get("attempts"),
                "attempts",
                minimum=0,
                maximum=_MAX_ATTEMPTS,
            )
            next_attempt_at = _finite_timestamp(
                raw.get("next_attempt_at"),
                "next_attempt_at",
            )
            created_at = _finite_timestamp(raw.get("created_at"), "created_at")
            updated_at = _finite_timestamp(raw.get("updated_at"), "updated_at")
        except (TypeError, ValueError):
            return None
        return {
            "job_id": job_id,
            "owner_id": owner_id,
            "status": status,
            "filename": filename,
            "message": message,
            "doc_id": doc_id,
            "source_path": source_path,
            "attempts": attempts,
            "next_attempt_at": next_attempt_at,
            "created_at": created_at,
            "updated_at": updated_at,
        }

    def get(self, job_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
        owner = normalize_owner_id(owner_id)
        identifier = _job_identifier(job_id)
        self.prune()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, message, doc_id,
                       source_path, attempts, next_attempt_at, created_at, updated_at
                FROM jobs WHERE job_id=? AND owner_id=?
                """,
                (identifier, owner),
            ).fetchone()
        record = self._sanitize_internal_record(dict(row)) if row is not None else None
        if record is None:
            return None
        return {
            "job_id": record["job_id"],
            "status": record["status"],
            "filename": record["filename"],
            "message": record["message"],
            "doc_id": record["doc_id"],
        }

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
        return self._sanitize_internal_record(dict(row)) if row is not None else None

    @staticmethod
    def _sanitize_recovery_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        record = JobStore._sanitize_internal_record(raw)
        if record is None or record["status"] not in _ACTIVE_STATUSES:
            return None
        return {
            "job_id": record["job_id"],
            "owner_id": record["owner_id"],
            "status": record["status"],
            "filename": record["filename"],
            "doc_id": record["doc_id"],
            "source_path": record["source_path"],
            "attempts": record["attempts"],
            "next_attempt_at": record["next_attempt_at"],
        }

    def recoverable(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT job_id, owner_id, status, filename, message, doc_id,
                       source_path, attempts, next_attempt_at, created_at, updated_at
                FROM jobs
                WHERE status IN ('queued', 'processing', 'finalizing')
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (_MAX_RECOVERABLE_JOBS,),
            ).fetchall()
        records: List[Dict[str, Any]] = []
        for row in rows:
            record = self._sanitize_recovery_record(dict(row))
            if record is not None:
                records.append(record)
        return records

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
