"""Private owner-scoped registry for retained source documents.

Source filesystem paths never belong in the vector database or API responses. This
registry is the single authority for resolving an uploaded document to a retained
source file used by visual tools.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools.privacy import mask_metadata_text
from tools.security import DEFAULT_MAX_UPLOAD_BYTES, normalize_owner_id


class DocumentStore:
    """SQLite-backed document/source registry with strict owner isolation."""

    def __init__(
        self,
        path: str | Path | None = None,
        upload_root: str | Path | None = None,
    ) -> None:
        self.path = Path(
            path or os.getenv("DOCUMENT_DB_PATH", "data/documents.sqlite3")
        ).resolve()
        self.upload_root = Path(
            upload_root or os.getenv("UPLOAD_DIR", "uploads")
        ).resolve()
        self.orphan_grace_seconds = max(
            int(os.getenv("ORPHAN_GRACE_SECONDS", "3600")),
            60,
        )
        self.last_cleanup_deleted = 0
        self.last_cleanup_errors: List[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()
        if os.getenv("ORPHAN_CLEANUP_ON_STARTUP", "true").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.last_cleanup_deleted = self.cleanup_orphans()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialise(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    owner_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    source_path TEXT,
                    source_retained INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, doc_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_owner_updated "
                "ON documents(owner_id, updated_at)"
            )

    def ping(self) -> bool:
        """Return whether the registry database can complete a trivial read."""

        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("SELECT 1 AS ok").fetchone()
            return bool(row and int(row["ok"]) == 1)
        except sqlite3.Error:
            return False

    def _resolve_source_path(self, source_path: str | Path | None) -> Optional[Path]:
        """Resolve one existing regular source without following a symbolic link."""

        if source_path in (None, ""):
            return None
        unresolved = Path(source_path)
        if unresolved.is_symlink():
            return None
        candidate = unresolved.resolve()
        try:
            candidate.relative_to(self.upload_root)
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

    def _validated_source_path(self, source_path: str | Path | None) -> Optional[str]:
        if source_path in (None, ""):
            return None
        raw_path = Path(source_path)
        if raw_path.is_symlink():
            raise ValueError("Retained source files may not be symbolic links.")
        candidate = raw_path.resolve()
        try:
            candidate.relative_to(self.upload_root)
        except ValueError as exc:
            raise ValueError("Retained source path must be inside UPLOAD_DIR.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise ValueError("Retained source file does not exist.")
        return str(candidate)

    def copy_source(
        self,
        *,
        owner_id: str,
        source_path: str | Path,
        max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> Path:
        """Copy a regular external source into a bounded random owner-scoped path."""

        owner = normalize_owner_id(owner_id)
        limit = int(max_bytes)
        if limit <= 0:
            raise ValueError("max_bytes must be positive.")
        raw_source = Path(source_path)
        if raw_source.is_symlink():
            raise ValueError("Source files may not be symbolic links.")
        source = raw_source.resolve()
        if not source.exists() or not source.is_file():
            raise ValueError("Source file does not exist.")
        if source.stat().st_size > limit:
            raise ValueError(f"Source file exceeds the {limit}-byte retention limit.")
        suffix = source.suffix.lower()
        if suffix not in {".pdf", ".docx", ".txt", ".md"}:
            raise ValueError("Unsupported source-file suffix.")
        owner_dir = self.upload_root / owner
        owner_dir.mkdir(parents=True, exist_ok=True)
        destination = owner_dir / f"{uuid.uuid4().hex}{suffix}"
        total = 0
        try:
            with source.open("rb") as input_handle, destination.open("xb") as output_handle:
                while True:
                    chunk = input_handle.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise ValueError(
                            f"Source file exceeds the {limit}-byte retention limit."
                        )
                    output_handle.write(chunk)
                output_handle.flush()
                os.fsync(output_handle.fileno())
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination.resolve()

    def remove_source(self, source_path: str | Path | None) -> bool:
        """Remove one retained regular file without following a symlink."""

        candidate = self._resolve_source_path(source_path)
        if candidate is None:
            return False
        candidate.unlink(missing_ok=True)
        return True

    def retained_source_paths(self) -> Set[Path]:
        """Return valid retained paths used to protect files during orphan sweeping."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT source_path FROM documents "
                "WHERE source_retained=1 AND source_path IS NOT NULL"
            ).fetchall()
        paths: Set[Path] = set()
        for row in rows:
            candidate = self._resolve_source_path(str(row["source_path"] or ""))
            if candidate is not None:
                paths.add(candidate)
        return paths

    def cleanup_orphans(
        self,
        *,
        now: Optional[float] = None,
        job_store: Optional[Any] = None,
    ) -> int:
        """Delete only old regular files unreferenced by documents or active jobs.

        If either reference store cannot be read, cleanup fails closed and deletes
        nothing. Recent files are protected by ``ORPHAN_GRACE_SECONDS`` to cover the
        interval between writing an upload and committing its job record.
        """

        self.last_cleanup_errors = []
        try:
            retained = self.retained_source_paths()
            if job_store is None:
                from tools.job_store import JobStore

                job_store = JobStore()
            active = set(job_store.active_source_paths())
        except Exception as exc:
            self.last_cleanup_errors.append(
                f"reference_lookup_failed:{type(exc).__name__}"
            )
            return 0

        referenced: Set[Path] = set()
        for raw_path in retained | active:
            unresolved = Path(raw_path)
            if unresolved.is_symlink():
                continue
            candidate = unresolved.resolve()
            try:
                candidate.relative_to(self.upload_root)
            except ValueError:
                continue
            referenced.add(candidate)

        current_time = time.time() if now is None else float(now)
        cutoff = current_time - self.orphan_grace_seconds
        deleted = 0
        try:
            candidates = list(self.upload_root.rglob("*"))
        except OSError as exc:
            self.last_cleanup_errors.append(
                f"upload_scan_failed:{type(exc).__name__}"
            )
            return 0

        for raw_path in candidates:
            try:
                if raw_path.is_symlink() or not raw_path.is_file():
                    continue
                candidate = raw_path.resolve()
                candidate.relative_to(self.upload_root)
                if candidate in referenced:
                    continue
                if raw_path.stat().st_mtime >= cutoff:
                    continue
                raw_path.unlink()
                deleted += 1
            except (OSError, ValueError) as exc:
                self.last_cleanup_errors.append(
                    f"orphan_delete_failed:{type(exc).__name__}"
                )
        self.last_cleanup_deleted = deleted
        return deleted

    def register(
        self,
        *,
        owner_id: str,
        doc_id: str,
        filename: str,
        mime_type: str,
        source_path: str | Path | None = None,
    ) -> Optional[str]:
        """Upsert a record and return the previous retained path, if it changed."""

        owner = normalize_owner_id(owner_id)
        document_id = (doc_id or "").strip()
        if not document_id or len(document_id) > 200:
            raise ValueError("doc_id must contain 1-200 characters.")
        safe_filename = mask_metadata_text(Path(filename or "document").name)[:500]
        safe_mime = str(mime_type or "application/octet-stream")[:200]
        validated_path = self._validated_source_path(source_path)
        now = time.time()
        with self._lock, self._connect() as connection:
            existing = connection.execute(
                "SELECT source_path FROM documents WHERE owner_id=? AND doc_id=?",
                (owner, document_id),
            ).fetchone()
            connection.execute(
                """
                INSERT INTO documents(
                    owner_id, doc_id, filename, mime_type, source_path,
                    source_retained, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id, doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    mime_type=excluded.mime_type,
                    source_path=excluded.source_path,
                    source_retained=excluded.source_retained,
                    updated_at=excluded.updated_at
                """,
                (
                    owner,
                    document_id,
                    safe_filename,
                    safe_mime,
                    validated_path,
                    1 if validated_path else 0,
                    now,
                    now,
                ),
            )
        previous = str(existing["source_path"]) if existing and existing["source_path"] else None
        return previous if previous and previous != validated_path else None

    def get(self, *, owner_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return a record with source capability validated against the filesystem."""

        owner = normalize_owner_id(owner_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT owner_id, doc_id, filename, mime_type, source_path,
                       source_retained, created_at, updated_at
                FROM documents WHERE owner_id=? AND doc_id=?
                """,
                (owner, doc_id),
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        source = self._resolve_source_path(record.get("source_path"))
        record["source_path"] = str(source) if source is not None else None
        record["source_retained"] = 1 if source is not None else 0
        return record

    def source_path(self, *, owner_id: str, doc_id: str) -> Optional[Path]:
        record = self.get(owner_id=owner_id, doc_id=doc_id)
        raw_path = str((record or {}).get("source_path") or "")
        return Path(raw_path) if raw_path else None

    def delete(self, *, owner_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        owner = normalize_owner_id(owner_id)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT owner_id, doc_id, filename, mime_type, source_path,
                       source_retained, created_at, updated_at
                FROM documents WHERE owner_id=? AND doc_id=?
                """,
                (owner, doc_id),
            ).fetchone()
            connection.execute(
                "DELETE FROM documents WHERE owner_id=? AND doc_id=?",
                (owner, doc_id),
            )
        return dict(row) if row is not None else None


_DOCUMENT_STORES: Dict[tuple[str, str], DocumentStore] = {}
_DOCUMENT_STORE_LOCK = threading.Lock()


def get_document_store(
    path: str | Path | None = None,
    upload_root: str | Path | None = None,
) -> DocumentStore:
    resolved_path = str(
        Path(path or os.getenv("DOCUMENT_DB_PATH", "data/documents.sqlite3")).resolve()
    )
    resolved_root = str(
        Path(upload_root or os.getenv("UPLOAD_DIR", "uploads")).resolve()
    )
    key = (resolved_path, resolved_root)
    with _DOCUMENT_STORE_LOCK:
        store = _DOCUMENT_STORES.get(key)
        if store is None:
            store = DocumentStore(resolved_path, resolved_root)
            _DOCUMENT_STORES[key] = store
        return store
