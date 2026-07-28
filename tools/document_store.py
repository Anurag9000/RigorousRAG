"""Private owner-scoped registry for retained source documents.

Source filesystem paths never belong in the vector database or API responses.  This
registry is the single authority for resolving an uploaded document to a retained
source file used by visual tools.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tools.privacy import mask_metadata_text
from tools.security import normalize_owner_id


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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialise()

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

    def _validated_source_path(self, source_path: str | Path | None) -> Optional[str]:
        if source_path in (None, ""):
            return None
        candidate = Path(source_path).resolve()
        try:
            candidate.relative_to(self.upload_root)
        except ValueError as exc:
            raise ValueError("Retained source path must be inside UPLOAD_DIR.") from exc
        if not candidate.exists() or not candidate.is_file():
            raise ValueError("Retained source file does not exist.")
        return str(candidate)

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
        return dict(row) if row is not None else None

    def source_path(self, *, owner_id: str, doc_id: str) -> Optional[Path]:
        record = self.get(owner_id=owner_id, doc_id=doc_id)
        raw_path = str((record or {}).get("source_path") or "")
        if not raw_path:
            return None
        candidate = Path(raw_path).resolve()
        try:
            candidate.relative_to(self.upload_root)
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_file():
            return None
        return candidate

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
