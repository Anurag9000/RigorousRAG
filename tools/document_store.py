"""Private owner-scoped registry for retained source documents.

Source filesystem paths never belong in the vector database or API responses. This
registry is the single authority for resolving an uploaded document to a retained
source file used by visual tools.
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tools.privacy import mask_metadata_text
from tools.security import DEFAULT_MAX_UPLOAD_BYTES, normalize_owner_id
from tools.upload_storage import copy_path_to_owner, remove_owner_file


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
        self.visual_max_pdf_pages = max(
            1,
            min(int(os.getenv("VISUAL_MAX_PDF_PAGES", "500")), 5000),
        )
        self.visual_max_render_pixels = max(
            1_000_000,
            min(int(os.getenv("VISUAL_MAX_RENDER_PIXELS", "2000000")), 100_000_000),
        )
        # The renderer captures at most 520 points above and 45 below a caption.
        # Keep this as a truthful capability value rather than a configurable value
        # that could underestimate the pixmap allocated by tools.integrity.
        self.visual_clip_height_points = 565.0
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

    def _source_matches_document(
        self,
        candidate: Path,
        owner_id: str,
        doc_id: str,
    ) -> bool:
        """Verify retained bytes still derive the immutable owner/content document ID."""

        try:
            if candidate.stat().st_size > DEFAULT_MAX_UPLOAD_BYTES:
                return False
            digest = hashlib.sha256()
            total = 0
            with candidate.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > DEFAULT_MAX_UPLOAD_BYTES:
                        return False
                    digest.update(chunk)
        except OSError:
            return False
        owner = normalize_owner_id(owner_id)
        expected = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rigorousrag:{owner}:{digest.hexdigest()}",
            )
        )
        return expected == str(doc_id or "")

    def _visual_pdf_is_safe(self, candidate: Path) -> bool:
        """Fail closed before rendering PDFs that exceed visual complexity limits."""

        if candidate.suffix.lower() != ".pdf":
            return False
        try:
            import fitz
        except ImportError:
            return False
        try:
            document = fitz.open(candidate)
        except Exception:
            return False
        try:
            page_count = int(document.page_count)
            if document.needs_pass or not 1 <= page_count <= self.visual_max_pdf_pages:
                return False
            for page_index in range(page_count):
                try:
                    rect = document.load_page(page_index).rect
                    width = float(rect.width)
                    height = float(rect.height)
                except Exception:
                    return False
                if (
                    not math.isfinite(width)
                    or not math.isfinite(height)
                    or width <= 0
                    or height <= 0
                ):
                    return False
                # tools.integrity renders a 2x clip extending up to a dynamic
                # 520 points above and 45 points below the matched caption. Use
                # that exact worst-case geometry before pixmap allocation.
                renderer_clip_height = min(
                    height,
                    min(max(height * 0.48, 220.0), 520.0) + 45.0,
                )
                render_width = math.ceil(width * 2.0)
                render_height = math.ceil(renderer_clip_height * 2.0)
                if render_width * render_height > self.visual_max_render_pixels:
                    return False
            return True
        finally:
            document.close()

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
        """Copy a regular external source into descriptor-anchored owner storage."""

        return copy_path_to_owner(
            source_path,
            upload_root=self.upload_root,
            owner_id=owner_id,
            max_bytes=max_bytes,
        )

    def remove_source(self, source_path: str | Path | None) -> bool:
        """Remove one retained owner file through descriptor-relative lookup."""

        return remove_owner_file(self.upload_root, source_path)

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
                if not self.remove_source(raw_path):
                    raise OSError("Descriptor-relative orphan deletion was refused.")
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

    def get(
        self,
        *,
        owner_id: str,
        doc_id: str,
        verify_visual: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return current source capability; expensive PDF checks are opt-in."""

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
        visual_candidate = bool(source is not None and source.suffix.lower() == ".pdf")
        record["source_path"] = str(source) if source is not None else None
        record["source_retained"] = 1 if source is not None else 0
        record["visual_source_verified"] = bool(verify_visual and visual_candidate)
        record["visual_source_available"] = bool(
            visual_candidate
            and (
                not verify_visual
                or (
                    self._source_matches_document(source, owner, doc_id)
                    and self._visual_pdf_is_safe(source)
                )
            )
        )
        return record

    def retained_source_path(self, *, owner_id: str, doc_id: str) -> Optional[Path]:
        """Return a valid retained source without performing visual-analysis checks."""

        record = self.get(owner_id=owner_id, doc_id=doc_id, verify_visual=False)
        raw_path = str((record or {}).get("source_path") or "")
        return Path(raw_path) if raw_path else None

    def source_path(self, *, owner_id: str, doc_id: str) -> Optional[Path]:
        """Return an owner-scoped retained PDF only after identity and safety checks."""

        record = self.get(owner_id=owner_id, doc_id=doc_id, verify_visual=True)
        raw_path = str((record or {}).get("source_path") or "")
        if not raw_path or not bool((record or {}).get("visual_source_available")):
            return None
        return Path(raw_path)

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
