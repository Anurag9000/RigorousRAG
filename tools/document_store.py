"""Truthful and path-safe retained-source registry boundary.

The complete SQLite implementation remains in ``document_store_legacy``. This module
normalizes registry budgets, makes verification flags truthful, and prevents
registry/upload roots from being silently redirected through final-path symlinks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tools.config import bounded_int_env
from tools import document_store_legacy as _implementation

_original_document_store = _implementation.DocumentStore


def _lexical_absolute(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _normalize_registry_environment() -> None:
    for name, default, minimum, maximum in (
        ("ORPHAN_GRACE_SECONDS", 3600, 60, 31_536_000),
        ("VISUAL_MAX_PDF_PAGES", 500, 1, 5000),
        ("VISUAL_MAX_RENDER_PIXELS", 2_000_000, 1_000_000, 100_000_000),
    ):
        bounded_int_env(
            name,
            default,
            minimum=minimum,
            maximum=maximum,
            write_back=True,
        )


class DocumentStore(_original_document_store):
    """Registry with bounded budgets, truthful flags, and safe storage roots."""

    def __init__(
        self,
        path: str | Path | None = None,
        upload_root: str | Path | None = None,
    ) -> None:
        _normalize_registry_environment()
        raw_path = Path(path or os.getenv("DOCUMENT_DB_PATH", "data/documents.sqlite3"))
        raw_root = Path(upload_root or os.getenv("UPLOAD_DIR", "uploads"))
        if raw_path.is_symlink():
            raise ValueError("DOCUMENT_DB_PATH may not be a symbolic link.")
        if raw_root.is_symlink():
            raise ValueError("UPLOAD_DIR may not be a symbolic link.")
        super().__init__(
            path=_lexical_absolute(raw_path),
            upload_root=_lexical_absolute(raw_root),
        )
        if self.path.is_symlink():
            raise ValueError("DOCUMENT_DB_PATH may not be a symbolic link.")
        if self.upload_root.is_symlink():
            raise ValueError("UPLOAD_DIR may not be a symbolic link.")

    def _connect(self):
        if self.path.is_symlink():
            raise ValueError("DOCUMENT_DB_PATH became a symbolic link.")
        return super()._connect()

    def ping(self) -> bool:
        try:
            return super().ping()
        except (OSError, ValueError):
            return False

    def get(
        self,
        *,
        owner_id: str,
        doc_id: str,
        verify_visual: bool = False,
    ) -> Optional[Dict[str, Any]]:
        record = super().get(
            owner_id=owner_id,
            doc_id=doc_id,
            verify_visual=verify_visual,
        )
        if record is None:
            return None
        raw_path = str(record.get("source_path") or "")
        check_performed = bool(
            verify_visual
            and raw_path
            and Path(raw_path).suffix.lower() == ".pdf"
        )
        record["visual_source_check_performed"] = check_performed
        record["visual_source_verified"] = bool(
            check_performed and record.get("visual_source_available")
        )
        return record


def get_document_store(
    path: str | Path | None = None,
    upload_root: str | Path | None = None,
) -> DocumentStore:
    _normalize_registry_environment()
    raw_path = Path(path or os.getenv("DOCUMENT_DB_PATH", "data/documents.sqlite3"))
    raw_root = Path(upload_root or os.getenv("UPLOAD_DIR", "uploads"))
    if raw_path.is_symlink():
        raise ValueError("DOCUMENT_DB_PATH may not be a symbolic link.")
    if raw_root.is_symlink():
        raise ValueError("UPLOAD_DIR may not be a symbolic link.")
    resolved_path = str(_lexical_absolute(raw_path))
    resolved_root = str(_lexical_absolute(raw_root))
    key = (resolved_path, resolved_root)
    with _implementation._DOCUMENT_STORE_LOCK:
        store = _implementation._DOCUMENT_STORES.get(key)
        if store is None or not isinstance(store, DocumentStore):
            store = DocumentStore(resolved_path, resolved_root)
            _implementation._DOCUMENT_STORES[key] = store
        return store


_implementation.DocumentStore = DocumentStore
_implementation.get_document_store = get_document_store
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
