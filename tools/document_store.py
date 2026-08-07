"""Truthful and path-safe retained-source registry boundary.

The complete SQLite implementation remains in ``document_store_legacy``. This module
normalizes registry budgets, makes verification flags truthful, and prevents registry
or upload roots from being redirected through symbolic-link path components.
"""

from __future__ import annotations

import math
import os
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tools.config import bounded_int_env
from tools.privacy import mask_metadata_text
from tools.security import normalize_owner_id
from tools import document_store_legacy as _implementation

if not hasattr(_implementation, "_boundary_original_DocumentStore"):
    _implementation._boundary_original_DocumentStore = _implementation.DocumentStore
_original_document_store = _implementation._boundary_original_DocumentStore
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


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


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_redirecting(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0))
        & _WINDOWS_REPARSE_POINT
    )


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return int(metadata.st_dev), int(metadata.st_ino)


def _path_identity(path: Path, label: str, *, directory: bool) -> tuple[int, int]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise OSError(f"{label} could not be inspected safely.") from exc
    if _is_redirecting(metadata):
        raise ValueError(f"{label} may not be a symbolic link or reparse point.")
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise OSError(f"{label} must remain a {kind}.")
    return _identity(metadata)


def _lexical_absolute(value: str | os.PathLike[str], label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not rendered
        or len(rendered) > 4096
        or _contains_ascii_control(rendered)
    ):
        raise ValueError(f"{label} is invalid or too long.")
    path = Path(rendered)
    if not path.is_absolute():
        path = Path.cwd() / path
    absolute = Path(os.path.abspath(path))
    for candidate in (absolute, *absolute.parents):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError(f"{label} could not be validated safely.") from exc
        if _is_redirecting(metadata):
            raise ValueError(
                f"{label} may not contain symbolic links or reparse points."
            )
    return absolute


def _document_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("doc_id must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > 200
        or _contains_ascii_control(result)
    ):
        raise ValueError("doc_id must contain 1-200 valid characters.")
    return result


def _filename(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("filename must be a string.")
    if _contains_ascii_control(value):
        raise ValueError("filename may not contain control characters.")
    return mask_metadata_text(Path(value or "document").name)[:500] or "document"


def _mime_type(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("mime_type must be a string.")
    rendered = " ".join(value.replace("\r", " ").replace("\n", " ").split())[:200]
    return rendered or "application/octet-stream"


class _DocumentStoreBoundary(_original_document_store):
    """Registry with bounded budgets, truthful flags, and safe storage roots."""

    def __init__(
        self,
        path: str | Path | None = None,
        upload_root: str | Path | None = None,
    ) -> None:
        _normalize_registry_environment()
        selected_path = path if path is not None else os.getenv(
            "DOCUMENT_DB_PATH", "data/documents.sqlite3"
        )
        selected_root = upload_root if upload_root is not None else os.getenv(
            "UPLOAD_DIR", "uploads"
        )
        safe_path = _lexical_absolute(selected_path, "DOCUMENT_DB_PATH")
        safe_root = _lexical_absolute(selected_root, "UPLOAD_DIR")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_root.mkdir(parents=True, exist_ok=True)
        self._boundary_database_parent_identity = _path_identity(
            safe_path.parent,
            "DOCUMENT_DB_PATH parent",
            directory=True,
        )
        self._boundary_upload_root_identity = _path_identity(
            safe_root,
            "UPLOAD_DIR",
            directory=True,
        )
        self._boundary_database_identity: tuple[int, int] | None = None
        super().__init__(path=safe_path, upload_root=safe_root)
        self._boundary_database_identity = _path_identity(
            self.path,
            "DOCUMENT_DB_PATH",
            directory=False,
        )
        self._ensure_storage_paths()

    def _ensure_storage_paths(self) -> None:
        safe_path = _lexical_absolute(self.path, "DOCUMENT_DB_PATH")
        safe_root = _lexical_absolute(self.upload_root, "UPLOAD_DIR")
        if _path_identity(
            safe_path.parent,
            "DOCUMENT_DB_PATH parent",
            directory=True,
        ) != self._boundary_database_parent_identity:
            raise OSError("DOCUMENT_DB_PATH parent identity changed after initialization.")
        if _path_identity(
            safe_root,
            "UPLOAD_DIR",
            directory=True,
        ) != self._boundary_upload_root_identity:
            raise OSError("UPLOAD_DIR identity changed after initialization.")
        expected_database = self._boundary_database_identity
        if safe_path.exists():
            current_database = _path_identity(
                safe_path,
                "DOCUMENT_DB_PATH",
                directory=False,
            )
            if expected_database is not None and current_database != expected_database:
                raise OSError("DOCUMENT_DB_PATH identity changed after initialization.")
        elif expected_database is not None:
            raise OSError("DOCUMENT_DB_PATH disappeared after initialization.")

    def _connect(self):
        self._ensure_storage_paths()
        return super()._connect()

    def ping(self) -> bool:
        try:
            self._ensure_storage_paths()
            return super().ping()
        except (OSError, ValueError):
            return False

    def register(
        self,
        *,
        owner_id: str,
        doc_id: str,
        filename: str,
        mime_type: str,
        source_path: str | Path | None = None,
    ) -> Optional[str]:
        return super().register(
            owner_id=normalize_owner_id(owner_id),
            doc_id=_document_id(doc_id),
            filename=_filename(filename),
            mime_type=_mime_type(mime_type),
            source_path=source_path,
        )

    def get(
        self,
        *,
        owner_id: str,
        doc_id: str,
        verify_visual: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(verify_visual, bool):
            raise ValueError("verify_visual must be a boolean.")
        record = super().get(
            owner_id=normalize_owner_id(owner_id),
            doc_id=_document_id(doc_id),
            verify_visual=verify_visual,
        )
        if record is None:
            return None
        raw_path = str(record.get("source_path") or "")
        check_performed = bool(
            verify_visual and raw_path and Path(raw_path).suffix.lower() == ".pdf"
        )
        record["visual_source_check_performed"] = check_performed
        record["visual_source_verified"] = bool(
            check_performed and record.get("visual_source_available")
        )
        return record

    def delete(self, *, owner_id: str, doc_id: str) -> Optional[Dict[str, Any]]:
        return super().delete(
            owner_id=normalize_owner_id(owner_id),
            doc_id=_document_id(doc_id),
        )

    def cleanup_orphans(
        self,
        *,
        now: Optional[float] = None,
        job_store: Optional[Any] = None,
    ) -> int:
        if now is not None:
            if isinstance(now, bool):
                raise ValueError("now must be numeric.")
            try:
                current = float(now)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("now must be numeric.") from exc
            if not math.isfinite(current) or current < 0:
                raise ValueError("now must be finite and non-negative.")
            now = current
        self._ensure_storage_paths()
        return super().cleanup_orphans(now=now, job_store=job_store)


if not hasattr(_implementation, "_boundary_public_DocumentStore"):
    _implementation._boundary_public_DocumentStore = _DocumentStoreBoundary
DocumentStore = _implementation._boundary_public_DocumentStore
# Preserve the documented public class identity even though the implementation is
# wrapped by a hardened boundary subclass.
DocumentStore.__name__ = "DocumentStore"
DocumentStore.__qualname__ = "DocumentStore"


def get_document_store(
    path: str | Path | None = None,
    upload_root: str | Path | None = None,
) -> DocumentStore:
    _normalize_registry_environment()
    selected_path = path if path is not None else os.getenv(
        "DOCUMENT_DB_PATH", "data/documents.sqlite3"
    )
    selected_root = upload_root if upload_root is not None else os.getenv(
        "UPLOAD_DIR", "uploads"
    )
    resolved_path = str(_lexical_absolute(selected_path, "DOCUMENT_DB_PATH"))
    resolved_root = str(_lexical_absolute(selected_root, "UPLOAD_DIR"))
    key = (resolved_path, resolved_root)
    with _implementation._DOCUMENT_STORE_LOCK:
        store = _implementation._DOCUMENT_STORES.get(key)
        if store is None or not isinstance(store, DocumentStore):
            store = DocumentStore(resolved_path, resolved_root)
            _implementation._DOCUMENT_STORES[key] = store
        else:
            store._ensure_storage_paths()
        return store


_implementation.DocumentStore = DocumentStore
_implementation.get_document_store = get_document_store
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
