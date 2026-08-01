"""Thread/context-local bridge for batch retained-source lifecycle intent.

Batch ingestion copies a source into owner storage before calling the shared
``index_document`` service. The copied path is recorded here and consumed once
for the matching owner/document identity. During indexing the document's private
``file_path`` is temporarily set to that verified retained copy, allowing the
existing authoritative lifecycle boundary to journal and commit the registry
side effect before the batch process can crash.
"""

from __future__ import annotations

import hashlib
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from tools.security import DEFAULT_MAX_UPLOAD_BYTES, normalize_owner_id
from tools.upload_storage import read_owner_file, validated_owner_file_path

_PENDING: ContextVar["PendingRetainedSource | None"] = ContextVar(
    "rigorousrag_pending_retained_source",
    default=None,
)


@dataclass(frozen=True)
class PendingRetainedSource:
    owner_id: str
    source_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        if not isinstance(self.source_path, str):
            raise ValueError("source_path must be a string.")
        rendered = self.source_path.strip()
        if (
            not rendered
            or len(rendered) > 4096
            or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
        ):
            raise ValueError("source_path is invalid.")
        object.__setattr__(self, "source_path", rendered)


def remember_retained_source(*, owner_id: str, source_path: str | Path) -> None:
    _PENDING.set(PendingRetainedSource(owner_id, str(source_path)))


def clear_retained_source() -> None:
    _PENDING.set(None)


def consume_retained_source(
    *,
    owner_id: str,
    doc_id: str,
    upload_root: str | Path,
) -> str | None:
    pending = _PENDING.get()
    _PENDING.set(None)
    if pending is None:
        return None
    owner = normalize_owner_id(owner_id)
    if pending.owner_id != owner:
        raise RuntimeError("retained-source intent belongs to another owner.")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise RuntimeError("retained-source intent requires a document identity.")
    candidate = validated_owner_file_path(upload_root, pending.source_path)
    if candidate is None:
        raise RuntimeError("retained-source intent no longer identifies a safe file.")
    payload = read_owner_file(
        upload_root,
        candidate,
        max_bytes=DEFAULT_MAX_UPLOAD_BYTES,
    )
    if payload is None:
        raise RuntimeError("retained-source intent could not read the copied source.")
    digest = hashlib.sha256(payload).hexdigest()
    expected = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"rigorousrag:{owner}:{digest}",
        )
    )
    if expected != doc_id.strip():
        raise RuntimeError("retained-source intent does not match the document bytes.")
    return str(candidate)


def install_document_store_source_boundary(module: ModuleType) -> None:
    document_store = getattr(module, "DocumentStore", None)
    if document_store is None:
        raise ImportError("document store boundary is unavailable.")
    if not hasattr(document_store, "_lifecycle_original_copy_source"):
        document_store._lifecycle_original_copy_source = document_store.copy_source
    if not hasattr(document_store, "_lifecycle_original_register"):
        document_store._lifecycle_original_register = document_store.register
    original_copy = document_store._lifecycle_original_copy_source
    original_register = document_store._lifecycle_original_register

    def copy_source(
        self: Any,
        *,
        owner_id: str,
        source_path: Any,
        max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ):
        copied = original_copy(
            self,
            owner_id=owner_id,
            source_path=source_path,
            max_bytes=max_bytes,
        )
        remember_retained_source(owner_id=owner_id, source_path=copied)
        return copied

    def register(
        self: Any,
        *,
        owner_id: str,
        doc_id: str,
        filename: str,
        mime_type: str,
        source_path: str | Path | None = None,
    ):
        candidate = (
            validated_owner_file_path(self.upload_root, source_path)
            if source_path is not None
            else None
        )
        if source_path is None or candidate is not None:
            current = self.get(owner_id=owner_id, doc_id=doc_id)
            current_path = str((current or {}).get("source_path") or "") or None
            candidate_path = str(candidate) if candidate is not None else None
            current_mime = str((current or {}).get("mime_type") or "")
            requested_mime = str(mime_type or "application/octet-stream")[:200]
            if (
                current is not None
                and current_path == candidate_path
                and current_mime == requested_mime
            ):
                return None
        return original_register(
            self,
            owner_id=owner_id,
            doc_id=doc_id,
            filename=filename,
            mime_type=mime_type,
            source_path=source_path,
        )

    copy_source._rigorousrag_lifecycle_source_boundary = True
    register._rigorousrag_lifecycle_source_boundary = True
    document_store.copy_source = copy_source
    document_store.register = register


def install_document_service_source_boundary(module: ModuleType) -> None:
    if not hasattr(module, "_lifecycle_original_index_document"):
        module._lifecycle_original_index_document = module.index_document
    original = module._lifecycle_original_index_document

    def index_document(document: Any, *, owner_id: str, **kwargs: Any) -> Any:
        from tools.document_store import get_document_store

        registry = get_document_store()
        retained = consume_retained_source(
            owner_id=owner_id,
            doc_id=str(getattr(document, "id", "")),
            upload_root=registry.upload_root,
        )
        if retained is None:
            return original(document, owner_id=owner_id, **kwargs)
        prior = getattr(document, "file_path", None)
        document.file_path = retained
        try:
            return original(document, owner_id=owner_id, **kwargs)
        finally:
            document.file_path = prior

    index_document._rigorousrag_lifecycle_source_boundary = True
    module.index_document = index_document


__all__ = [
    "PendingRetainedSource",
    "clear_retained_source",
    "consume_retained_source",
    "install_document_service_source_boundary",
    "install_document_store_source_boundary",
    "remember_retained_source",
]
