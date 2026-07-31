"""Compensating owner/document transaction across vector and sparse generations."""

from __future__ import annotations

import hashlib
import itertools
import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tools.sparse_index import SparseDocumentSnapshot, SparseField, SparseIndex
from tools.vector_generation import (
    VectorGenerationSnapshot,
    capture_vector_generation,
    restore_vector_generation,
)

try:
    from tools.security import normalize_owner_id
except ImportError:  # focused-test fallback
    def normalize_owner_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("owner_id is required.")
        return value.strip()

_LOCK_STRIPES = tuple(threading.RLock() for _ in range(257))
_MAX_TEXT_CHARS = 50_000_000
_MAX_SECTIONS = 10_000
_MAX_METADATA_ITEMS = 1_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _sha256(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip().lower()
    if allow_empty and not cleaned:
        return ""
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 hex digest.")
    return cleaned


def _metadata(value: Mapping[str, Any]) -> dict[str, str | int | float | bool]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    result: dict[str, str | int | float | bool] = {}
    try:
        for index, (raw_key, raw_value) in enumerate(
            itertools.islice(value.items(), _MAX_METADATA_ITEMS + 1)
        ):
            if index >= _MAX_METADATA_ITEMS:
                raise ValueError("metadata contains too many fields.")
            key = _identifier(raw_key, "metadata key", maximum=200)
            if isinstance(raw_value, bool) or isinstance(raw_value, int):
                result[key] = raw_value
            elif isinstance(raw_value, float) and math.isfinite(raw_value):
                result[key] = raw_value
            elif isinstance(raw_value, str) and len(raw_value) <= 100_000 and "\x00" not in raw_value:
                result[key] = raw_value
            else:
                raise ValueError("metadata contains an unsupported value.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("metadata is not safely iterable.") from exc
    return result


def _document_lock(owner_id: str, doc_id: str) -> threading.RLock:
    digest = hashlib.sha256(f"{owner_id}\x00{doc_id}".encode("utf-8")).digest()
    return _LOCK_STRIPES[int.from_bytes(digest[:4], "big") % len(_LOCK_STRIPES)]


@dataclass(frozen=True)
class DocumentGenerationManifest:
    owner_id: str
    doc_id: str
    content_sha256: str
    profile_fingerprint: str
    vector_rows: int
    sparse_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self,
            "content_sha256",
            _sha256(self.content_sha256, "content_sha256"),
        )
        object.__setattr__(
            self,
            "profile_fingerprint",
            _sha256(self.profile_fingerprint, "profile_fingerprint"),
        )
        if isinstance(self.vector_rows, bool) or not isinstance(self.vector_rows, int) or self.vector_rows < 0:
            raise ValueError("vector_rows must be a non-negative integer.")
        if isinstance(self.sparse_generation, bool) or not isinstance(self.sparse_generation, int) or self.sparse_generation <= 0:
            raise ValueError("sparse_generation must be a positive integer.")


@dataclass(frozen=True)
class CrossStoreSnapshot:
    vector: VectorGenerationSnapshot
    sparse: SparseDocumentSnapshot | None


class IndexCoordinationError(RuntimeError):
    """Raised when a store operation fails, with rollback status retained."""

    def __init__(self, message: str, *, rollback_errors: Sequence[str] = ()) -> None:
        self.rollback_errors = tuple(rollback_errors)
        suffix = (
            " Rollback errors: " + ", ".join(self.rollback_errors)
            if self.rollback_errors
            else ""
        )
        super().__init__(message + suffix)


class IndexCoordinator:
    def __init__(self, *, rag: Any, sparse: SparseIndex) -> None:
        if not hasattr(rag, "add_document") or not hasattr(rag, "delete_document"):
            raise ValueError("rag must expose add_document and delete_document.")
        if not isinstance(sparse, SparseIndex):
            raise ValueError("sparse must be a SparseIndex.")
        self.rag = rag
        self.sparse = sparse

    def snapshot(self, *, owner_id: str, doc_id: str) -> CrossStoreSnapshot:
        owner = normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id", 200)
        return CrossStoreSnapshot(
            vector=capture_vector_generation(
                self.rag,
                owner_id=owner,
                doc_id=document_id,
            ),
            sparse=self.sparse.snapshot_document(
                owner_id=owner,
                doc_id=document_id,
            ),
        )

    def _restore(
        self,
        *,
        owner_id: str,
        doc_id: str,
        snapshot: CrossStoreSnapshot,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        try:
            restore_vector_generation(
                self.rag,
                owner_id=owner_id,
                doc_id=doc_id,
                snapshot=snapshot.vector,
            )
        except Exception as exc:
            errors.append(f"vector:{type(exc).__name__}")
        try:
            self.sparse.restore_document(
                owner_id=owner_id,
                doc_id=doc_id,
                snapshot=snapshot.sparse,
            )
        except Exception as exc:
            errors.append(f"sparse:{type(exc).__name__}")
        return tuple(errors)

    def replace_document(
        self,
        *,
        owner_id: str,
        doc_id: str,
        text: str,
        sections: Iterable[Any] | None,
        metadata: Mapping[str, Any],
        sparse_fields: Iterable[SparseField],
        content_sha256: str,
        profile_fingerprint: str,
        chunk_size: int = 1_000,
        overlap: int = 120,
        expected_sparse_generation: int | None = None,
    ) -> DocumentGenerationManifest:
        owner = normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id", 200)
        if not isinstance(text, str) or not text.strip() or len(text) > _MAX_TEXT_CHARS:
            raise ValueError("text is empty or exceeds the document limit.")
        clean_metadata = _metadata(metadata)
        if clean_metadata.get("owner_id") != owner:
            raise ValueError("metadata.owner_id must match owner_id.")
        clean_content_hash = _sha256(content_sha256, "content_sha256")
        clean_profile = _sha256(profile_fingerprint, "profile_fingerprint")
        clean_metadata["content_sha256"] = clean_content_hash
        clean_metadata["embedding_profile_fingerprint"] = clean_profile
        bounded_sections = None
        if sections is not None:
            if isinstance(sections, (str, bytes, bytearray)):
                raise ValueError("sections must be an iterable of section objects.")
            bounded_sections = list(itertools.islice(iter(sections), _MAX_SECTIONS + 1))
            if len(bounded_sections) > _MAX_SECTIONS:
                raise ValueError("sections exceed the document limit.")
        bounded_fields = list(itertools.islice(iter(sparse_fields), 10_001))
        if len(bounded_fields) > 10_000:
            raise ValueError("sparse_fields exceed the document limit.")
        lock = _document_lock(owner, document_id)
        with lock:
            prior = self.snapshot(owner_id=owner, doc_id=document_id)
            try:
                vector_rows = self.rag.add_document(
                    doc_id=document_id,
                    text=text,
                    sections=bounded_sections,
                    metadata=clean_metadata,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    replace=True,
                )
                if isinstance(vector_rows, bool) or not isinstance(vector_rows, int) or vector_rows <= 0:
                    raise RuntimeError("Vector backend returned an invalid chunk count.")
                sparse_generation = self.sparse.replace_document(
                    owner_id=owner,
                    doc_id=document_id,
                    fields=bounded_fields,
                    profile_fingerprint=clean_profile,
                    metadata={
                        **clean_metadata,
                        "vector_rows": vector_rows,
                    },
                    expected_generation=expected_sparse_generation,
                )
                return DocumentGenerationManifest(
                    owner_id=owner,
                    doc_id=document_id,
                    content_sha256=clean_content_hash,
                    profile_fingerprint=clean_profile,
                    vector_rows=vector_rows,
                    sparse_generation=sparse_generation,
                )
            except Exception as exc:
                rollback_errors = self._restore(
                    owner_id=owner,
                    doc_id=document_id,
                    snapshot=prior,
                )
                raise IndexCoordinationError(
                    f"Cross-store replacement failed ({type(exc).__name__}).",
                    rollback_errors=rollback_errors,
                ) from exc

    def delete_document(self, *, owner_id: str, doc_id: str) -> bool:
        owner = normalize_owner_id(owner_id)
        document_id = _identifier(doc_id, "doc_id", 200)
        lock = _document_lock(owner, document_id)
        with lock:
            prior = self.snapshot(owner_id=owner, doc_id=document_id)
            existed = prior.vector.row_count > 0 or prior.sparse is not None
            if not existed:
                return False
            try:
                self.rag.delete_document(owner_id=owner, doc_id=document_id)
                self.sparse.delete_document(owner_id=owner, doc_id=document_id)
                return True
            except Exception as exc:
                rollback_errors = self._restore(
                    owner_id=owner,
                    doc_id=document_id,
                    snapshot=prior,
                )
                raise IndexCoordinationError(
                    f"Cross-store deletion failed ({type(exc).__name__}).",
                    rollback_errors=rollback_errors,
                ) from exc

    def scan_owner(self, *, owner_id: str) -> dict[str, tuple[str, ...]]:
        owner = normalize_owner_id(owner_id)
        vector_rows = self.rag.list_documents(owner_id=owner, limit=5_000)
        vector_ids = {
            str(item.get("doc_id"))
            for item in vector_rows
            if isinstance(item, Mapping) and isinstance(item.get("doc_id"), str)
        }
        sparse_ids = set(self.sparse.list_document_ids(owner_id=owner, limit=100_000))
        return {
            "vector_only": tuple(sorted(vector_ids - sparse_ids)),
            "sparse_only": tuple(sorted(sparse_ids - vector_ids)),
            "aligned": tuple(sorted(vector_ids & sparse_ids)),
        }


__all__ = [
    "CrossStoreSnapshot",
    "DocumentGenerationManifest",
    "IndexCoordinationError",
    "IndexCoordinator",
]
