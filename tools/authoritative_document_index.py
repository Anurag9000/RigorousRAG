"""Integration boundary for privacy-finalized document indexing and deletion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from tools.embedding_registry import resolve_embedding_profile
from tools.generation_store import GenerationRecord
from tools.index_coordinator import CrossStoreSnapshot, _document_lock
from tools.security import normalize_owner_id
from tools.sparse_fields import build_sparse_fields
from tools.sparse_runtime import get_authoritative_index_coordinator
from tools.three_store_coordinator import AuthoritativeIndexCoordinator


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _text(value: Any, label: str, maximum: int = 50_000_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is empty or invalid.")
    return cleaned


@dataclass(frozen=True)
class AuthoritativeIndexResult:
    generation: GenerationRecord
    sparse_field_count: int

    @property
    def vector_rows(self) -> int:
        return self.generation.vector_rows


@dataclass(frozen=True)
class AuthoritativeDocumentSnapshot:
    owner_id: str
    doc_id: str
    stores: CrossStoreSnapshot
    generation: GenerationRecord | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id"))
        if not isinstance(self.stores, CrossStoreSnapshot):
            raise ValueError("stores must be a CrossStoreSnapshot.")
        if (
            self.stores.vector.owner_id != self.owner_id
            or self.stores.vector.doc_id != self.doc_id
        ):
            raise ValueError("snapshot store scope is invalid.")
        if self.generation is not None and (
            self.generation.owner_id != self.owner_id
            or self.generation.doc_id != self.doc_id
        ):
            raise ValueError("snapshot generation scope is invalid.")


def capture_authoritative_document(
    *,
    owner_id: str,
    doc_id: str,
    rag: Any,
    coordinator: AuthoritativeIndexCoordinator | None = None,
) -> AuthoritativeDocumentSnapshot:
    owner = normalize_owner_id(owner_id)
    document_id = _identifier(doc_id, "doc_id")
    selected = coordinator or get_authoritative_index_coordinator(rag=rag)
    lock = _document_lock(owner, document_id)
    with lock:
        return AuthoritativeDocumentSnapshot(
            owner,
            document_id,
            selected.index.snapshot(owner_id=owner, doc_id=document_id),
            selected.generations.current(owner_id=owner, doc_id=document_id),
        )


def restore_authoritative_document(
    snapshot: AuthoritativeDocumentSnapshot,
    *,
    rag: Any,
    coordinator: AuthoritativeIndexCoordinator | None = None,
) -> None:
    if not isinstance(snapshot, AuthoritativeDocumentSnapshot):
        raise ValueError("snapshot must be an AuthoritativeDocumentSnapshot.")
    selected = coordinator or get_authoritative_index_coordinator(rag=rag)
    lock = _document_lock(snapshot.owner_id, snapshot.doc_id)
    with lock:
        errors = list(
            selected._restore_index(
                owner_id=snapshot.owner_id,
                doc_id=snapshot.doc_id,
                prior=snapshot.stores,
            )
        )
        errors.extend(
            selected._restore_generation(
                owner_id=snapshot.owner_id,
                doc_id=snapshot.doc_id,
                prior=snapshot.generation,
            )
        )
        if errors:
            raise RuntimeError(
                "Authoritative document restoration was incomplete: "
                + ", ".join(errors)
            )


def commit_finalized_document(
    document: Any,
    *,
    owner_id: str,
    rag: Any,
    metadata: Mapping[str, Any],
    coordinator: AuthoritativeIndexCoordinator | None = None,
    profile_name: str | None = None,
    chunk_size: int = 1_000,
    overlap: int = 120,
    audit_metadata: Mapping[str, Any] | None = None,
) -> AuthoritativeIndexResult:
    """Commit only after privacy masking and metadata protection are complete."""

    owner = normalize_owner_id(owner_id)
    doc_id = _identifier(getattr(document, "id", None), "document.id")
    text = _text(getattr(document, "text", None), "document.text")
    if not isinstance(metadata, Mapping) or metadata.get("owner_id") != owner:
        raise ValueError("metadata.owner_id must match owner_id.")
    content_hash = metadata.get("content_sha256")
    computed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if content_hash is None:
        content_hash = computed_hash
    if content_hash != computed_hash:
        raise ValueError(
            "metadata.content_sha256 does not match finalized document text."
        )
    profile = resolve_embedding_profile(profile_name)
    fields = build_sparse_fields(document, doc_id=doc_id)
    selected = coordinator or get_authoritative_index_coordinator(rag=rag)
    record = selected.replace_document(
        owner_id=owner,
        doc_id=doc_id,
        text=text,
        sections=getattr(document, "sections", None),
        metadata=metadata,
        sparse_fields=fields,
        content_sha256=content_hash,
        profile_fingerprint=profile.fingerprint,
        chunk_size=chunk_size,
        overlap=overlap,
        audit_metadata=audit_metadata,
    )
    return AuthoritativeIndexResult(record, len(fields))


def delete_authoritative_document(
    *,
    owner_id: str,
    doc_id: str,
    rag: Any,
    coordinator: AuthoritativeIndexCoordinator | None = None,
    audit_metadata: Mapping[str, Any] | None = None,
) -> bool:
    owner = normalize_owner_id(owner_id)
    document_id = _identifier(doc_id, "doc_id")
    selected = coordinator or get_authoritative_index_coordinator(rag=rag)
    return selected.delete_document(
        owner_id=owner,
        doc_id=document_id,
        audit_metadata=audit_metadata,
    )


def install_authoritative_rag_deletion() -> None:
    """Make public RAG deletion coordinate vector, sparse, and manifest state."""

    from tools.rag import RAGLayer

    current = RAGLayer.delete_document
    if not hasattr(RAGLayer, "_authoritative_raw_delete_document"):
        setattr(RAGLayer, "_authoritative_raw_delete_document", current)
    if getattr(current, "_rigorousrag_authoritative_delete", False):
        return

    def authoritative_delete(
        self: Any,
        *,
        owner_id: str,
        doc_id: str,
    ) -> bool:
        return delete_authoritative_document(
            owner_id=owner_id,
            doc_id=doc_id,
            rag=self,
            audit_metadata={"operation": "document_delete"},
        )

    setattr(authoritative_delete, "_rigorousrag_authoritative_delete", True)
    RAGLayer.delete_document = authoritative_delete


install_authoritative_rag_deletion()


__all__ = [
    "AuthoritativeDocumentSnapshot",
    "AuthoritativeIndexResult",
    "capture_authoritative_document",
    "commit_finalized_document",
    "delete_authoritative_document",
    "install_authoritative_rag_deletion",
    "restore_authoritative_document",
]
