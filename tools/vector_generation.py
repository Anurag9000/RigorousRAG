"""Bounded owner/document vector-generation snapshots and exact restoration."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from tools.security import normalize_owner_id
except ImportError:  # focused-test fallback
    def normalize_owner_id(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("owner_id is required.")
        return value.strip()

_MAX_VECTOR_ROWS = 100_000
_MAX_ID_CHARS = 500
_MAX_DOCUMENT_CHARS = 5_000_000
_MAX_METADATA_ITEMS = 1_000
_MAX_METADATA_TEXT_CHARS = 100_000
_BATCH_SIZE = 128


def _identifier(value: Any, label: str, maximum: int) -> str:
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


def _metadata(value: Any) -> dict[str, str | int | float | bool]:
    if not isinstance(value, Mapping):
        raise ValueError("Vector metadata must be a mapping.")
    result: dict[str, str | int | float | bool] = {}
    try:
        for index, (raw_key, raw_value) in enumerate(
            itertools.islice(value.items(), _MAX_METADATA_ITEMS + 1)
        ):
            if index >= _MAX_METADATA_ITEMS:
                raise ValueError("Vector metadata contains too many fields.")
            key = _identifier(raw_key, "metadata key", 200)
            if isinstance(raw_value, bool) or isinstance(raw_value, int):
                result[key] = raw_value
            elif isinstance(raw_value, float):
                if not math.isfinite(raw_value):
                    raise ValueError("Vector metadata contains a non-finite number.")
                result[key] = raw_value
            elif isinstance(raw_value, str):
                if (
                    len(raw_value) > _MAX_METADATA_TEXT_CHARS
                    or "\x00" in raw_value
                ):
                    raise ValueError("Vector metadata contains invalid text.")
                result[key] = raw_value
            else:
                raise ValueError("Vector metadata contains an unsupported value.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Vector metadata is not safely iterable.") from exc
    return result


def _scope_filter(owner_id: str, doc_id: str) -> dict[str, Any]:
    return {
        "$and": [
            {"owner_id": {"$eq": owner_id}},
            {"doc_id": {"$eq": doc_id}},
        ]
    }


def _bounded_array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"Vector backend returned invalid {label} data.")
    if len(value) > _MAX_VECTOR_ROWS:
        raise ValueError("Vector generation exceeds the row limit.")
    return value


@dataclass(frozen=True)
class VectorGenerationSnapshot:
    owner_id: str
    doc_id: str
    ids: tuple[str, ...]
    documents: tuple[str, ...]
    metadatas: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        document_id = _identifier(self.doc_id, "doc_id", 200)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "doc_id", document_id)
        if not (
            isinstance(self.ids, tuple)
            and isinstance(self.documents, tuple)
            and isinstance(self.metadatas, tuple)
        ):
            raise ValueError("Vector snapshot arrays must be tuples.")
        if not len(self.ids) == len(self.documents) == len(self.metadatas):
            raise ValueError("Vector snapshot arrays must have equal lengths.")
        if len(self.ids) > _MAX_VECTOR_ROWS:
            raise ValueError("Vector snapshot exceeds the row limit.")
        clean_ids: list[str] = []
        clean_documents: list[str] = []
        clean_metadatas: list[dict[str, str | int | float | bool]] = []
        seen: set[str] = set()
        for raw_id, raw_document, raw_metadata in zip(
            self.ids,
            self.documents,
            self.metadatas,
        ):
            row_id = _identifier(raw_id, "vector row ID", _MAX_ID_CHARS)
            if row_id in seen:
                raise ValueError("Vector snapshot contains duplicate row IDs.")
            seen.add(row_id)
            if not isinstance(raw_document, str):
                raise ValueError("Vector documents must be strings.")
            if len(raw_document) > _MAX_DOCUMENT_CHARS or "\x00" in raw_document:
                raise ValueError("Vector document text is invalid or too large.")
            metadata = _metadata(raw_metadata)
            if metadata.get("owner_id") != owner:
                raise ValueError("Vector snapshot metadata escaped owner scope.")
            if metadata.get("doc_id") != document_id:
                raise ValueError("Vector snapshot metadata escaped document scope.")
            clean_ids.append(row_id)
            clean_documents.append(raw_document)
            clean_metadatas.append(metadata)
        object.__setattr__(self, "ids", tuple(clean_ids))
        object.__setattr__(self, "documents", tuple(clean_documents))
        object.__setattr__(self, "metadatas", tuple(clean_metadatas))

    @property
    def row_count(self) -> int:
        return len(self.ids)


def capture_vector_generation(
    rag: Any,
    *,
    owner_id: str,
    doc_id: str,
) -> VectorGenerationSnapshot:
    """Capture one complete owner/document generation from the vector backend."""

    owner = normalize_owner_id(owner_id)
    document_id = _identifier(doc_id, "doc_id", 200)
    collection = getattr(rag, "collection", None)
    getter = getattr(collection, "get", None)
    if not callable(getter):
        raise ValueError("rag.collection.get is required for vector snapshots.")
    try:
        result = getter(
            where=_scope_filter(owner, document_id),
            include=["documents", "metadatas"],
            limit=_MAX_VECTOR_ROWS,
        )
    except Exception as exc:
        raise RuntimeError("Vector generation capture failed.") from exc
    if not isinstance(result, Mapping):
        raise ValueError("Vector backend returned a non-object snapshot.")
    ids = _bounded_array(result.get("ids"), "identifier")
    documents = _bounded_array(result.get("documents"), "document")
    metadatas = _bounded_array(result.get("metadatas"), "metadata")
    return VectorGenerationSnapshot(
        owner,
        document_id,
        tuple(ids),
        tuple(documents),
        tuple(metadatas),
    )


def _delete_generation(rag: Any, owner_id: str, doc_id: str) -> None:
    deleter = getattr(rag, "delete_document", None)
    if callable(deleter):
        deleter(owner_id=owner_id, doc_id=doc_id)
        return
    collection = getattr(rag, "collection", None)
    collection_delete = getattr(collection, "delete", None)
    if not callable(collection_delete):
        raise ValueError("A vector deletion API is required for restoration.")
    collection_delete(where=_scope_filter(owner_id, doc_id))


def restore_vector_generation(
    rag: Any,
    *,
    owner_id: str,
    doc_id: str,
    snapshot: VectorGenerationSnapshot,
) -> None:
    """Replace the current vector rows with one previously validated snapshot."""

    owner = normalize_owner_id(owner_id)
    document_id = _identifier(doc_id, "doc_id", 200)
    if not isinstance(snapshot, VectorGenerationSnapshot):
        raise ValueError("snapshot must be a VectorGenerationSnapshot.")
    if snapshot.owner_id != owner or snapshot.doc_id != document_id:
        raise ValueError("Vector snapshot scope does not match restoration scope.")
    collection = getattr(rag, "collection", None)
    upsert = getattr(collection, "upsert", None)
    if snapshot.row_count and not callable(upsert):
        raise ValueError("rag.collection.upsert is required for vector restoration.")
    _delete_generation(rag, owner, document_id)
    if snapshot.row_count == 0:
        return
    try:
        for start in range(0, snapshot.row_count, _BATCH_SIZE):
            stop = min(start + _BATCH_SIZE, snapshot.row_count)
            upsert(
                ids=list(snapshot.ids[start:stop]),
                documents=list(snapshot.documents[start:stop]),
                metadatas=[dict(value) for value in snapshot.metadatas[start:stop]],
            )
    except Exception as exc:
        try:
            _delete_generation(rag, owner, document_id)
        except Exception:
            pass
        raise RuntimeError("Vector generation restoration failed.") from exc


__all__ = [
    "VectorGenerationSnapshot",
    "capture_vector_generation",
    "restore_vector_generation",
]
