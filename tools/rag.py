"""Validated public boundary over the persistent owner-scoped RAG implementation."""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tools.config import bounded_int_env

for _name, _default, _minimum, _maximum in (
    ("MAX_CHUNKS_PER_DOCUMENT", 10_000, 100, 100_000),
    ("DOCUMENT_LIST_SCAN_BATCH", 500, 50, 5000),
    ("MAX_DOCUMENT_LIST_SCAN_CHUNKS", 100_000, 50, 1_000_000),
    ("MAX_VECTOR_METADATA_ITEMS", 200, 10, 2000),
    ("MAX_SECTIONS_PER_DOCUMENT", 10_000, 1, 100_000),
    ("MAX_RAG_QUERY_CHARS", 20_000, 1000, 100_000),
):
    bounded_int_env(
        _name,
        _default,
        minimum=_minimum,
        maximum=_maximum,
        write_back=True,
    )
_raw_chroma_path = Path(os.getenv("CHROMA_PATH", "rag_storage"))
if _raw_chroma_path.is_symlink():
    raise ValueError("CHROMA_PATH may not be a symbolic link.")

from tools import rag_legacy as _implementation
from tools.security import normalize_owner_id

_MAX_METADATA_ITEMS = bounded_int_env(
    "MAX_VECTOR_METADATA_ITEMS",
    200,
    minimum=10,
    maximum=2000,
)
_MAX_SECTIONS = bounded_int_env(
    "MAX_SECTIONS_PER_DOCUMENT",
    10_000,
    minimum=1,
    maximum=100_000,
)
_MAX_QUERY_CHARS = bounded_int_env(
    "MAX_RAG_QUERY_CHARS",
    20_000,
    minimum=1000,
    maximum=100_000,
)
_MAX_WHERE_CHARS = 20_000


def _bounded_identifier(value: str, label: str, *, limit: int = 200) -> str:
    result = str(value or "").strip()
    if not result or len(result) > limit:
        raise ValueError(f"{label} must contain between 1 and {limit} characters.")
    return result


def _bounded_model(value: str) -> str:
    return _bounded_identifier(value, "model", limit=200)


def _bounded_query(value: str) -> str:
    query = str(value or "").strip()
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(
            f"RAG queries may contain at most {_MAX_QUERY_CHARS} characters."
        )
    return query


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object.")
    if len(metadata) > _MAX_METADATA_ITEMS:
        raise ValueError(
            f"metadata may contain at most {_MAX_METADATA_ITEMS} fields."
        )
    cleaned: Dict[str, str | int | float | bool] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key).strip()
        if not key or len(key) > 200:
            raise ValueError("Vector metadata keys must contain 1-200 characters.")
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, int):
            cleaned[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                cleaned[key] = value
        elif isinstance(value, str):
            cleaned[key] = value[:4000]
    owner = normalize_owner_id(str(cleaned.get("owner_id") or ""))
    cleaned["owner_id"] = owner
    return cleaned


def _bounded_sections(sections: Optional[Iterable[Any]]) -> Optional[List[Any]]:
    if sections is None:
        return None
    values = list(itertools.islice(iter(sections), _MAX_SECTIONS + 1))
    if len(values) > _MAX_SECTIONS:
        raise ValueError(
            f"A document may contain at most {_MAX_SECTIONS} semantic sections."
        )
    return values


def _bounded_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if where is None:
        return None
    if not isinstance(where, dict):
        raise ValueError("where must be a metadata-filter object.")
    try:
        encoded = json.dumps(where, ensure_ascii=False, allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError("where contains unsupported filter values.") from exc
    if len(encoded) > _MAX_WHERE_CHARS:
        raise ValueError("where exceeds the metadata-filter size limit.")
    return where


class RAGLayer(_implementation.RAGLayer):
    """RAG implementation with caller-independent input and result validation."""

    def __init__(
        self,
        persist_directory: str = _implementation.CHROMA_PATH,
        *,
        collection_name: str = _implementation.COLLECTION_NAME,
        embedding_model: str = _implementation.DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        raw_path = Path(persist_directory)
        if raw_path.is_symlink():
            raise ValueError("CHROMA_PATH may not be a symbolic link.")
        path = str(Path(os.path.abspath(raw_path)))
        super().__init__(
            persist_directory=path,
            collection_name=_bounded_identifier(
                collection_name,
                "collection_name",
                limit=200,
            ),
            embedding_model=_bounded_model(embedding_model),
        )

    def add_document(
        self,
        doc_id: str,
        text: Optional[str],
        metadata: Dict[str, Any],
        *,
        sections: Optional[Iterable[Any]] = None,
        chunk_size: int = 1000,
        overlap: int = 120,
        replace: bool = True,
    ) -> int:
        document_id = _bounded_identifier(doc_id, "doc_id")
        cleaned_metadata = _clean_metadata(metadata)
        bounded_sections = _bounded_sections(sections)
        if text is not None and len(str(text)) > 50_000_000:
            raise ValueError("Document text exceeds the vector-ingestion character limit.")
        return super().add_document(
            document_id,
            text,
            cleaned_metadata,
            sections=bounded_sections,
            chunk_size=int(chunk_size),
            overlap=int(overlap),
            replace=bool(replace),
        )

    def delete_document(self, *, owner_id: str, doc_id: str) -> None:
        return super().delete_document(
            owner_id=normalize_owner_id(owner_id),
            doc_id=_bounded_identifier(doc_id, "doc_id"),
        )

    def generate_hyde_query(
        self,
        query: str,
        agent_client: Optional[Any] = None,
        *,
        model: str = "gpt-4o-mini",
    ) -> str:
        bounded = _bounded_query(query)
        if not bounded:
            return ""
        generated = super().generate_hyde_query(
            bounded,
            agent_client,
            model=_bounded_model(model),
        )
        return str(generated or bounded)[:_MAX_QUERY_CHARS]

    def generate_expanded_queries(
        self,
        query: str,
        agent_client: Optional[Any] = None,
        *,
        model: str = "gpt-4o-mini",
        count: int = 3,
    ) -> List[str]:
        bounded = _bounded_query(query)
        if not bounded:
            return []
        requested = max(1, min(int(count), 4))
        generated = super().generate_expanded_queries(
            bounded,
            agent_client,
            model=_bounded_model(model),
            count=requested,
        )
        unique: List[str] = []
        for item in generated:
            value = str(item or "").strip()[:2000]
            if value and value not in unique:
                unique.append(value)
            if len(unique) >= requested + 1:
                break
        return unique or [bounded]

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        *,
        owner_id: str = "default_user",
        doc_id: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
        use_multi_query: bool = False,
        agent_client: Optional[Any] = None,
        expansion_model: str = "gpt-4o-mini",
    ) -> List[_implementation.Chunk]:
        query = _bounded_query(query_text)
        if not query:
            return []
        owner = normalize_owner_id(owner_id)
        document_id = (
            _bounded_identifier(doc_id, "doc_id") if doc_id is not None else None
        )
        scoped_where = _bounded_where(where)
        requested = max(1, min(int(n_results), 50))
        queries = [query]
        if use_multi_query:
            queries = self.generate_expanded_queries(
                query,
                agent_client,
                model=_bounded_model(expansion_model),
            )

        filters: List[Optional[Dict[str, Any]]] = [self._owner_filter(owner)]
        if document_id:
            filters.append({"doc_id": {"$eq": document_id}})
        filters.append(scoped_where)
        combined_where = _implementation._combine_filters(filters)
        candidates: Dict[str, _implementation.Chunk] = {}
        errors: List[Exception] = []
        successful_queries = 0
        for current_query in queries[:5]:
            try:
                results = self.collection.query(
                    query_texts=[str(current_query)[:_MAX_QUERY_CHARS]],
                    n_results=requested,
                    where=combined_where,
                    include=["documents", "metadatas", "distances"],
                )
                successful_queries += 1
            except Exception as exc:
                errors.append(exc)
                continue
            docs = (results.get("documents") or [[]])[0]
            result_ids = (results.get("ids") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            for index, raw_id in enumerate(result_ids):
                chunk_id = str(raw_id or "").strip()
                if not chunk_id or len(chunk_id) > 500:
                    continue
                raw_metadata = metas[index] if index < len(metas) else {}
                if not isinstance(raw_metadata, dict):
                    continue
                metadata = dict(raw_metadata)
                if str(metadata.get("owner_id") or "") != owner:
                    continue
                if document_id and str(metadata.get("doc_id") or "") != document_id:
                    continue
                try:
                    distance = float(distances[index]) if index < len(distances) else 1.0
                except (TypeError, ValueError):
                    distance = 1.0
                if not math.isfinite(distance) or distance < 0:
                    distance = 1.0
                text = str(docs[index]) if index < len(docs) else ""
                candidate = _implementation.Chunk(
                    id=chunk_id,
                    text=text[:100_000],
                    metadata=metadata,
                    distance=distance,
                    score=1.0 / (1.0 + distance),
                )
                existing = candidates.get(candidate.id)
                if existing is None or candidate.distance < existing.distance:
                    candidates[candidate.id] = candidate
        if successful_queries == 0 and errors:
            raise RuntimeError("Vector retrieval is unavailable.") from errors[0]
        return sorted(candidates.values(), key=lambda item: item.distance)[:requested]

    def list_documents(self, *, owner_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        owner = normalize_owner_id(owner_id)
        documents = super().list_documents(owner_id=owner, limit=limit)
        return [
            item
            for item in documents
            if str(item.get("owner_id") or "") == owner
            and 0 < len(str(item.get("doc_id") or "")) <= 200
        ]


_RAG_INSTANCES: Dict[str, RAGLayer] = {}
_RAG_LOCK = _implementation.threading.Lock()


def get_rag_layer(persist_directory: str = _implementation.CHROMA_PATH) -> RAGLayer:
    raw_path = Path(persist_directory)
    if raw_path.is_symlink():
        raise ValueError("CHROMA_PATH may not be a symbolic link.")
    path = str(Path(os.path.abspath(raw_path)))
    with _RAG_LOCK:
        instance = _RAG_INSTANCES.get(path)
        if instance is None:
            instance = RAGLayer(persist_directory=path)
            _RAG_INSTANCES[path] = instance
        return instance


_implementation.RAGLayer = RAGLayer
_implementation.get_rag_layer = get_rag_layer
_implementation._RAG_INSTANCES = _RAG_INSTANCES
_implementation._RAG_LOCK = _RAG_LOCK
_implementation.__doc__ = __doc__
sys.modules[__name__] = _implementation
