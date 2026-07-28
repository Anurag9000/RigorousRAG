"""Validated public boundary over the persistent owner-scoped RAG implementation."""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.config import bounded_int_env
from tools.privacy import mask_metadata_text

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
    "MAX_VECTOR_METADATA_ITEMS", 200, minimum=10, maximum=2000
)
_MAX_SECTIONS = bounded_int_env(
    "MAX_SECTIONS_PER_DOCUMENT", 10_000, minimum=1, maximum=100_000
)
_MAX_QUERY_CHARS = bounded_int_env(
    "MAX_RAG_QUERY_CHARS", 20_000, minimum=1000, maximum=100_000
)
_MAX_WHERE_CHARS = 20_000
_MAX_DOCUMENT_TEXT_CHARS = 50_000_000
_MAX_SECTION_TEXT_CHARS = 5_000_000
_MAX_CHUNK_SIZE = 100_000
_MAX_RESULT_TEXT_CHARS = 100_000


def _safe_text(value: Any, *, limit: int, default: str = "") -> str:
    try:
        rendered = str(value if value is not None else default)
    except Exception:
        rendered = default
    return rendered[:limit]


def _bounded_identifier(value: Any, label: str, *, limit: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if not result or len(result) > limit or "\x00" in result:
        raise ValueError(f"{label} must contain between 1 and {limit} valid characters.")
    return result


def _bounded_integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _bounded_model(value: Any) -> str:
    return _bounded_identifier(value, "model", limit=200)


def _bounded_query(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("RAG queries must be strings.")
    query = value.strip()
    if len(query) > _MAX_QUERY_CHARS:
        raise ValueError(
            f"RAG queries may contain at most {_MAX_QUERY_CHARS} characters."
        )
    return query


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object.")
    if len(metadata) > _MAX_METADATA_ITEMS:
        raise ValueError(f"metadata may contain at most {_MAX_METADATA_ITEMS} fields.")
    cleaned: Dict[str, str | int | float | bool] = {}
    for raw_key, value in metadata.items():
        if not isinstance(raw_key, str):
            raise ValueError("Vector metadata keys must be strings.")
        key = raw_key.strip()
        if not key or len(key) > 200 or "\x00" in key:
            raise ValueError("Vector metadata keys must contain 1-200 valid characters.")
        if isinstance(value, bool):
            cleaned[key] = value
        elif isinstance(value, int):
            cleaned[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                cleaned[key] = value
        elif isinstance(value, str):
            cleaned[key] = mask_metadata_text(value)[:4000]
    owner = normalize_owner_id(str(cleaned.get("owner_id") or ""))
    cleaned["owner_id"] = owner
    return cleaned


def _section_data(section: Any, index: int) -> Dict[str, Any]:
    if hasattr(section, "model_dump"):
        raw = section.model_dump()
    elif isinstance(section, dict):
        raw = dict(section)
    else:
        raw = {
            "title": getattr(section, "title", f"Section {index + 1}"),
            "content": getattr(section, "content", ""),
            "page_number": getattr(section, "page_number", None),
        }
    if not isinstance(raw, dict):
        raise ValueError("Every semantic section must be an object-like value.")
    content = raw.get("content")
    if not isinstance(content, str):
        raise ValueError("Semantic section content must be text.")
    if len(content) > _MAX_SECTION_TEXT_CHARS:
        raise ValueError(
            f"A semantic section may contain at most {_MAX_SECTION_TEXT_CHARS} characters."
        )
    return raw


def _bounded_sections(sections: Optional[Iterable[Any]]) -> Optional[List[Any]]:
    if sections is None:
        return None
    if isinstance(sections, (str, bytes)):
        raise ValueError("sections must be an iterable of semantic-section objects.")
    try:
        values = list(itertools.islice(iter(sections), _MAX_SECTIONS + 1))
    except TypeError as exc:
        raise ValueError("sections must be iterable.") from exc
    if len(values) > _MAX_SECTIONS:
        raise ValueError(f"A document may contain at most {_MAX_SECTIONS} semantic sections.")
    total = 0
    for index, section in enumerate(values):
        data = _section_data(section, index)
        total += len(data.get("content") or "")
        if total > _MAX_DOCUMENT_TEXT_CHARS:
            raise ValueError("Semantic sections exceed the document text character limit.")
    return values


def _bounded_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if where is None:
        return None
    if not isinstance(where, dict):
        raise ValueError("where must be a metadata-filter object.")
    try:
        encoded = json.dumps(where, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("where contains unsupported filter values.") from exc
    if len(encoded) > _MAX_WHERE_CHARS:
        raise ValueError("where exceeds the metadata-filter size limit.")
    return where


def _absolute_storage_path(value: Any) -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("persist_directory must be a filesystem path.")
    rendered = os.fspath(value)
    if not rendered or len(rendered) > 4096 or "\x00" in rendered:
        raise ValueError("persist_directory is invalid or too long.")
    raw = Path(rendered)
    absolute = Path(os.path.abspath(raw))
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError("CHROMA_PATH may not contain symbolic-link components.")
    return str(absolute)


def _row(value: Any, *, maximum: int) -> List[Any]:
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        return []
    return value[0][:_bounded_integer(maximum, "maximum", minimum=1, maximum=100)]


def _result_metadata(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    cleaned: Dict[str, Any] = {}
    for index, (raw_key, item) in enumerate(value.items()):
        if index >= _MAX_METADATA_ITEMS or not isinstance(raw_key, str):
            break
        key = raw_key[:200]
        if isinstance(item, bool) or item is None:
            cleaned[key] = item
        elif isinstance(item, int):
            cleaned[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            cleaned[key] = item
        elif isinstance(item, str):
            cleaned[key] = mask_metadata_text(item)[:4000]
    return cleaned


class RAGLayer(_implementation.RAGLayer):
    """RAG implementation with caller-independent input and result validation."""

    def __init__(
        self,
        persist_directory: str = _implementation.CHROMA_PATH,
        *,
        collection_name: str = _implementation.COLLECTION_NAME,
        embedding_model: str = _implementation.DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        super().__init__(
            persist_directory=_absolute_storage_path(persist_directory),
            collection_name=_bounded_identifier(
                collection_name, "collection_name", limit=200
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
        if text is not None:
            if not isinstance(text, str):
                raise ValueError("Document text must be a string.")
            if len(text) > _MAX_DOCUMENT_TEXT_CHARS:
                raise ValueError("Document text exceeds the vector-ingestion character limit.")
        size = _bounded_integer(
            chunk_size, "chunk_size", minimum=1, maximum=_MAX_CHUNK_SIZE
        )
        overlap_value = _bounded_integer(
            overlap, "overlap", minimum=0, maximum=_MAX_CHUNK_SIZE - 1
        )
        if overlap_value >= size:
            raise ValueError("overlap must be smaller than chunk_size.")
        if not isinstance(replace, bool):
            raise ValueError("replace must be a boolean.")
        return super().add_document(
            document_id,
            text,
            cleaned_metadata,
            sections=bounded_sections,
            chunk_size=size,
            overlap=overlap_value,
            replace=replace,
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
            bounded, agent_client, model=_bounded_model(model)
        )
        return _safe_text(generated or bounded, limit=_MAX_QUERY_CHARS)

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
        requested = _bounded_integer(count, "count", minimum=1, maximum=4)
        generated = super().generate_expanded_queries(
            bounded, agent_client, model=_bounded_model(model), count=requested
        )
        unique: List[str] = []
        for item in itertools.islice(iter(generated or []), requested + 1):
            value = _safe_text(item, limit=2000).strip()
            if value and value not in unique:
                unique.append(value)
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
        requested = _bounded_integer(n_results, "n_results", minimum=1, maximum=50)
        owner = normalize_owner_id(owner_id)
        document_id = (
            _bounded_identifier(doc_id, "doc_id") if doc_id is not None else None
        )
        scoped_where = _bounded_where(where)
        if not isinstance(use_multi_query, bool):
            raise ValueError("use_multi_query must be a boolean.")
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
                    query_texts=[_safe_text(current_query, limit=_MAX_QUERY_CHARS)],
                    n_results=requested,
                    where=combined_where,
                    include=["documents", "metadatas", "distances"],
                )
                if not isinstance(results, dict):
                    raise ValueError("Vector backend returned a non-object response.")
                result_ids = _row(results.get("ids"), maximum=requested)
                docs = _row(results.get("documents"), maximum=requested)
                metas = _row(results.get("metadatas"), maximum=requested)
                distances = _row(results.get("distances"), maximum=requested)
                successful_queries += 1
            except Exception as exc:
                errors.append(exc)
                continue
            for index, raw_id in enumerate(result_ids):
                chunk_id = _safe_text(raw_id, limit=501).strip()
                if not chunk_id or len(chunk_id) > 500:
                    continue
                metadata = _result_metadata(metas[index] if index < len(metas) else {})
                if str(metadata.get("owner_id") or "") != owner:
                    continue
                if document_id and str(metadata.get("doc_id") or "") != document_id:
                    continue
                try:
                    distance = float(distances[index]) if index < len(distances) else 1.0
                except (TypeError, ValueError, OverflowError):
                    distance = 1.0
                if not math.isfinite(distance) or distance < 0:
                    distance = 1.0
                text = _safe_text(
                    docs[index] if index < len(docs) else "",
                    limit=_MAX_RESULT_TEXT_CHARS,
                )
                candidate = _implementation.Chunk(
                    id=chunk_id,
                    text=text,
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

    def list_documents(
        self, *, owner_id: str, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        owner = normalize_owner_id(owner_id)
        requested = _bounded_integer(limit, "limit", minimum=1, maximum=5000)
        seen: Dict[str, Dict[str, Any]] = {}
        offset = 0
        scanned = 0
        while len(seen) < requested and scanned < _implementation.MAX_LIST_SCAN_CHUNKS:
            batch_limit = min(
                _implementation.LIST_SCAN_BATCH,
                _implementation.MAX_LIST_SCAN_CHUNKS - scanned,
            )
            try:
                results = self.collection.get(
                    where=self._owner_filter(owner),
                    include=["metadatas"],
                    limit=batch_limit,
                    offset=offset,
                )
                if not isinstance(results, dict):
                    raise ValueError("Vector backend returned a non-object response.")
                result_ids = results.get("ids") or []
                metadatas = results.get("metadatas") or []
                if not isinstance(result_ids, list) or not isinstance(metadatas, list):
                    raise ValueError("Vector backend returned invalid document arrays.")
                result_ids = result_ids[:batch_limit]
                metadatas = metadatas[:batch_limit]
            except Exception as exc:
                raise RuntimeError("Vector document listing is unavailable.") from exc
            if not result_ids:
                break
            for raw_metadata in metadatas:
                metadata = _result_metadata(raw_metadata)
                if str(metadata.get("owner_id") or "") != owner:
                    continue
                doc_id_value = _safe_text(metadata.get("doc_id"), limit=201).strip()
                if not doc_id_value or len(doc_id_value) > 200 or doc_id_value in seen:
                    continue
                seen[doc_id_value] = {
                    "doc_id": doc_id_value,
                    "filename": mask_metadata_text(
                        _safe_text(metadata.get("filename") or doc_id_value, limit=500)
                    ),
                    "owner_id": owner,
                    "llm_summary": mask_metadata_text(
                        _safe_text(metadata.get("llm_summary"), limit=4000)
                    ) or None,
                    "mime_type": _safe_text(metadata.get("mime_type"), limit=200) or None,
                    "created_at": _safe_text(metadata.get("created_at"), limit=100) or None,
                }
                if len(seen) >= requested:
                    break
            consumed = len(result_ids)
            scanned += consumed
            offset += consumed
            if consumed < batch_limit:
                break
        return sorted(
            seen.values(),
            key=lambda item: str(item.get("created_at") or ""),
            reverse=True,
        )[:requested]


_RAG_INSTANCES: Dict[str, RAGLayer] = {}
_RAG_LOCK = _implementation.threading.Lock()


def get_rag_layer(persist_directory: str = _implementation.CHROMA_PATH) -> RAGLayer:
    path = _absolute_storage_path(persist_directory)
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
