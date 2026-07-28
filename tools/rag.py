"""Owner-scoped persistent vector retrieval."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

CHROMA_PATH = os.getenv("CHROMA_PATH", "rag_storage")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "academic_rag_v2")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SCHEMA_VERSION = 2
MAX_CHUNKS_PER_DOCUMENT = max(
    100,
    min(int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "10000")), 100_000),
)
LIST_SCAN_BATCH = max(50, min(int(os.getenv("DOCUMENT_LIST_SCAN_BATCH", "500")), 5000))
MAX_LIST_SCAN_CHUNKS = max(
    LIST_SCAN_BATCH,
    min(int(os.getenv("MAX_DOCUMENT_LIST_SCAN_CHUNKS", "100000")), 1_000_000),
)


class Chunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float = Field(default=0.0, ge=0.0)
    score: float = Field(default=0.0, ge=0.0)


def _metadata_scalar(value: Any) -> Optional[str | int | float | bool]:
    return value if isinstance(value, (str, int, float, bool)) else None


def _combine_filters(filters: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    usable = [item for item in filters if item]
    if not usable:
        return None
    return usable[0] if len(usable) == 1 else {"$and": usable}


class RAGLayer:
    """Chroma wrapper with mandatory owner scoping and compensating writes."""

    def __init__(
        self,
        persist_directory: str = CHROMA_PATH,
        *,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine", "schema_version": SCHEMA_VERSION},
        )
        self._write_lock = threading.RLock()

    @staticmethod
    def _owner_filter(owner_id: str) -> Dict[str, Any]:
        owner = (owner_id or "").strip()
        if not owner:
            raise ValueError("owner_id is required for every vector-store operation.")
        return {"owner_id": {"$eq": owner}}

    @staticmethod
    def _batched(values: Sequence[Any], size: int = 128) -> Iterable[Sequence[Any]]:
        for start in range(0, len(values), size):
            yield values[start:start + size]

    def ping(self) -> bool:
        """Return whether the vector collection can complete a metadata read."""

        try:
            self.collection.count()
            return True
        except Exception:
            return False

    def _rollback_document_write(
        self,
        *,
        new_ids: Sequence[str],
        existing_ids: Sequence[str],
        existing_documents: Sequence[str],
        existing_metadatas: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """Best-effort restoration after a partial upsert/delete sequence."""

        errors: List[str] = []
        old_id_set = set(existing_ids)
        new_only_ids = sorted(set(new_ids) - old_id_set)
        if new_only_ids:
            try:
                for batch in self._batched(new_only_ids):
                    self.collection.delete(ids=list(batch))
            except Exception as exc:
                errors.append(f"delete_new:{type(exc).__name__}")
        if existing_ids:
            try:
                for start in range(0, len(existing_ids), 128):
                    stop = start + 128
                    self.collection.upsert(
                        ids=list(existing_ids[start:stop]),
                        documents=list(existing_documents[start:stop]),
                        metadatas=list(existing_metadatas[start:stop]),
                    )
            except Exception as exc:
                errors.append(f"restore_old:{type(exc).__name__}")
        return errors

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
        """Index semantic sections with deterministic IDs and rollback on failure."""

        if not doc_id or not doc_id.strip():
            raise ValueError("doc_id must be non-empty.")
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_size must be positive and greater than overlap.")

        owner_id = str(metadata.get("owner_id", "")).strip()
        document_filter = _combine_filters([
            self._owner_filter(owner_id),
            {"doc_id": {"$eq": doc_id}},
        ])
        source_sections: List[Dict[str, Any]] = []
        if sections:
            for index, section in enumerate(sections):
                if hasattr(section, "model_dump"):
                    data = section.model_dump()
                elif isinstance(section, dict):
                    data = dict(section)
                else:
                    data = {
                        "title": getattr(section, "title", f"Section {index + 1}"),
                        "content": getattr(section, "content", ""),
                        "page_number": getattr(section, "page_number", None),
                    }
                content = str(data.get("content") or "").strip()
                if content:
                    source_sections.append({
                        "title": str(data.get("title") or f"Section {index + 1}")[:500],
                        "content": content,
                        "page_number": data.get("page_number"),
                    })
        if not source_sections:
            raw_text = (text or "").strip()
            if not raw_text:
                raise ValueError("Document contains no indexable text.")
            source_sections = [{"title": "Full Text", "content": raw_text, "page_number": None}]

        base_metadata: Dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            scalar = _metadata_scalar(value)
            if scalar is not None:
                base_metadata[str(key)] = scalar
        base_metadata.update({
            "owner_id": owner_id,
            "doc_id": doc_id,
            "schema_version": SCHEMA_VERSION,
        })

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for section_index, section in enumerate(source_sections):
            parents = self._chunk_text(section["content"], chunk_size * 3, overlap * 2)
            for parent_index, parent_text in enumerate(parents):
                parent_id = f"{doc_id}:s{section_index}:p{parent_index}"
                for child_index, child_text in enumerate(
                    self._chunk_text(parent_text, chunk_size, overlap)
                ):
                    chunk_id = f"{parent_id}:c{child_index}"
                    chunk_metadata = dict(base_metadata)
                    chunk_metadata.update({
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "section_title": section["title"],
                        "section_index": section_index,
                        "child_index": child_index,
                    })
                    page_number = section.get("page_number")
                    if isinstance(page_number, int) and page_number > 0:
                        chunk_metadata["page_number"] = page_number
                    ids.append(chunk_id)
                    documents.append(child_text)
                    metadatas.append(chunk_metadata)
                    if len(ids) > MAX_CHUNKS_PER_DOCUMENT:
                        raise ValueError(
                            "Document produced more than the configured chunk limit."
                        )
        if not ids:
            raise ValueError("Document produced no vector chunks.")

        with self._write_lock:
            existing_ids: List[str] = []
            existing_documents: List[str] = []
            existing_metadatas: List[Dict[str, Any]] = []
            if replace:
                existing = self.collection.get(
                    where=document_filter,
                    include=["documents", "metadatas"],
                )
                existing_ids = [str(value) for value in existing.get("ids") or []]
                old_documents = existing.get("documents") or []
                old_metadatas = existing.get("metadatas") or []
                existing_documents = [
                    str(old_documents[index]) if index < len(old_documents) else ""
                    for index in range(len(existing_ids))
                ]
                existing_metadatas = [
                    dict(old_metadatas[index] or {}) if index < len(old_metadatas) else {}
                    for index in range(len(existing_ids))
                ]
            try:
                for start in range(0, len(ids), 128):
                    stop = start + 128
                    self.collection.upsert(
                        ids=ids[start:stop],
                        documents=documents[start:stop],
                        metadatas=metadatas[start:stop],
                    )
                stale_ids = sorted(set(existing_ids) - set(ids))
                if stale_ids:
                    for batch in self._batched(stale_ids):
                        self.collection.delete(ids=list(batch))
            except Exception as exc:
                rollback_errors = self._rollback_document_write(
                    new_ids=ids,
                    existing_ids=existing_ids,
                    existing_documents=existing_documents,
                    existing_metadatas=existing_metadatas,
                )
                if rollback_errors:
                    raise RuntimeError(
                        "Vector write failed and rollback was incomplete: "
                        + ", ".join(rollback_errors)
                    ) from exc
                raise
        return len(ids)

    def delete_document(self, *, owner_id: str, doc_id: str) -> None:
        if not (doc_id or "").strip():
            raise ValueError("doc_id is required for deletion.")
        where = _combine_filters([
            self._owner_filter(owner_id),
            {"doc_id": {"$eq": doc_id}},
        ])
        with self._write_lock:
            self.collection.delete(where=where)

    def generate_hyde_query(
        self,
        query: str,
        agent_client: Optional[Any] = None,
        *,
        model: str = "gpt-4o-mini",
    ) -> str:
        if not agent_client:
            return query
        try:
            response = agent_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write one short hypothetical evidence passage that would answer "
                            "the query. Do not include instructions, citations, or unsupported "
                            "specific numbers. The passage is used only for retrieval."
                        ),
                    },
                    {"role": "user", "content": query[:4000]},
                ],
                max_tokens=180,
                temperature=0.0,
            )
            hypothetical = (response.choices[0].message.content or "").strip()
            return f"{query}\n{hypothetical}" if hypothetical else query
        except Exception:
            return query

    def generate_expanded_queries(
        self,
        query: str,
        agent_client: Optional[Any] = None,
        *,
        model: str = "gpt-4o-mini",
        count: int = 3,
    ) -> List[str]:
        if not agent_client or count <= 0:
            return [query]
        try:
            response = agent_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Return exactly {count} concise alternative academic search "
                            "queries, one per line, with no numbering or commentary."
                        ),
                    },
                    {"role": "user", "content": query[:4000]},
                ],
                max_tokens=160,
                temperature=0.2,
            )
            lines = [
                line.strip(" -\t")
                for line in (response.choices[0].message.content or "").splitlines()
                if line.strip(" -\t")
            ]
            unique: List[str] = [query]
            for line in lines:
                if line not in unique:
                    unique.append(line)
                if len(unique) >= count + 1:
                    break
            return unique
        except Exception:
            return [query]

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
    ) -> List[Chunk]:
        query_text = (query_text or "").strip()
        if not query_text:
            return []
        n_results = max(1, min(int(n_results), 50))
        filters: List[Optional[Dict[str, Any]]] = [self._owner_filter(owner_id)]
        if doc_id:
            filters.append({"doc_id": {"$eq": doc_id}})
        filters.append(where)
        scoped_where = _combine_filters(filters)
        queries = [query_text]
        if use_multi_query:
            queries = self.generate_expanded_queries(
                query_text,
                agent_client,
                model=expansion_model,
            )

        candidates: Dict[str, Chunk] = {}
        errors: List[Exception] = []
        successful_queries = 0
        for current_query in queries[:5]:
            try:
                results = self.collection.query(
                    query_texts=[current_query],
                    n_results=n_results,
                    where=scoped_where,
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
            for index, chunk_id in enumerate(result_ids):
                chunk_text = docs[index] if index < len(docs) else ""
                metadata = metas[index] if index < len(metas) and metas[index] else {}
                distance = float(distances[index]) if index < len(distances) else 1.0
                candidate = Chunk(
                    id=str(chunk_id),
                    text=str(chunk_text),
                    metadata=dict(metadata),
                    distance=max(distance, 0.0),
                    score=1.0 / (1.0 + max(distance, 0.0)),
                )
                existing = candidates.get(candidate.id)
                if existing is None or candidate.distance < existing.distance:
                    candidates[candidate.id] = candidate
        if successful_queries == 0 and errors:
            raise RuntimeError("Vector retrieval is unavailable.") from errors[0]
        return sorted(candidates.values(), key=lambda item: item.distance)[:n_results]

    def list_documents(self, *, owner_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        requested = max(1, min(int(limit), 5000))
        seen: Dict[str, Dict[str, Any]] = {}
        offset = 0
        scanned = 0
        while len(seen) < requested and scanned < MAX_LIST_SCAN_CHUNKS:
            batch_limit = min(LIST_SCAN_BATCH, MAX_LIST_SCAN_CHUNKS - scanned)
            results = self.collection.get(
                where=self._owner_filter(owner_id),
                include=["metadatas"],
                limit=batch_limit,
                offset=offset,
            )
            result_ids = results.get("ids") or []
            metadatas = results.get("metadatas") or []
            if not result_ids:
                break
            for metadata in metadatas:
                metadata = dict(metadata or {})
                doc_id = str(metadata.get("doc_id") or "")
                if not doc_id or doc_id in seen:
                    continue
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get("filename", doc_id),
                    "owner_id": owner_id,
                    "llm_summary": metadata.get("llm_summary"),
                    "mime_type": metadata.get("mime_type"),
                    "created_at": metadata.get("created_at"),
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

    @staticmethod
    def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
        if size <= 0 or overlap < 0 or size <= overlap:
            raise ValueError("Chunk size must be positive and greater than overlap.")
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            hard_end = min(start + size, len(cleaned))
            end = hard_end
            if hard_end < len(cleaned):
                lower_bound = start + int(size * 0.65)
                candidates = [
                    cleaned.rfind("\n\n", lower_bound, hard_end),
                    cleaned.rfind(". ", lower_bound, hard_end),
                    cleaned.rfind(" ", lower_bound, hard_end),
                ]
                boundary = max(candidates)
                if boundary > start:
                    end = boundary + (2 if cleaned[boundary:boundary + 2] == ". " else 0)
            chunk = cleaned[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(cleaned):
                break
            start = max(end - overlap, start + 1)
        return chunks


_RAG_INSTANCES: Dict[str, RAGLayer] = {}
_RAG_LOCK = threading.Lock()


def get_rag_layer(persist_directory: str = CHROMA_PATH) -> RAGLayer:
    path = str(Path(persist_directory).resolve())
    with _RAG_LOCK:
        instance = _RAG_INSTANCES.get(path)
        if instance is None:
            instance = RAGLayer(persist_directory=path)
            _RAG_INSTANCES[path] = instance
        return instance
