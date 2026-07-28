"""Agent tool for owner-scoped uploaded-document retrieval."""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable, List, Optional

from tools.models import Citation
from tools.rag import get_rag_layer
from tools.security import normalize_owner_id

RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs",
        "description": (
            "Search only the authenticated user's uploaded documents. Optionally "
            "restrict retrieval to one document ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10_000,
                    "description": "Question or topic to search for.",
                },
                "doc_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Optional exact document ID from the document library.",
                },
                "use_hyde": {
                    "type": "boolean",
                    "description": "Use one hypothetical evidence passage to improve recall.",
                    "default": False,
                },
                "use_multi_query": {
                    "type": "boolean",
                    "description": "Generate a small number of alternative retrieval queries.",
                    "default": False,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

_MAX_CITATIONS = 50


def _text(value: Any, label: str, *, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if "\x00" in rendered or len(rendered) > maximum:
        raise ValueError(f"{label} may contain at most {maximum:,} valid characters.")
    if not rendered and not allow_empty:
        raise ValueError(f"{label} is required.")
    return rendered


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
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


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _finite_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(score, 1.0))


def _bounded_chunks(values: Any, maximum: int) -> Iterable[Any]:
    if isinstance(values, (str, bytes, bytearray)):
        return ()
    try:
        return itertools.islice(iter(values), maximum)
    except Exception:
        return ()


def search_uploaded_docs(
    query: str,
    *,
    owner_id: str = "default_user",
    doc_id: Optional[str] = None,
    use_hyde: bool = False,
    use_multi_query: bool = False,
    agent_client: Optional[Any] = None,
    expansion_model: str = "gpt-4o-mini",
    n_results: int = 5,
) -> List[Citation]:
    """Retrieve evidence with mandatory owner and document provenance checks."""

    retrieval_query = _text(query, "query", maximum=10_000, allow_empty=True)
    if not retrieval_query:
        return []
    if not isinstance(owner_id, str):
        raise ValueError("owner_id must be a string.")
    owner = normalize_owner_id(owner_id)
    document_id = None
    if doc_id is not None:
        document_id = _text(doc_id, "doc_id", maximum=200)
    if not isinstance(use_hyde, bool):
        raise ValueError("use_hyde must be a boolean.")
    if not isinstance(use_multi_query, bool):
        raise ValueError("use_multi_query must be a boolean.")
    model = _text(expansion_model, "expansion_model", maximum=200)
    requested = _integer(
        n_results,
        "n_results",
        minimum=1,
        maximum=_MAX_CITATIONS,
    )

    rag = get_rag_layer()
    if use_hyde:
        retrieval_query = rag.generate_hyde_query(
            retrieval_query,
            agent_client,
            model=model,
        )
        if not retrieval_query:
            return []
    chunks = rag.query(
        retrieval_query,
        n_results=requested,
        owner_id=owner,
        doc_id=document_id,
        use_multi_query=use_multi_query,
        agent_client=agent_client,
        expansion_model=model,
    )

    citations: List[Citation] = []
    for chunk in _bounded_chunks(chunks, requested):
        raw_metadata = _safe_attr(chunk, "metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        metadata_owner = metadata.get("owner_id")
        actual_doc_id = metadata.get("doc_id")
        if not isinstance(metadata_owner, str) or metadata_owner.strip() != owner:
            continue
        if not isinstance(actual_doc_id, str):
            continue
        actual_doc_id = actual_doc_id.strip()
        if not actual_doc_id or len(actual_doc_id) > 200 or "\x00" in actual_doc_id:
            continue
        if document_id is not None and actual_doc_id != document_id:
            continue
        raw_text = _safe_attr(chunk, "text", "")
        chunk_text = raw_text if isinstance(raw_text, str) else ""
        raw_parent = metadata.get("parent_text")
        parent_text = raw_parent if isinstance(raw_parent, str) else chunk_text
        page_number = metadata.get("page_number")
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            page_number = None
        raw_chunk_id = _safe_attr(chunk, "id", "")
        if not isinstance(raw_chunk_id, str):
            continue
        chunk_id = raw_chunk_id.strip()
        if not chunk_id or len(chunk_id) > 500 or "\x00" in chunk_id:
            continue
        filename = metadata.get("filename")
        title = filename if isinstance(filename, str) and filename.strip() else "Uploaded document"
        citations.append(
            Citation(
                label=f"[{len(citations) + 1}]",
                title=title,
                url=f"local://{actual_doc_id}",
                source_type="uploaded_document",
                snippet=parent_text,
                quote=chunk_text,
                source_id=chunk_id,
                doc_id=actual_doc_id,
                chunk_id=chunk_id,
                page_number=page_number,
                metadata={
                    "section_title": (
                        metadata.get("section_title")
                        if isinstance(metadata.get("section_title"), str)
                        else None
                    ),
                    "relevance": round(_finite_score(_safe_attr(chunk, "score", 0.0)), 6),
                },
            )
        )
    return citations
