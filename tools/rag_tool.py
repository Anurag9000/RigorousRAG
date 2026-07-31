"""Agent tool for owner-scoped uploaded-document retrieval."""

from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Mapping
from typing import Any, List, Optional

from tools.hybrid_retrieval import RetrievalCandidate, rank_candidates
from tools.models import Citation
from tools.rag import get_rag_layer
from tools.reranking import build_reranker
from tools.security import normalize_owner_id

RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs",
        "description": (
            "Search only the authenticated user's uploaded documents. Optionally "
            "restrict retrieval to one document ID and use bounded hybrid ranking."
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
                "retrieval_mode": {
                    "type": "string",
                    "enum": ["dense", "lexical", "hybrid"],
                    "description": "Dense ordering, candidate-pool BM25, or fused hybrid ranking.",
                    "default": "dense",
                },
                "reranker": {
                    "type": "string",
                    "enum": ["none", "heuristic", "cross-encoder"],
                    "description": "Optional bounded second-stage reranker.",
                    "default": "none",
                },
                "candidate_pool": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Dense candidate pool before lexical fusion and diversity selection.",
                    "default": 20,
                },
                "diversity_lambda": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "MMR relevance weight; 1.0 disables redundancy penalty.",
                    "default": 0.82,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

_MAX_CITATIONS = 50
_MAX_PAGE_NUMBER = 1_000_000


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _prose(value: Any, label: str, *, maximum: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if _contains_ascii_control(rendered) or len(rendered) > maximum:
        raise ValueError(f"{label} may contain at most {maximum:,} valid characters.")
    if not rendered and not allow_empty:
        raise ValueError(f"{label} is required.")
    return rendered


def _identifier(value: Any, label: str, *, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if not rendered or len(rendered) > maximum or _contains_ascii_control(rendered):
        raise ValueError(f"{label} must contain 1-{maximum} valid characters.")
    return rendered


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = operator.index(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    numeric = int(parsed)
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _choice(value: Any, label: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(allowed))}.")
    return value


def _unit_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return numeric


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _finite_score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return max(0.0, min(score, 1.0))


def _bounded_chunks(values: Any, maximum: int) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError("The vector backend returned an invalid chunk collection.")
    try:
        return list(itertools.islice(iter(values), maximum))
    except Exception as exc:
        raise RuntimeError("The vector backend returned an invalid chunk collection.") from exc


def _metadata(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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
    retrieval_mode: str = "dense",
    reranker: str = "none",
    candidate_pool: int = 20,
    diversity_lambda: float = 0.82,
) -> List[Citation]:
    """Retrieve evidence with mandatory owner/document provenance checks.

    Dense/no-reranker mode intentionally preserves the historical ordering and raw
    relevance contract.  Lexical and hybrid modes rerank only the already scoped dense
    candidate pool; the persistent corpus-level sparse index is a separate capability.
    """

    retrieval_query = _prose(query, "query", maximum=10_000, allow_empty=True)
    if not retrieval_query:
        return []
    if not isinstance(owner_id, str):
        raise ValueError("owner_id must be a string.")
    owner = normalize_owner_id(owner_id)
    document_id = _identifier(doc_id, "doc_id", maximum=200) if doc_id is not None else None
    if not isinstance(use_hyde, bool):
        raise ValueError("use_hyde must be a boolean.")
    if not isinstance(use_multi_query, bool):
        raise ValueError("use_multi_query must be a boolean.")
    model = _identifier(expansion_model, "expansion_model", maximum=200)
    requested = _integer(n_results, "n_results", minimum=1, maximum=_MAX_CITATIONS)
    mode = _choice(retrieval_mode, "retrieval_mode", {"dense", "lexical", "hybrid"})
    reranker_name = _choice(reranker, "reranker", {"none", "heuristic", "cross-encoder"})
    pool = max(requested, _integer(candidate_pool, "candidate_pool", minimum=1, maximum=_MAX_CITATIONS))
    diversity = _unit_float(diversity_lambda, "diversity_lambda")

    rag = get_rag_layer()
    if use_hyde:
        generated = rag.generate_hyde_query(retrieval_query, agent_client, model=model)
        if not isinstance(generated, str):
            raise RuntimeError("The retrieval expansion backend returned invalid text.")
        retrieval_query = generated.strip()
        if not retrieval_query:
            return []
        if len(retrieval_query) > 20_000 or _contains_ascii_control(retrieval_query):
            raise RuntimeError("The retrieval expansion backend returned invalid text.")

    backend_limit = requested if mode == "dense" and reranker_name == "none" else pool
    chunks = rag.query(
        retrieval_query,
        n_results=backend_limit,
        owner_id=owner,
        doc_id=document_id,
        use_multi_query=use_multi_query,
        agent_client=agent_client,
        expansion_model=model,
    )

    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for chunk in _bounded_chunks(chunks, backend_limit):
        metadata = _metadata(_safe_attr(chunk, "metadata", {}))
        try:
            metadata_owner = metadata.get("owner_id")
            actual_doc_id = metadata.get("doc_id")
            raw_parent = metadata.get("parent_text")
            page_number = metadata.get("page_number")
            filename = metadata.get("filename")
            section_title = metadata.get("section_title")
            parent_id = metadata.get("parent_id")
        except Exception:
            continue
        if not isinstance(metadata_owner, str) or metadata_owner != owner:
            continue
        if not isinstance(actual_doc_id, str):
            continue
        actual_doc_id = actual_doc_id.strip()
        if not actual_doc_id or len(actual_doc_id) > 200 or _contains_ascii_control(actual_doc_id):
            continue
        if document_id is not None and actual_doc_id != document_id:
            continue
        raw_chunk_id = _safe_attr(chunk, "id", "")
        if not isinstance(raw_chunk_id, str):
            continue
        chunk_id = raw_chunk_id.strip()
        if not chunk_id or len(chunk_id) > 500 or _contains_ascii_control(chunk_id) or chunk_id in records:
            continue
        raw_text = _safe_attr(chunk, "text", "")
        chunk_text = raw_text[:4000] if isinstance(raw_text, str) else ""
        parent_text = raw_parent[:4000] if isinstance(raw_parent, str) else chunk_text
        if isinstance(page_number, bool) or not isinstance(page_number, int) or not 1 <= page_number <= _MAX_PAGE_NUMBER:
            page_number = None
        dense_score = _finite_score(_safe_attr(chunk, "score", 0.0))
        source_id = parent_id if isinstance(parent_id, str) and parent_id.strip() else actual_doc_id
        try:
            candidate = RetrievalCandidate(
                candidate_id=chunk_id,
                text=chunk_text,
                source_id=source_id[:500],
                dense_score=dense_score,
                metadata={"doc_id": actual_doc_id},
            )
        except Exception:
            continue
        records[chunk_id] = {
            "candidate": candidate,
            "doc_id": actual_doc_id,
            "text": chunk_text,
            "parent_text": parent_text,
            "page_number": page_number,
            "filename": filename,
            "section_title": section_title,
            "dense_score": dense_score,
        }
        order.append(chunk_id)

    if mode == "dense" and reranker_name == "none":
        ranked_rows = [(identifier, records[identifier]["dense_score"], {"dense": records[identifier]["dense_score"]}) for identifier in order[:requested]]
    else:
        built = build_reranker(reranker_name)
        rerank_callable = None if reranker_name == "none" else built.score
        ranked = rank_candidates(
            retrieval_query,
            [records[identifier]["candidate"] for identifier in order],
            mode=mode,
            top_k=requested,
            reranker=rerank_callable,
            diversity_lambda=diversity,
            max_per_source=requested,
        )
        ranked_rows = [(item.candidate.candidate_id, item.score, item.components) for item in ranked]

    citations: List[Citation] = []
    for chunk_id, fused_score, components in ranked_rows:
        row = records[chunk_id]
        filename = row["filename"]
        title = filename[:500] if isinstance(filename, str) and filename.strip() else "Uploaded document"
        section_title = row["section_title"]
        try:
            citation = Citation(
                label=f"[{len(citations) + 1}]",
                title=title,
                url=f"local://{row['doc_id']}",
                source_type="uploaded_document",
                snippet=row["parent_text"] or None,
                quote=row["text"] or None,
                source_id=chunk_id,
                doc_id=row["doc_id"],
                chunk_id=chunk_id,
                page_number=row["page_number"],
                metadata={
                    "section_title": section_title[:500] if isinstance(section_title, str) else None,
                    "relevance": round(row["dense_score"], 6),
                    "fused_score": round(_finite_score(fused_score), 6),
                    "dense_score": round(_finite_score(components.get("dense", 0.0)), 6),
                    "lexical_score": round(_finite_score(components.get("lexical", 0.0)), 6),
                    "reranker_score": round(_finite_score(components.get("reranker", 0.0)), 6),
                    "retrieval_mode": mode,
                    "reranker": reranker_name,
                },
            )
        except Exception:
            continue
        citations.append(citation)
    return citations
