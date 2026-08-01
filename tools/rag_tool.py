"""Owner-scoped uploaded-document retrieval with candidate and corpus modes."""

from __future__ import annotations

import itertools
import math
import operator
from collections.abc import Mapping
from typing import Any, List, Optional

from tools.corpus_hybrid_retrieval import CorpusEvidence, retrieve_corpus_evidence
from tools.hybrid_retrieval import RetrievalCandidate, rank_candidates
from tools.models import Citation
from tools.rag import get_rag_layer
from tools.reranking import build_reranker
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_generation_store, get_sparse_index

_RETRIEVAL_MODES = {
    "dense",
    "lexical",
    "hybrid",
    "corpus-sparse",
    "corpus-hybrid",
}
_RERANKERS = {"none", "heuristic", "cross-encoder"}
_MAX_CITATIONS = 50
_MAX_PAGE_NUMBER = 1_000_000
_CORPUS_EXTRA_FIELDS = {
    "filename",
    "document_score",
    "field_type",
    "term_frequencies",
    "positions",
}

RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs",
        "description": (
            "Search the authenticated user's uploaded documents using dense, "
            "candidate-pool hybrid, or generation-validated corpus retrieval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10_000,
                },
                "doc_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                },
                "use_hyde": {"type": "boolean", "default": False},
                "use_multi_query": {"type": "boolean", "default": False},
                "retrieval_mode": {
                    "type": "string",
                    "enum": sorted(_RETRIEVAL_MODES),
                    "default": "dense",
                },
                "reranker": {
                    "type": "string",
                    "enum": sorted(_RERANKERS),
                    "default": "none",
                },
                "candidate_pool": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
                "diversity_lambda": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.82,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def _contains_ascii_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _prose(
    value: Any,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if _contains_ascii_control(rendered) or len(rendered) > maximum:
        raise ValueError(
            f"{label} may contain at most {maximum:,} valid characters."
        )
    if not rendered and not allow_empty:
        raise ValueError(f"{label} is required.")
    return rendered


def _identifier(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or _contains_ascii_control(rendered)
    ):
        raise ValueError(
            f"{label} must contain 1-{maximum} valid characters."
        )
    return rendered


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        numeric = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= numeric <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return numeric


def _choice(value: Any, label: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{label} must be one of: {', '.join(sorted(allowed))}."
        )
    return value


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return numeric


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(numeric, 1.0))


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


def _metadata(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bounded_chunks(values: Any, maximum: int) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError(
            "The vector backend returned an invalid chunk collection."
        )
    try:
        return list(itertools.islice(iter(values), maximum))
    except Exception as exc:
        raise RuntimeError(
            "The vector backend returned an invalid chunk collection."
        ) from exc


def _expanded_queries(
    rag: Any,
    query: str,
    *,
    enabled: bool,
    agent_client: Any,
    model: str,
) -> list[str]:
    if not enabled:
        return [query]
    try:
        raw = rag.generate_expanded_queries(
            query,
            agent_client,
            model=model,
            count=3,
        )
    except Exception:
        return [query]
    if raw is None or isinstance(raw, (str, bytes, bytearray)):
        return [query]
    result = [query]
    try:
        for value in itertools.islice(iter(raw), 4):
            if not isinstance(value, str):
                continue
            candidate = value.strip()
            if (
                candidate
                and len(candidate) <= 20_000
                and not _contains_ascii_control(candidate)
                and candidate not in result
            ):
                result.append(candidate)
    except Exception:
        return [query]
    return result


def _safe_corpus_extra(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        for key in _CORPUS_EXTRA_FIELDS:
            item = value.get(key)
            if key == "filename":
                if isinstance(item, str) and item.strip():
                    result[key] = item.strip()[:500]
            elif key == "document_score":
                result[key] = round(_score(item), 6)
            elif key == "field_type":
                if isinstance(item, str) and item.strip():
                    result[key] = item.strip()[:200]
            elif key == "term_frequencies" and isinstance(item, Mapping):
                result[key] = {
                    str(term)[:200]: _integer(frequency, "frequency", 1, 1_000_000)
                    for term, frequency in itertools.islice(item.items(), 100)
                    if isinstance(term, str) and term
                }
            elif key == "positions" and isinstance(item, Mapping):
                positions: dict[str, tuple[int, ...]] = {}
                for term, raw_positions in itertools.islice(item.items(), 100):
                    if not isinstance(term, str) or not term:
                        continue
                    if isinstance(raw_positions, (str, bytes, bytearray)):
                        continue
                    try:
                        positions[term[:200]] = tuple(
                            _integer(position, "position", 0, 100_000_000)
                            for position in itertools.islice(
                                iter(raw_positions),
                                1_000,
                            )
                        )
                    except Exception:
                        continue
                result[key] = positions
    except Exception:
        return {}
    return result


def _corpus_citations(
    queries: list[str],
    *,
    owner: str,
    document_id: str | None,
    rag: Any,
    mode: str,
    reranker_name: str,
    pool: int,
    requested: int,
    diversity: float,
) -> list[Citation]:
    sparse = get_sparse_index()
    generations = get_generation_store()
    corpus_mode = "sparse" if mode == "corpus-sparse" else "hybrid"
    merged: dict[str, CorpusEvidence] = {}
    errors: list[Exception] = []
    successes = 0
    for current_query in queries[:4]:
        try:
            evidence = retrieve_corpus_evidence(
                current_query,
                owner_id=owner,
                rag=rag,
                sparse=sparse,
                generations=generations,
                doc_id=document_id,
                mode=corpus_mode,
                top_k=pool,
                dense_pool=pool,
                sparse_pool=pool,
                diversity_lambda=diversity,
            )
            successes += 1
        except Exception as exc:
            errors.append(exc)
            continue
        for item in evidence:
            if not isinstance(item, CorpusEvidence):
                continue
            prior = merged.get(item.evidence_id)
            if prior is None or item.score > prior.score:
                merged[item.evidence_id] = item
    if successes == 0 and errors:
        raise RuntimeError("Corpus retrieval is unavailable.") from errors[0]

    rows = list(merged.values())
    if reranker_name != "none" and rows:
        reranker = build_reranker(reranker_name)
        ranked = rank_candidates(
            queries[0],
            [
                RetrievalCandidate(
                    item.evidence_id,
                    item.text,
                    item.doc_id,
                    item.score,
                )
                for item in rows
            ],
            mode="dense",
            top_k=requested,
            reranker=reranker.score,
            diversity_lambda=diversity,
            max_per_source=requested,
            dense_weight=0.75,
            reranker_weight=0.25,
        )
        by_id = {item.evidence_id: item for item in rows}
        rows = [
            by_id[item.candidate.candidate_id]
            for item in ranked
            if item.candidate.candidate_id in by_id
        ]
    else:
        rows = sorted(
            rows,
            key=lambda item: (item.score, item.evidence_id),
            reverse=True,
        )[:requested]

    citations: list[Citation] = []
    for item in rows[:requested]:
        extra = _safe_corpus_extra(item.metadata)
        title = extra.pop("filename", "Uploaded document")
        protected = {
            "section_title": item.section,
            "relevance": round(item.dense_score, 6),
            "fused_score": round(item.score, 6),
            "dense_score": round(item.dense_score, 6),
            "sparse_score": round(item.sparse_score, 6),
            "retrieval_mode": mode,
            "reranker": reranker_name,
            "generation_sequence": item.generation_sequence,
            "embedding_profile_fingerprint": item.profile_fingerprint,
            "evidence_kind": item.source_kind,
        }
        try:
            citations.append(
                Citation(
                    label=f"[{len(citations) + 1}]",
                    title=title,
                    url=f"local://{item.doc_id}",
                    source_type="uploaded_document",
                    snippet=item.text[:4_000],
                    quote=item.text[:4_000],
                    source_id=item.evidence_id,
                    doc_id=item.doc_id,
                    chunk_id=item.evidence_id,
                    page_number=item.page_number,
                    metadata={**extra, **protected},
                )
            )
        except Exception:
            continue
    return citations


def _candidate_citations(
    query: str,
    *,
    owner: str,
    document_id: str | None,
    rag: Any,
    mode: str,
    reranker_name: str,
    pool: int,
    requested: int,
    diversity: float,
    use_multi_query: bool,
    agent_client: Any,
    expansion_model: str,
) -> list[Citation]:
    backend_limit = (
        requested
        if mode == "dense" and reranker_name == "none"
        else pool
    )
    chunks = rag.query(
        query,
        n_results=backend_limit,
        owner_id=owner,
        doc_id=document_id,
        use_multi_query=use_multi_query,
        agent_client=agent_client,
        expansion_model=expansion_model,
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
        if metadata_owner != owner or not isinstance(actual_doc_id, str):
            continue
        try:
            actual_doc_id = _identifier(actual_doc_id, "doc_id", 200)
        except ValueError:
            continue
        if document_id is not None and actual_doc_id != document_id:
            continue
        raw_chunk_id = _safe_attr(chunk, "id", "")
        try:
            chunk_id = _identifier(raw_chunk_id, "chunk_id", 500)
        except ValueError:
            continue
        if chunk_id in records:
            continue
        raw_text = _safe_attr(chunk, "text", "")
        chunk_text = raw_text[:4_000] if isinstance(raw_text, str) else ""
        parent_text = (
            raw_parent[:4_000]
            if isinstance(raw_parent, str)
            else chunk_text
        )
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or not 1 <= page_number <= _MAX_PAGE_NUMBER
        ):
            page_number = None
        dense_score = _score(_safe_attr(chunk, "score", 0.0))
        source_id = (
            parent_id
            if isinstance(parent_id, str) and parent_id.strip()
            else actual_doc_id
        )
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
        ranked_rows = [
            (
                identifier,
                records[identifier]["dense_score"],
                {"dense": records[identifier]["dense_score"]},
            )
            for identifier in order[:requested]
        ]
    else:
        reranker = (
            None
            if reranker_name == "none"
            else build_reranker(reranker_name).score
        )
        ranked = rank_candidates(
            query,
            [records[identifier]["candidate"] for identifier in order],
            mode=mode,
            top_k=requested,
            reranker=reranker,
            diversity_lambda=diversity,
            max_per_source=requested,
        )
        ranked_rows = [
            (
                item.candidate.candidate_id,
                item.score,
                item.components,
            )
            for item in ranked
        ]

    citations: list[Citation] = []
    for chunk_id, fused_score, components in ranked_rows:
        row = records[chunk_id]
        filename = row["filename"]
        title = (
            filename[:500]
            if isinstance(filename, str) and filename.strip()
            else "Uploaded document"
        )
        section = row["section_title"]
        try:
            citations.append(
                Citation(
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
                        "section_title": (
                            section[:500]
                            if isinstance(section, str)
                            else None
                        ),
                        "relevance": round(row["dense_score"], 6),
                        "fused_score": round(_score(fused_score), 6),
                        "dense_score": round(
                            _score(components.get("dense", 0.0)),
                            6,
                        ),
                        "lexical_score": round(
                            _score(components.get("lexical", 0.0)),
                            6,
                        ),
                        "reranker_score": round(
                            _score(components.get("reranker", 0.0)),
                            6,
                        ),
                        "retrieval_mode": mode,
                        "reranker": reranker_name,
                    },
                )
            )
        except Exception:
            continue
    return citations


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
    """Retrieve evidence after owner, document, and backend validation."""

    retrieval_query = _prose(
        query,
        "query",
        maximum=10_000,
        allow_empty=True,
    )
    if not retrieval_query:
        return []
    if not isinstance(owner_id, str):
        raise ValueError("owner_id must be a string.")
    owner = normalize_owner_id(owner_id)
    document_id = (
        _identifier(doc_id, "doc_id", 200)
        if doc_id is not None
        else None
    )
    if not isinstance(use_hyde, bool):
        raise ValueError("use_hyde must be a boolean.")
    if not isinstance(use_multi_query, bool):
        raise ValueError("use_multi_query must be a boolean.")
    model = _identifier(expansion_model, "expansion_model", 200)
    requested = _integer(n_results, "n_results", 1, _MAX_CITATIONS)
    mode = _choice(retrieval_mode, "retrieval_mode", _RETRIEVAL_MODES)
    reranker_name = _choice(reranker, "reranker", _RERANKERS)
    pool = max(
        requested,
        _integer(candidate_pool, "candidate_pool", 1, _MAX_CITATIONS),
    )
    diversity = _unit(diversity_lambda, "diversity_lambda")

    rag = get_rag_layer()
    if use_hyde:
        generated = rag.generate_hyde_query(
            retrieval_query,
            agent_client,
            model=model,
        )
        if not isinstance(generated, str):
            raise RuntimeError(
                "The retrieval expansion backend returned invalid text."
            )
        retrieval_query = generated.strip()
        if (
            not retrieval_query
            or len(retrieval_query) > 20_000
            or _contains_ascii_control(retrieval_query)
        ):
            raise RuntimeError(
                "The retrieval expansion backend returned invalid text."
            )
    queries = _expanded_queries(
        rag,
        retrieval_query,
        enabled=use_multi_query,
        agent_client=agent_client,
        model=model,
    )
    if mode.startswith("corpus-"):
        return _corpus_citations(
            queries,
            owner=owner,
            document_id=document_id,
            rag=rag,
            mode=mode,
            reranker_name=reranker_name,
            pool=pool,
            requested=requested,
            diversity=diversity,
        )
    return _candidate_citations(
        retrieval_query,
        owner=owner,
        document_id=document_id,
        rag=rag,
        mode=mode,
        reranker_name=reranker_name,
        pool=pool,
        requested=requested,
        diversity=diversity,
        use_multi_query=use_multi_query,
        agent_client=agent_client,
        expansion_model=model,
    )
