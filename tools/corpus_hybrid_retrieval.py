"""Generation-validated corpus-level dense+sparse evidence retrieval."""

from __future__ import annotations

import itertools
import math
import operator
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tools.generation_store import GenerationStore
from tools.hybrid_retrieval import RetrievalCandidate, mmr_select, weighted_fusion
from tools.security import normalize_owner_id
from tools.sparse_index import SparseIndex
from tools.sparse_types import SparseSearchHit

_MAX_RESULTS = 50
_MAX_POOL = 100
_MAX_TEXT = 100_000
_MAX_IDENTIFIER = 500
_MAX_FIELDS_PER_DOCUMENT = 20


def _identifier(value: Any, label: str, maximum: int = _MAX_IDENTIFIER) -> str:
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


def _exact_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(parsed, 1.0)) if math.isfinite(parsed) else 0.0


def _metadata(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name, default)
    except Exception:
        return default


@dataclass(frozen=True)
class CorpusEvidence:
    evidence_id: str
    doc_id: str
    text: str
    score: float
    dense_score: float
    sparse_score: float
    generation_sequence: int
    profile_fingerprint: str
    source_kind: str
    page_number: int | None = None
    section: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        if not isinstance(self.text, str) or not self.text or len(self.text) > _MAX_TEXT:
            raise ValueError("evidence text is invalid.")
        object.__setattr__(self, "score", _score(self.score))
        object.__setattr__(self, "dense_score", _score(self.dense_score))
        object.__setattr__(self, "sparse_score", _score(self.sparse_score))
        if (
            isinstance(self.generation_sequence, bool)
            or not isinstance(self.generation_sequence, int)
            or self.generation_sequence <= 0
        ):
            raise ValueError("generation_sequence must be positive.")
        object.__setattr__(
            self,
            "profile_fingerprint",
            _identifier(self.profile_fingerprint, "profile_fingerprint", 64),
        )
        if self.source_kind not in {"dense_chunk", "sparse_field"}:
            raise ValueError("source_kind is invalid.")
        if self.page_number is not None and (
            isinstance(self.page_number, bool)
            or not isinstance(self.page_number, int)
            or not 1 <= self.page_number <= 1_000_000
        ):
            raise ValueError("page_number is invalid.")
        if self.section is not None:
            _identifier(self.section, "section", 1_000)
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


def _bounded_dense(values: Any, maximum: int) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (str, bytes, bytearray)):
        raise RuntimeError("Dense backend returned an invalid result collection.")
    try:
        return list(itertools.islice(iter(values), maximum + 1))[:maximum]
    except Exception as exc:
        raise RuntimeError("Dense backend returned an invalid result collection.") from exc


def retrieve_corpus_evidence(
    query: str,
    *,
    owner_id: str,
    rag: Any,
    sparse: SparseIndex,
    generations: GenerationStore,
    doc_id: str | None = None,
    mode: str = "hybrid",
    top_k: int = 5,
    dense_pool: int = 30,
    sparse_pool: int = 30,
    dense_weight: float = 0.55,
    sparse_weight: float = 0.45,
    diversity_lambda: float = 0.82,
) -> tuple[CorpusEvidence, ...]:
    """Retrieve independently, fuse by document, validate generation, select evidence."""

    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    bounded_query = query.strip()
    if not bounded_query or len(bounded_query) > 20_000 or any(
        ord(ch) < 32 and ch not in "\t\r\n" for ch in bounded_query
    ):
        raise ValueError("query is empty, invalid, or too long.")
    owner = normalize_owner_id(owner_id)
    document_id = _identifier(doc_id, "doc_id", 200) if doc_id is not None else None
    if mode not in {"dense", "sparse", "hybrid"}:
        raise ValueError("mode must be dense, sparse, or hybrid.")
    requested = _exact_int(top_k, "top_k", 1, _MAX_RESULTS)
    dense_limit = _exact_int(dense_pool, "dense_pool", 1, _MAX_POOL)
    sparse_limit = _exact_int(sparse_pool, "sparse_pool", 1, _MAX_POOL)
    dense_w = _unit(dense_weight, "dense_weight")
    sparse_w = _unit(sparse_weight, "sparse_weight")
    diversity = _unit(diversity_lambda, "diversity_lambda")

    dense_chunks: list[Any] = []
    if mode in {"dense", "hybrid"}:
        dense_chunks = _bounded_dense(
            rag.query(
                bounded_query,
                n_results=dense_limit,
                owner_id=owner,
                doc_id=document_id,
            ),
            dense_limit,
        )
    sparse_hits: list[SparseSearchHit] = []
    if mode in {"sparse", "hybrid"}:
        sparse_hits = sparse.search(
            bounded_query,
            owner_id=owner,
            limit=sparse_limit,
            doc_id=document_id,
        )

    dense_by_doc: defaultdict[str, list[tuple[Any, float, Mapping[str, Any]]]] = defaultdict(list)
    dense_scores: dict[str, float] = {}
    for chunk in dense_chunks:
        metadata = _metadata(_attr(chunk, "metadata", {}))
        try:
            metadata_owner = metadata.get("owner_id")
            raw_doc_id = metadata.get("doc_id")
        except Exception:
            continue
        if metadata_owner != owner or not isinstance(raw_doc_id, str):
            continue
        try:
            current_doc = _identifier(raw_doc_id, "dense doc_id", 200)
        except ValueError:
            continue
        if document_id is not None and current_doc != document_id:
            continue
        dense_score = _score(_attr(chunk, "score", 0.0))
        dense_by_doc[current_doc].append((chunk, dense_score, metadata))
        dense_scores[current_doc] = max(dense_scores.get(current_doc, 0.0), dense_score)

    sparse_by_doc: dict[str, SparseSearchHit] = {}
    sparse_scores: dict[str, float] = {}
    for hit in sparse_hits[:sparse_limit]:
        if not isinstance(hit, SparseSearchHit):
            continue
        if document_id is not None and hit.doc_id != document_id:
            continue
        sparse_by_doc[hit.doc_id] = hit
        sparse_scores[hit.doc_id] = max(sparse_scores.get(hit.doc_id, 0.0), _score(hit.score))

    components: dict[str, Mapping[str, float]] = {}
    weights: dict[str, float] = {}
    if mode in {"dense", "hybrid"}:
        components["dense"] = dense_scores
        weights["dense"] = dense_w
    if mode in {"sparse", "hybrid"}:
        components["sparse"] = sparse_scores
        weights["sparse"] = sparse_w
    fused = weighted_fusion(components, weights=weights)
    ordered_docs = sorted(fused, key=lambda value: (fused[value], value), reverse=True)[:_MAX_POOL]

    candidates: list[tuple[RetrievalCandidate, float]] = []
    evidence_rows: dict[str, CorpusEvidence] = {}
    for current_doc in ordered_docs:
        record = generations.current(owner_id=owner, doc_id=current_doc)
        if record is None or record.state not in {"active", "restored"}:
            continue
        hit = sparse_by_doc.get(current_doc)
        if hit is not None and (
            hit.generation != record.sparse_generation
            or hit.profile_fingerprint != record.profile_fingerprint
        ):
            continue
        valid_dense: list[tuple[Any, float, Mapping[str, Any]]] = []
        for chunk, dense_score, metadata in dense_by_doc.get(current_doc, ()):
            try:
                fingerprint = metadata.get("embedding_profile_fingerprint")
                content_hash = metadata.get("content_sha256")
            except Exception:
                continue
            if (
                fingerprint != record.profile_fingerprint
                or content_hash != record.content_sha256
            ):
                continue
            valid_dense.append((chunk, dense_score, metadata))
        if mode == "dense" and not valid_dense:
            continue
        if mode == "sparse" and hit is None:
            continue
        if mode == "hybrid" and not valid_dense and hit is None:
            continue

        document_score = _score(fused.get(current_doc, 0.0))
        for chunk, dense_score, metadata in valid_dense[:_MAX_FIELDS_PER_DOCUMENT]:
            raw_id = _attr(chunk, "id", "")
            raw_text = _attr(chunk, "text", "")
            if not isinstance(raw_text, str) or not raw_text.strip():
                continue
            try:
                evidence_id = _identifier(raw_id, "chunk_id")
            except ValueError:
                continue
            text = raw_text.strip()[:_MAX_TEXT]
            page = metadata.get("page_number")
            page_number = page if isinstance(page, int) and not isinstance(page, bool) and 1 <= page <= 1_000_000 else None
            section = metadata.get("section_title")
            section_value = section[:1_000] if isinstance(section, str) and section.strip() else None
            local_score = _score(0.7 * document_score + 0.3 * dense_score)
            row = CorpusEvidence(
                evidence_id=evidence_id,
                doc_id=current_doc,
                text=text,
                score=local_score,
                dense_score=dense_score,
                sparse_score=sparse_scores.get(current_doc, 0.0),
                generation_sequence=record.sequence,
                profile_fingerprint=record.profile_fingerprint,
                source_kind="dense_chunk",
                page_number=page_number,
                section=section_value,
                metadata={"document_score": document_score},
            )
            evidence_rows[evidence_id] = row
            candidates.append((RetrievalCandidate(evidence_id, text, current_doc, local_score), local_score))

        if hit is not None:
            snapshot = sparse.snapshot_document(owner_id=owner, doc_id=current_doc)
            if snapshot is None or snapshot.generation != record.sparse_generation:
                continue
            fields = {field.field_id: field for field in snapshot.fields}
            for match in hit.matches[:_MAX_FIELDS_PER_DOCUMENT]:
                field = fields.get(match.field_id)
                if field is None:
                    continue
                evidence_id = f"sparse:{current_doc}:{field.field_id}"
                local_score = _score(0.7 * document_score + 0.3 * hit.score)
                row = CorpusEvidence(
                    evidence_id=evidence_id,
                    doc_id=current_doc,
                    text=field.text[:_MAX_TEXT],
                    score=local_score,
                    dense_score=dense_scores.get(current_doc, 0.0),
                    sparse_score=hit.score,
                    generation_sequence=record.sequence,
                    profile_fingerprint=record.profile_fingerprint,
                    source_kind="sparse_field",
                    page_number=field.page_number,
                    section=field.section,
                    metadata={
                        "document_score": document_score,
                        "field_type": field.field_type,
                        "term_frequencies": dict(match.term_frequencies),
                        "positions": {key: tuple(value) for key, value in match.positions.items()},
                    },
                )
                evidence_rows[evidence_id] = row
                candidates.append((RetrievalCandidate(evidence_id, row.text, current_doc, local_score), local_score))

    selected = mmr_select(
        candidates,
        top_k=requested,
        diversity_lambda=diversity,
        max_per_source=max(1, requested),
    )
    return tuple(evidence_rows[candidate.candidate_id] for candidate, _score_value in selected)


__all__ = ["CorpusEvidence", "retrieve_corpus_evidence"]
