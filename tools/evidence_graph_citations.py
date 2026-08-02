"""Convert authoritative GraphRAG evidence into the canonical Citation model."""

from __future__ import annotations

import hashlib
import json
import math
import operator
from collections.abc import Iterable
from typing import Any

from tools.models import Citation

_MAX_CITATIONS = 50


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("evidence score must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("evidence score must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("evidence score must be finite and non-negative.")
    return result


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    cleaned = value.strip()
    if len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid or too long.")
    if not cleaned and not allow_empty:
        raise ValueError(f"{label} is required.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _terms(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("matched_terms must be an iterable.")
    try:
        result = tuple(sorted(set(values)))
    except Exception as exc:
        raise ValueError("matched_terms is not safely iterable.") from exc
    if len(result) > 256 or any(
        not isinstance(value, str) or not value or len(value) > 200
        for value in result
    ):
        raise ValueError("matched_terms is invalid or too large.")
    return result


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _lineage(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("lineage_step_digests must be an iterable.")
    try:
        result = tuple(values)
    except Exception as exc:
        raise ValueError("lineage_step_digests is not safely iterable.") from exc
    if len(result) > 100:
        raise ValueError("lineage_step_digests exceeds the item limit.")
    return tuple(_digest(value, "lineage_step_digest") for value in result)


def graph_evidence_to_citations(
    selection: Any,
    *,
    start_index: int = 1,
    max_citations: int = _MAX_CITATIONS,
    allowed_origins: Iterable[str] | None = None,
) -> list[Citation]:
    """Create canonical server-owned citations from one authority-checked selection."""

    first_label = _integer(start_index, "start_index", 1, 100_000)
    limit = _integer(max_citations, "max_citations", 1, _MAX_CITATIONS)
    owner_id = _text(getattr(selection, "owner_id", None), "owner_id", 200)
    graph_set_key = _text(
        getattr(selection, "graph_set_key", None), "graph_set_key", 500
    )
    graph_set_id = _digest(getattr(selection, "graph_set_id", None), "graph_set_id")
    graph_set_digest = _digest(
        getattr(selection, "graph_set_digest", None), "graph_set_digest"
    )
    authority_digest = _digest(
        getattr(selection, "authority_digest", None), "authority_digest"
    )
    query_digest = _digest(getattr(selection, "query_digest", None), "query_digest")
    selection_digest = _digest(
        getattr(selection, "selection_digest", None), "selection_digest"
    )
    if getattr(selection, "citation_conversion_performed", False) is not False:
        raise ValueError("selection must not already contain converted citations.")
    if getattr(selection, "answer_generated", False) is not False:
        raise ValueError("selection must not contain a generated answer.")
    abstained = getattr(selection, "abstained", None)
    if not isinstance(abstained, bool):
        raise ValueError("selection.abstained must be boolean.")
    items = getattr(selection, "items", None)
    if isinstance(items, (str, bytes, bytearray)):
        raise ValueError("selection.items must be an iterable.")
    try:
        values = tuple(items)
    except Exception as exc:
        raise ValueError("selection.items is not safely iterable.") from exc
    if abstained:
        if values:
            raise ValueError("an abstained selection may not contain evidence.")
        return []
    if not values:
        raise ValueError("a non-abstained selection requires evidence.")

    allowed = None
    if allowed_origins is not None:
        if isinstance(allowed_origins, (str, bytes, bytearray)):
            raise ValueError("allowed_origins must be an iterable.")
        allowed = frozenset(allowed_origins)
        if not allowed or not allowed <= {
            "lexical",
            "within_document",
            "cross_document",
        }:
            raise ValueError("allowed_origins contains unsupported values.")

    citations: list[Citation] = []
    seen: set[tuple[str, int, str]] = set()
    for item in values:
        origin = _text(getattr(item, "origin", None), "origin", 50)
        if origin not in {"lexical", "within_document", "cross_document"}:
            raise ValueError("evidence origin is unsupported.")
        if allowed is not None and origin not in allowed:
            continue
        item_owner_id = _text(
            getattr(item, "owner_id", None), "item owner_id", 200
        )
        if item_owner_id != owner_id:
            raise ValueError("evidence item escaped selection owner scope.")
        doc_id = _text(getattr(item, "doc_id", None), "doc_id", 200)
        generation = _integer(
            getattr(item, "generation", None),
            "generation",
            1,
            2**63 - 1,
        )
        node_id = _digest(getattr(item, "node_id", None), "node_id")
        identity = (doc_id, generation, node_id)
        if identity in seen:
            continue
        seen.add(identity)
        if len(citations) >= limit:
            break
        item_graph_digest = _digest(
            getattr(item, "graph_digest", None), "item graph_digest"
        )
        node_type = _text(getattr(item, "node_type", None), "node_type", 50)
        title = _text(getattr(item, "label", None), "label", 2_000)[:500]
        evidence_text = _text(
            getattr(item, "text", None),
            "evidence text",
            50_000_000,
            allow_empty=True,
        )
        page_number = getattr(item, "page_number", None)
        if page_number is not None:
            page_number = _integer(page_number, "page_number", 1, 1_000_000)
        section = getattr(item, "section", None)
        if section is not None:
            section = _text(section, "section", 2_000)
        score = _score(getattr(item, "score", None))
        provenance_digest = _digest(
            getattr(item, "provenance_digest", None), "provenance_digest"
        )
        evidence_digest = _digest(
            getattr(item, "evidence_digest", None), "evidence_digest"
        )
        text_sha256 = _digest(
            getattr(item, "text_sha256", None), "text_sha256"
        )
        matched_terms = _terms(getattr(item, "matched_terms", ()))
        lineage = _lineage(getattr(item, "lineage_step_digests", ()))
        citations.append(
            Citation(
                label=f"[{first_label + len(citations)}]",
                title=title,
                url=f"local://{doc_id}",
                source_type="uploaded_document",
                snippet=evidence_text[:4_000] or None,
                quote=evidence_text[:4_000] or None,
                source_id=evidence_digest,
                doc_id=doc_id,
                chunk_id=node_id,
                page_number=page_number,
                metadata={
                    "retrieval_strategy": "graph",
                    "graph_set_key": graph_set_key,
                    "graph_set_id": graph_set_id,
                    "graph_set_digest": graph_set_digest,
                    "graph_authority_digest": authority_digest,
                    "graph_query_digest": query_digest,
                    "graph_selection_digest": selection_digest,
                    "graph_generation": generation,
                    "graph_member_digest": item_graph_digest,
                    "graph_node_id": node_id,
                    "graph_node_type": node_type,
                    "graph_node_provenance_digest": provenance_digest,
                    "graph_evidence_digest": evidence_digest,
                    "graph_text_sha256": text_sha256,
                    "graph_origin": origin,
                    "graph_score": round(score, 8),
                    "graph_matched_term_count": len(matched_terms),
                    "graph_matched_terms_digest": _canonical_digest(
                        list(matched_terms)
                    ),
                    "graph_lineage_step_digests": list(lineage),
                    "section_title": section,
                },
            )
        )
    return citations


__all__ = ["graph_evidence_to_citations"]
