"""Deterministic lexical hard-negative mining for retrieval experiments."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_MAX_DOCUMENTS = 1_000_000
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid.")
    return result


def _tokens(text: Any, label: str) -> frozenset[str]:
    if not isinstance(text, str) or not text.strip() or len(text) > 5_000_000:
        raise ValueError(f"{label} is invalid.")
    return frozenset(token.casefold() for token in _TOKEN_RE.findall(text) if token)


@dataclass(frozen=True)
class HardNegative:
    document_id: str
    lexical_overlap: float
    shared_terms: int
    rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        if isinstance(self.lexical_overlap, bool):
            raise ValueError("lexical_overlap must be between 0 and 1.")
        score = float(self.lexical_overlap)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("lexical_overlap must be between 0 and 1.")
        object.__setattr__(self, "lexical_overlap", score)
        if isinstance(self.shared_terms, bool) or not isinstance(self.shared_terms, int) or self.shared_terms < 0:
            raise ValueError("shared_terms must be a non-negative integer.")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer.")


def mine_lexical_hard_negatives(
    *,
    query: str,
    documents: Mapping[str, str],
    relevant_ids: Sequence[str],
    limit: int = 20,
    minimum_overlap: float = 0.0,
) -> tuple[HardNegative, ...]:
    """Mine non-relevant documents with maximum query-token Jaccard overlap."""

    query_terms = _tokens(query, "query")
    if not isinstance(documents, Mapping) or len(documents) > _MAX_DOCUMENTS:
        raise ValueError("documents must be a bounded mapping.")
    if isinstance(relevant_ids, (str, bytes, bytearray)):
        raise ValueError("relevant_ids must be a sequence.")
    relevant = {_identifier(value, "relevant_id") for value in relevant_ids}
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
        raise ValueError("limit must be between 1 and 10000.")
    if isinstance(minimum_overlap, bool):
        raise ValueError("minimum_overlap must be between 0 and 1.")
    threshold = float(minimum_overlap)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_overlap must be between 0 and 1.")

    scored: list[tuple[float, int, str]] = []
    for raw_id, text in documents.items():
        document_id = _identifier(raw_id, "document_id")
        if document_id in relevant:
            continue
        document_terms = _tokens(text, "document text")
        shared = len(query_terms & document_terms)
        union = len(query_terms | document_terms)
        overlap = shared / union if union else 0.0
        if overlap >= threshold:
            scored.append((overlap, shared, document_id))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return tuple(
        HardNegative(document_id, overlap, shared, rank)
        for rank, (overlap, shared, document_id) in enumerate(scored[:limit], start=1)
    )


__all__ = ["HardNegative", "mine_lexical_hard_negatives"]
