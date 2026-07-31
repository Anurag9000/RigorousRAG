"""Bounded hybrid retrieval, score fusion, and diversity selection.

The module is dependency-free so the same ranking primitives can be used by the
service, offline benchmarks, and constrained test environments.  It deliberately
operates on an already owner-scoped candidate pool; persistent lexical retrieval is
implemented separately so caller-visible tenant checks remain at the storage boundary.
"""

from __future__ import annotations

import math
import operator
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")
_MAX_CANDIDATES = 500
_MAX_TEXT_CHARS = 100_000
_MAX_TOKENS = 20_000


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


def _finite(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _unit(value: Any, label: str) -> float:
    parsed = _finite(value, default=math.nan)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def tokenize(value: Any) -> tuple[str, ...]:
    """Return bounded lower-case lexical tokens from one text value."""

    if not isinstance(value, str):
        raise ValueError("Retrieval text must be a string.")
    bounded = value[:_MAX_TEXT_CHARS]
    if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in bounded):
        raise ValueError("Retrieval text contains invalid control characters.")
    return tuple(token.lower() for token in _TOKEN_RE.findall(bounded)[:_MAX_TOKENS])


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    text: str
    source_id: str
    dense_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _identifier(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        if not isinstance(self.text, str):
            raise ValueError("text must be a string.")
        if len(self.text) > _MAX_TEXT_CHARS:
            raise ValueError("text exceeds the retrieval candidate limit.")
        object.__setattr__(self, "dense_score", max(0.0, min(_finite(self.dense_score), 1.0)))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping.")


@dataclass(frozen=True)
class RankedCandidate:
    candidate: RetrievalCandidate
    rank: int
    score: float
    components: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SparseDocument:
    document_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _bounded_candidates(values: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("candidates must be an iterable of RetrievalCandidate values.")
    result: list[RetrievalCandidate] = []
    seen: set[str] = set()
    try:
        iterator = iter(values)
    except Exception as exc:
        raise ValueError("candidates must be safely iterable.") from exc
    for value in iterator:
        if len(result) >= _MAX_CANDIDATES:
            break
        if not isinstance(value, RetrievalCandidate):
            continue
        if value.candidate_id in seen:
            continue
        seen.add(value.candidate_id)
        result.append(value)
    return result


def bm25_scores(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    """Compute normalized BM25 scores for one bounded in-memory candidate pool."""

    terms = tokenize(query)
    docs = _bounded_candidates(candidates)
    if not terms or not docs:
        return {item.candidate_id: 0.0 for item in docs}
    k1_value = _finite(k1, default=math.nan)
    b_value = _finite(b, default=math.nan)
    if not math.isfinite(k1_value) or not 0.0 < k1_value <= 10.0:
        raise ValueError("k1 must be finite and between 0 and 10.")
    if not math.isfinite(b_value) or not 0.0 <= b_value <= 1.0:
        raise ValueError("b must be finite and between 0 and 1.")

    token_rows: dict[str, tuple[str, ...]] = {
        item.candidate_id: tokenize(item.text) for item in docs
    }
    counts = {identifier: Counter(tokens) for identifier, tokens in token_rows.items()}
    document_frequency: Counter[str] = Counter()
    for row in counts.values():
        document_frequency.update(row.keys())
    average_length = sum(len(tokens) for tokens in token_rows.values()) / max(len(docs), 1)
    average_length = max(average_length, 1.0)
    query_counts = Counter(terms)
    raw: dict[str, float] = {}
    corpus_size = len(docs)
    for item in docs:
        row = counts[item.candidate_id]
        length = max(len(token_rows[item.candidate_id]), 1)
        total = 0.0
        for term, query_frequency in query_counts.items():
            frequency = row.get(term, 0)
            if frequency <= 0:
                continue
            df = document_frequency.get(term, 0)
            inverse_document_frequency = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            denominator = frequency + k1_value * (1.0 - b_value + b_value * length / average_length)
            total += query_frequency * inverse_document_frequency * frequency * (k1_value + 1.0) / denominator
        raw[item.candidate_id] = total
    maximum = max(raw.values(), default=0.0)
    if maximum <= 0.0:
        return {identifier: 0.0 for identifier in raw}
    return {identifier: score / maximum for identifier, score in raw.items()}


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    weights: Mapping[str, float] | None = None,
    rank_constant: int = 60,
) -> dict[str, float]:
    """Fuse named rank lists while deduplicating IDs inside every list."""

    constant = _exact_int(rank_constant, "rank_constant", 1, 10_000)
    if not isinstance(rankings, Mapping):
        raise ValueError("rankings must be a mapping.")
    totals: defaultdict[str, float] = defaultdict(float)
    for name, raw_ids in rankings.items():
        if not isinstance(name, str) or not name:
            continue
        weight = 1.0 if weights is None else _finite(weights.get(name, 0.0))
        if weight <= 0.0 or isinstance(raw_ids, (str, bytes, bytearray)):
            continue
        seen: set[str] = set()
        for rank, identifier in enumerate(raw_ids[:_MAX_CANDIDATES], start=1):
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                continue
            seen.add(identifier)
            totals[identifier] += weight / (constant + rank)
    maximum = max(totals.values(), default=0.0)
    return {identifier: value / maximum for identifier, value in totals.items()} if maximum else {}


def weighted_fusion(
    component_scores: Mapping[str, Mapping[str, float]],
    *,
    weights: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Fuse already normalized named score maps using a weighted average."""

    if not isinstance(component_scores, Mapping):
        raise ValueError("component_scores must be a mapping.")
    totals: defaultdict[str, float] = defaultdict(float)
    denominators: defaultdict[str, float] = defaultdict(float)
    for name, score_map in component_scores.items():
        if not isinstance(name, str) or not isinstance(score_map, Mapping):
            continue
        weight = 1.0 if weights is None else _finite(weights.get(name, 0.0))
        if weight <= 0.0:
            continue
        for identifier, raw_score in score_map.items():
            if not isinstance(identifier, str) or not identifier:
                continue
            score = max(0.0, min(_finite(raw_score), 1.0))
            totals[identifier] += weight * score
            denominators[identifier] += weight
    return {
        identifier: totals[identifier] / denominators[identifier]
        for identifier in totals
        if denominators[identifier] > 0.0
    }


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set, right_set = set(left), set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def mmr_select(
    ranked: Sequence[tuple[RetrievalCandidate, float]],
    *,
    top_k: int,
    diversity_lambda: float = 0.82,
    max_per_source: int = 3,
) -> list[tuple[RetrievalCandidate, float]]:
    """Select a relevance/diversity-balanced subset with a source cap."""

    limit = _exact_int(top_k, "top_k", 1, _MAX_CANDIDATES)
    relevance_weight = _unit(diversity_lambda, "diversity_lambda")
    per_source = _exact_int(max_per_source, "max_per_source", 1, _MAX_CANDIDATES)
    remaining = [(item, max(0.0, min(_finite(score), 1.0))) for item, score in ranked[:_MAX_CANDIDATES]]
    selected: list[tuple[RetrievalCandidate, float]] = []
    source_counts: Counter[str] = Counter()
    token_cache = {item.candidate_id: tokenize(item.text) for item, _ in remaining}
    while remaining and len(selected) < limit:
        best_index: int | None = None
        best_tuple: tuple[float, float, str] | None = None
        for index, (candidate, relevance) in enumerate(remaining):
            if source_counts[candidate.source_id] >= per_source:
                continue
            redundancy = max(
                (_jaccard(token_cache[candidate.candidate_id], token_cache[chosen.candidate_id]) for chosen, _ in selected),
                default=0.0,
            )
            mmr = relevance_weight * relevance - (1.0 - relevance_weight) * redundancy
            ordering = (mmr, relevance, candidate.candidate_id)
            if best_tuple is None or ordering > best_tuple:
                best_tuple, best_index = ordering, index
        if best_index is None:
            break
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        source_counts[chosen[0].source_id] += 1
    return selected


RerankCallable = Callable[[str, Sequence[RetrievalCandidate]], Mapping[str, float]]


def rank_candidates(
    query: str,
    candidates: Iterable[RetrievalCandidate],
    *,
    mode: str = "hybrid",
    top_k: int = 5,
    reranker: RerankCallable | None = None,
    diversity_lambda: float = 0.82,
    max_per_source: int = 3,
    dense_weight: float = 0.45,
    lexical_weight: float = 0.40,
    reranker_weight: float = 0.15,
) -> list[RankedCandidate]:
    """Rank an owner-scoped pool using dense, lexical, or fused hybrid scores."""

    if mode not in {"dense", "lexical", "hybrid"}:
        raise ValueError("mode must be dense, lexical, or hybrid.")
    limit = _exact_int(top_k, "top_k", 1, _MAX_CANDIDATES)
    values = _bounded_candidates(candidates)
    dense = {item.candidate_id: item.dense_score for item in values}
    lexical = bm25_scores(query, values) if mode != "dense" else {}
    component_scores: dict[str, Mapping[str, float]] = {}
    weights: dict[str, float] = {}
    if mode in {"dense", "hybrid"}:
        component_scores["dense"] = dense
        weights["dense"] = _finite(dense_weight)
    if mode in {"lexical", "hybrid"}:
        component_scores["lexical"] = lexical
        weights["lexical"] = _finite(lexical_weight)
    rerank_scores: Mapping[str, float] = {}
    if reranker is not None and values:
        try:
            raw = reranker(query, values)
            rerank_scores = raw if isinstance(raw, Mapping) else {}
        except Exception:
            rerank_scores = {}
        if rerank_scores:
            component_scores["reranker"] = rerank_scores
            weights["reranker"] = _finite(reranker_weight)
    fused = weighted_fusion(component_scores, weights=weights)
    ordered = sorted(values, key=lambda item: (fused.get(item.candidate_id, 0.0), item.dense_score, item.candidate_id), reverse=True)
    selected = mmr_select(
        [(item, fused.get(item.candidate_id, 0.0)) for item in ordered],
        top_k=min(limit, len(ordered)) if ordered else 1,
        diversity_lambda=diversity_lambda,
        max_per_source=max_per_source,
    ) if ordered else []
    result: list[RankedCandidate] = []
    for rank, (item, score) in enumerate(selected, start=1):
        components = {
            "dense": dense.get(item.candidate_id, 0.0),
            "lexical": lexical.get(item.candidate_id, 0.0),
            "reranker": max(0.0, min(_finite(rerank_scores.get(item.candidate_id, 0.0)), 1.0)),
        }
        result.append(RankedCandidate(item, rank, score, components))
    return result
