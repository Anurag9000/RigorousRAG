"""Bounded second-stage reranker interfaces with optional lazy model loading."""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import Any, Mapping, Protocol, Sequence

from tools.hybrid_retrieval import RetrievalCandidate, tokenize


class Reranker(Protocol):
    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> Mapping[str, float]: ...

    def __call__(self, query: str, candidates: Sequence[RetrievalCandidate]) -> Mapping[str, float]: ...


class NoOpReranker:
    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> Mapping[str, float]:
        del query
        return {item.candidate_id: item.dense_score for item in candidates}

    __call__ = score


class HeuristicReranker:
    """Dependency-free overlap/proximity reranker used as a safe baseline."""

    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> Mapping[str, float]:
        query_tokens = tokenize(query)
        query_counts = Counter(query_tokens)
        if not query_counts:
            return {item.candidate_id: 0.0 for item in candidates}
        scores: dict[str, float] = {}
        for item in candidates:
            tokens = tokenize(item.text)
            counts = Counter(tokens)
            coverage = sum(min(counts[token], count) for token, count in query_counts.items()) / sum(query_counts.values())
            positions = [index for index, token in enumerate(tokens) if token in query_counts]
            proximity = 0.0
            if positions:
                span = max(positions) - min(positions) + 1
                proximity = min(1.0, len(positions) / max(span, 1))
            phrase = 1.0 if " ".join(query_tokens) in " ".join(tokens) else 0.0
            scores[item.candidate_id] = max(0.0, min(0.65 * coverage + 0.20 * proximity + 0.15 * phrase, 1.0))
        return scores

    __call__ = score


class CrossEncoderReranker:
    """Optional sentence-transformers CrossEncoder loaded only on first use."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", *, max_candidates: int = 50) -> None:
        if not isinstance(model_name, str) or not model_name.strip() or len(model_name) > 300:
            raise ValueError("model_name is invalid.")
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or not 1 <= max_candidates <= 100:
            raise ValueError("max_candidates must be between 1 and 100.")
        self.model_name = model_name.strip()
        self.max_candidates = max_candidates
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder  # optional dependency

                self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, candidates: Sequence[RetrievalCandidate]) -> Mapping[str, float]:
        selected = list(candidates[: self.max_candidates])
        if not selected:
            return {}
        model = self._load()
        raw = model.predict([(query, item.text) for item in selected])
        values: list[float] = []
        for value in raw:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                number = 0.0
            values.append(number if math.isfinite(number) else 0.0)
        if not values:
            return {}
        low, high = min(values), max(values)
        if high <= low:
            normalized = [0.5 for _ in values]
        else:
            normalized = [(value - low) / (high - low) for value in values]
        return {item.candidate_id: normalized[index] for index, item in enumerate(selected)}

    __call__ = score


def build_reranker(name: str, *, model_name: str | None = None) -> Reranker:
    if not isinstance(name, str):
        raise ValueError("reranker name must be a string.")
    normalized = name.strip().lower()
    if normalized in {"", "none", "dense"}:
        return NoOpReranker()
    if normalized == "heuristic":
        return HeuristicReranker()
    if normalized in {"cross-encoder", "cross_encoder"}:
        return CrossEncoderReranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")
    raise ValueError("Unsupported reranker.")
