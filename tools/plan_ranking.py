"""Dependency-free learned ranking for adaptive retrieval attempts.

The ranker is intentionally small and auditable: a versioned linear utility model can
be trained from pairwise preferences, while callers retain a deterministic heuristic
fallback when no validated learned model is available.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from tools.adaptive_retrieval import QueryAnalysis, RetrievalAttempt

_MAX_EXAMPLES = 100_000
_MAX_EPOCHS = 10_000
_FEATURES = (
    "bias",
    "complexity",
    "cost",
    "dense",
    "sparse",
    "hybrid",
    "multi_query",
    "hyde",
    "reranker",
    "exact_sparse_match",
    "complex_hybrid_match",
    "method_rerank_match",
)


def _finite(value: Any, label: str, minimum: float = -1_000_000.0, maximum: float = 1_000_000.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} is outside its allowed range.")
    return parsed


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _version(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("version must be a string.")
    text = value.strip()
    if not text or len(text) > 200 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError("version is invalid.")
    return text


def plan_features(analysis: QueryAnalysis, attempt: RetrievalAttempt) -> Mapping[str, float]:
    if not isinstance(analysis, QueryAnalysis) or not isinstance(attempt, RetrievalAttempt):
        raise ValueError("analysis and attempt must use adaptive retrieval dataclasses.")
    return {
        "bias": 1.0,
        "complexity": analysis.complexity,
        "cost": min(attempt.estimated_cost / 100.0, 1.0),
        "dense": float(attempt.mode == "dense"),
        "sparse": float(attempt.mode == "corpus-sparse"),
        "hybrid": float(attempt.mode == "corpus-hybrid"),
        "multi_query": float(attempt.use_multi_query),
        "hyde": float(attempt.use_hyde),
        "reranker": float(attempt.reranker != "none"),
        "exact_sparse_match": float(analysis.exact_identifier and attempt.mode == "corpus-sparse"),
        "complex_hybrid_match": float(analysis.complexity >= 0.45 and attempt.mode == "corpus-hybrid"),
        "method_rerank_match": float(analysis.methodological and attempt.reranker != "none"),
    }


@dataclass(frozen=True)
class RankedAttempt:
    attempt: RetrievalAttempt
    score: float
    rank: int


@dataclass(frozen=True)
class LinearPlanRanker:
    version: str
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _version(self.version))
        if not isinstance(self.weights, Mapping) or len(self.weights) > len(_FEATURES):
            raise ValueError("weights must be a bounded mapping.")
        normalized: dict[str, float] = {name: 0.0 for name in _FEATURES}
        for name, value in self.weights.items():
            if name not in normalized:
                raise ValueError(f"unsupported plan-ranking feature: {name!r}.")
            normalized[name] = _finite(value, f"weight {name}")
        object.__setattr__(self, "weights", normalized)

    def score(self, analysis: QueryAnalysis, attempt: RetrievalAttempt) -> float:
        features = plan_features(analysis, attempt)
        return sum(self.weights[name] * features[name] for name in _FEATURES)

    def rank(self, analysis: QueryAnalysis, attempts: Sequence[RetrievalAttempt]) -> tuple[RankedAttempt, ...]:
        if isinstance(attempts, (str, bytes, bytearray)) or not attempts or len(attempts) > 100:
            raise ValueError("attempts must be a bounded non-empty sequence.")
        if any(not isinstance(attempt, RetrievalAttempt) for attempt in attempts):
            raise ValueError("attempts contains an invalid value.")
        rows = sorted(
            ((attempt, self.score(analysis, attempt), index) for index, attempt in enumerate(attempts)),
            key=lambda row: (-row[1], row[0].estimated_cost, row[2]),
        )
        return tuple(RankedAttempt(attempt, score, rank) for rank, (attempt, score, _index) in enumerate(rows, start=1))


@dataclass(frozen=True)
class PlanTrainingExample:
    analysis: QueryAnalysis
    preferred: RetrievalAttempt
    rejected: RetrievalAttempt
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.analysis, QueryAnalysis):
            raise ValueError("analysis must be QueryAnalysis.")
        if not isinstance(self.preferred, RetrievalAttempt) or not isinstance(self.rejected, RetrievalAttempt):
            raise ValueError("preferred/rejected must be RetrievalAttempt.")
        object.__setattr__(self, "weight", _finite(self.weight, "example weight", 0.0, 1_000.0))
        if self.weight <= 0.0:
            raise ValueError("example weight must be positive.")


def fit_pairwise_plan_ranker(
    examples: Iterable[PlanTrainingExample],
    *,
    version: str,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    initial_weights: Mapping[str, float] | None = None,
) -> LinearPlanRanker:
    """Fit pairwise logistic preferences with deterministic full-batch ordering."""

    if isinstance(examples, (str, bytes, bytearray)):
        raise ValueError("examples must be an iterable.")
    rows: list[PlanTrainingExample] = []
    try:
        iterator = iter(examples)
    except Exception as exc:
        raise ValueError("examples must be safely iterable.") from exc
    for example in iterator:
        if len(rows) >= _MAX_EXAMPLES:
            raise ValueError("examples exceeds the training limit.")
        if not isinstance(example, PlanTrainingExample):
            raise ValueError("examples contains an invalid value.")
        rows.append(example)
    if not rows:
        raise ValueError("at least one training example is required.")
    epoch_count = _integer(epochs, "epochs", 1, _MAX_EPOCHS)
    rate = _finite(learning_rate, "learning_rate", 1e-9, 10.0)
    regularization = _finite(l2, "l2", 0.0, 1_000.0)
    initial = LinearPlanRanker(version, initial_weights or {})
    weights = dict(initial.weights)
    for _epoch in range(epoch_count):
        for example in rows:
            preferred = plan_features(example.analysis, example.preferred)
            rejected = plan_features(example.analysis, example.rejected)
            delta = {name: preferred[name] - rejected[name] for name in _FEATURES}
            margin = sum(weights[name] * delta[name] for name in _FEATURES)
            if margin >= 0.0:
                exp_value = math.exp(-min(margin, 700.0))
                logistic = exp_value / (1.0 + exp_value)
            else:
                exp_value = math.exp(max(margin, -700.0))
                logistic = 1.0 / (1.0 + exp_value)
            for name in _FEATURES:
                gradient = example.weight * logistic * delta[name] - regularization * weights[name]
                weights[name] = max(-1_000_000.0, min(weights[name] + rate * gradient, 1_000_000.0))
    return LinearPlanRanker(version, weights)


def heuristic_rank_attempts(analysis: QueryAnalysis, attempts: Sequence[RetrievalAttempt]) -> tuple[RankedAttempt, ...]:
    """Deterministic fallback that rewards query/route fit and penalizes cost."""

    fallback = LinearPlanRanker(
        "heuristic-v1",
        {
            "cost": -0.35,
            "exact_sparse_match": 1.0,
            "complex_hybrid_match": 0.65,
            "method_rerank_match": 0.35,
            "multi_query": 0.05,
        },
    )
    return fallback.rank(analysis, attempts)


__all__ = [
    "LinearPlanRanker",
    "PlanTrainingExample",
    "RankedAttempt",
    "fit_pairwise_plan_ranker",
    "heuristic_rank_attempts",
    "plan_features",
]
