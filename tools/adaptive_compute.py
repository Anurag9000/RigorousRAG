"""Deterministic adaptive-compute budgeting for RAG requests."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DifficultySignals:
    retrieval_uncertainty: float = 0.0
    decomposition_complexity: float = 0.0
    contradiction_risk: float = 0.0
    freshness_risk: float = 0.0
    multimodal_need: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("retrieval_uncertainty", self.retrieval_uncertainty),
            ("decomposition_complexity", self.decomposition_complexity),
            ("contradiction_risk", self.contradiction_risk),
            ("freshness_risk", self.freshness_risk),
            ("multimodal_need", self.multimodal_need),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")

    def difficulty(self) -> float:
        return min(
            1.0,
            0.30 * self.retrieval_uncertainty
            + 0.20 * self.decomposition_complexity
            + 0.20 * self.contradiction_risk
            + 0.15 * self.freshness_risk
            + 0.15 * self.multimodal_need,
        )


@dataclass(frozen=True)
class ComputeCaps:
    max_retrieval_k: int = 50
    max_rerank_k: int = 30
    max_hops: int = 6
    max_query_expansions: int = 6
    max_generation_tokens: int = 4000

    def __post_init__(self) -> None:
        for name, value in (
            ("max_retrieval_k", self.max_retrieval_k),
            ("max_rerank_k", self.max_rerank_k),
            ("max_hops", self.max_hops),
            ("max_query_expansions", self.max_query_expansions),
            ("max_generation_tokens", self.max_generation_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


@dataclass(frozen=True)
class ComputeBudget:
    retrieval_k: int
    rerank_k: int
    max_hops: int
    query_expansions: int
    generation_tokens: int
    difficulty: float


def allocate_compute(
    signals: DifficultySignals,
    *,
    caps: ComputeCaps = ComputeCaps(),
    minimum_generation_tokens: int = 256,
) -> ComputeBudget:
    """Scale bounded compute monotonically with estimated request difficulty."""

    if (
        isinstance(minimum_generation_tokens, bool)
        or not isinstance(minimum_generation_tokens, int)
        or minimum_generation_tokens <= 0
        or minimum_generation_tokens > caps.max_generation_tokens
    ):
        raise ValueError("minimum_generation_tokens must fit within generation caps.")
    difficulty = signals.difficulty()

    def scaled(minimum: int, maximum: int) -> int:
        if maximum <= minimum:
            return maximum
        return min(maximum, max(minimum, round(minimum + difficulty * (maximum - minimum))))

    retrieval_k = scaled(min(5, caps.max_retrieval_k), caps.max_retrieval_k)
    rerank_k = min(
        retrieval_k,
        scaled(min(3, caps.max_rerank_k), caps.max_rerank_k),
    )
    return ComputeBudget(
        retrieval_k=retrieval_k,
        rerank_k=rerank_k,
        max_hops=scaled(1, caps.max_hops),
        query_expansions=scaled(1, caps.max_query_expansions),
        generation_tokens=scaled(minimum_generation_tokens, caps.max_generation_tokens),
        difficulty=difficulty,
    )
