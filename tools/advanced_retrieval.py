"""Calibrated hybrid pipeline spanning dense, lexical and advanced retrieval signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from tools.hybrid_retrieval import (
    RankedCandidate,
    RetrievalCandidate,
    RerankCallable,
    bm25_scores,
    mmr_select,
)
from tools.retrieval_architectures import (
    LateInteractionScorer,
    ScoreCalibration,
    SparseExpansionScorer,
    calibrated_weighted_fusion,
    colbert_maxsim,
    splade_sparse_similarity,
)

_MAX_CANDIDATES = 500


@dataclass(frozen=True)
class AdvancedRetrievalConfig:
    dense_weight: float = 0.35
    lexical_weight: float = 0.25
    splade_weight: float = 0.15
    late_interaction_weight: float = 0.15
    reranker_weight: float = 0.10
    calibrations: Mapping[str, ScoreCalibration] = field(default_factory=dict)
    diversity_lambda: float = 0.82
    max_per_source: int = 3

    def __post_init__(self) -> None:
        weights = (
            self.dense_weight,
            self.lexical_weight,
            self.splade_weight,
            self.late_interaction_weight,
            self.reranker_weight,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1000.0
            for value in weights
        ):
            raise ValueError("advanced retrieval weights must be non-negative and bounded.")
        if sum(float(value) for value in weights) <= 0.0:
            raise ValueError("at least one advanced retrieval weight must be positive.")
        if not isinstance(self.calibrations, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, ScoreCalibration)
            for key, value in self.calibrations.items()
        ):
            raise ValueError("calibrations must map component names to ScoreCalibration.")
        if (
            isinstance(self.diversity_lambda, bool)
            or not isinstance(self.diversity_lambda, (int, float))
            or not 0.0 <= float(self.diversity_lambda) <= 1.0
        ):
            raise ValueError("diversity_lambda must be between 0 and 1.")
        if (
            isinstance(self.max_per_source, bool)
            or not isinstance(self.max_per_source, int)
            or not 1 <= self.max_per_source <= _MAX_CANDIDATES
        ):
            raise ValueError("max_per_source is invalid.")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "dense": float(self.dense_weight),
            "lexical": float(self.lexical_weight),
            "splade": float(self.splade_weight),
            "late_interaction": float(self.late_interaction_weight),
            "reranker": float(self.reranker_weight),
        }


def _bounded_candidates(
    candidates: Sequence[RetrievalCandidate],
) -> tuple[RetrievalCandidate, ...]:
    if isinstance(candidates, (str, bytes, bytearray)):
        raise ValueError("candidates must be a sequence of RetrievalCandidate values.")
    result: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for item in candidates[:_MAX_CANDIDATES]:
        if not isinstance(item, RetrievalCandidate):
            raise ValueError("every candidate must be RetrievalCandidate.")
        if item.candidate_id in seen:
            continue
        seen.add(item.candidate_id)
        result.append(item)
    return tuple(result)


def _optional_splade_scores(
    query: str,
    candidates: tuple[RetrievalCandidate, ...],
    scorer: SparseExpansionScorer | None,
) -> dict[str, float]:
    if scorer is None:
        return {}
    if not callable(getattr(scorer, "query_weights", None)) or not callable(
        getattr(scorer, "document_weights", None)
    ):
        raise ValueError("sparse scorer does not implement the required contract.")
    try:
        query_weights = scorer.query_weights(query)
        return {
            item.candidate_id: splade_sparse_similarity(
                query_weights,
                scorer.document_weights(item.text),
            )
            for item in candidates
        }
    except Exception:
        return {}


def _optional_late_scores(
    query: str,
    candidates: tuple[RetrievalCandidate, ...],
    scorer: LateInteractionScorer | None,
) -> dict[str, float]:
    if scorer is None:
        return {}
    if not callable(getattr(scorer, "query_vectors", None)) or not callable(
        getattr(scorer, "document_vectors", None)
    ):
        raise ValueError("late-interaction scorer does not implement the required contract.")
    try:
        query_vectors = scorer.query_vectors(query)
        return {
            item.candidate_id: colbert_maxsim(
                query_vectors,
                scorer.document_vectors(item.text),
            )
            for item in candidates
        }
    except Exception:
        return {}


def _optional_rerank_scores(
    query: str,
    candidates: tuple[RetrievalCandidate, ...],
    reranker: RerankCallable | None,
) -> dict[str, float]:
    if reranker is None:
        return {}
    try:
        value = reranker(query, candidates)
    except Exception:
        return {}
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    candidate_ids = {item.candidate_id for item in candidates}
    for candidate_id, raw in value.items():
        if candidate_id not in candidate_ids or isinstance(raw, bool):
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0.0 <= score <= 1.0:
            result[candidate_id] = score
    return result


def rank_advanced_candidates(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    *,
    top_k: int = 5,
    config: AdvancedRetrievalConfig | None = None,
    sparse_scorer: SparseExpansionScorer | None = None,
    late_interaction_scorer: LateInteractionScorer | None = None,
    reranker: RerankCallable | None = None,
) -> list[RankedCandidate]:
    """Rank one bounded owner-scoped candidate pool using available architectures."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= _MAX_CANDIDATES:
        raise ValueError("top_k must be between 1 and 500.")
    selected_config = config or AdvancedRetrievalConfig()
    if not isinstance(selected_config, AdvancedRetrievalConfig):
        raise ValueError("config must be AdvancedRetrievalConfig.")
    values = _bounded_candidates(candidates)
    if not values:
        return []

    dense = {item.candidate_id: item.dense_score for item in values}
    lexical = bm25_scores(query, values)
    splade = _optional_splade_scores(query, values, sparse_scorer)
    late = _optional_late_scores(query, values, late_interaction_scorer)
    rerank = _optional_rerank_scores(query, values, reranker)
    components: dict[str, Mapping[str, float]] = {
        "dense": dense,
        "lexical": lexical,
    }
    if splade:
        components["splade"] = splade
    if late:
        components["late_interaction"] = late
    if rerank:
        components["reranker"] = rerank
    fused = calibrated_weighted_fusion(
        components,
        weights=selected_config.weights,
        calibrations=selected_config.calibrations,
    )
    ordered = sorted(
        values,
        key=lambda item: (
            fused.get(item.candidate_id, 0.0),
            item.dense_score,
            item.candidate_id,
        ),
        reverse=True,
    )
    selected = mmr_select(
        [(item, fused.get(item.candidate_id, 0.0)) for item in ordered],
        top_k=min(top_k, len(ordered)),
        diversity_lambda=float(selected_config.diversity_lambda),
        max_per_source=selected_config.max_per_source,
    )
    return [
        RankedCandidate(
            candidate=item,
            rank=rank,
            score=score,
            components={
                "dense": dense.get(item.candidate_id, 0.0),
                "lexical": lexical.get(item.candidate_id, 0.0),
                "splade": splade.get(item.candidate_id, 0.0),
                "late_interaction": late.get(item.candidate_id, 0.0),
                "reranker": rerank.get(item.candidate_id, 0.0),
            },
        )
        for rank, (item, score) in enumerate(selected, start=1)
    ]


__all__ = ["AdvancedRetrievalConfig", "rank_advanced_candidates"]
