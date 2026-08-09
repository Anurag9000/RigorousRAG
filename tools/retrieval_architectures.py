"""Model-agnostic primitives for advanced retrieval architectures.

The functions in this module deliberately accept already-computed representations.
They therefore support SPLADE-style sparse expansion, ColBERT-style late interaction,
multi-vector models such as BGE-M3, and Matryoshka embeddings without importing or
downloading any model. Production adapters remain explicit trust boundaries.
"""

from __future__ import annotations

import itertools
import math
import operator
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

_MAX_COMPONENTS = 64
_MAX_TERMS = 100_000
_MAX_VECTORS = 512
_MAX_DIMENSIONS = 16_384
_EPSILON = 1e-6


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number.")
    return result


def _unit(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _positive(value: Any, label: str, maximum: float = 1000.0) -> float:
    result = _finite(value, label)
    if not 0.0 < result <= maximum:
        raise ValueError(f"{label} must be positive and bounded.")
    return result


def _exact_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _score_map(value: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if len(value) > _MAX_TERMS:
        raise ValueError(f"{label} exceeds the item limit.")
    result: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key or len(key) > 500:
            raise ValueError(f"{label} contains an invalid key.")
        result[key] = _unit(raw, f"{label} score")
    return result


def _sparse_weights(value: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    if len(value) > _MAX_TERMS:
        raise ValueError(f"{label} exceeds the term limit.")
    result: dict[str, float] = {}
    for term, raw in value.items():
        if not isinstance(term, str) or not term or len(term) > 500:
            raise ValueError(f"{label} contains an invalid term.")
        weight = _finite(raw, f"{label} weight")
        if weight < 0.0 or weight > 1_000_000.0:
            raise ValueError(f"{label} weights must be non-negative and bounded.")
        if weight > 0.0:
            result[term] = weight
    return result


def _vector(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric vector.")
    try:
        raw = list(itertools.islice(iter(value), _MAX_DIMENSIONS + 1))
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    if not raw or len(raw) > _MAX_DIMENSIONS:
        raise ValueError(f"{label} is empty or exceeds the dimension limit.")
    result = tuple(_finite(item, f"{label} value") for item in raw)
    if not any(item != 0.0 for item in result):
        raise ValueError(f"{label} may not be an all-zero vector.")
    return result


def _vectors(value: Any, label: str) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a vector sequence.")
    try:
        raw = list(itertools.islice(iter(value), _MAX_VECTORS + 1))
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    if not raw or len(raw) > _MAX_VECTORS:
        raise ValueError(f"{label} is empty or exceeds the vector limit.")
    result = tuple(_vector(item, f"{label} vector") for item in raw)
    dimensions = len(result[0])
    if any(len(item) != dimensions for item in result):
        raise ValueError(f"{label} vectors must have consistent dimensions.")
    return result


def calibrate_score(
    score: Any,
    *,
    temperature: float = 1.0,
    bias: float = 0.0,
) -> float:
    """Apply bounded logit temperature/bias calibration to one probability score."""

    probability = _unit(score, "score")
    selected_temperature = _positive(temperature, "temperature", 100.0)
    selected_bias = _finite(bias, "bias")
    if abs(selected_bias) > 100.0:
        raise ValueError("bias must be between -100 and 100.")
    clipped = min(max(probability, _EPSILON), 1.0 - _EPSILON)
    logit = math.log(clipped / (1.0 - clipped))
    calibrated_logit = logit / selected_temperature + selected_bias
    if calibrated_logit >= 0.0:
        exponential = math.exp(-min(calibrated_logit, 700.0))
        result = 1.0 / (1.0 + exponential)
    else:
        exponential = math.exp(max(calibrated_logit, -700.0))
        result = exponential / (1.0 + exponential)
    return max(0.0, min(result, 1.0))


@dataclass(frozen=True)
class ScoreCalibration:
    temperature: float = 1.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temperature",
            _positive(self.temperature, "temperature", 100.0),
        )
        selected_bias = _finite(self.bias, "bias")
        if abs(selected_bias) > 100.0:
            raise ValueError("bias must be between -100 and 100.")
        object.__setattr__(self, "bias", selected_bias)

    def apply(self, score: Any) -> float:
        return calibrate_score(
            score,
            temperature=self.temperature,
            bias=self.bias,
        )


def calibrated_weighted_fusion(
    components: Mapping[str, Mapping[str, Any]],
    *,
    weights: Mapping[str, Any] | None = None,
    calibrations: Mapping[str, ScoreCalibration] | None = None,
) -> dict[str, float]:
    """Fuse named score maps after independently calibrating every component."""

    if not isinstance(components, Mapping) or len(components) > _MAX_COMPONENTS:
        raise ValueError("components must be a bounded mapping.")
    if weights is not None and not isinstance(weights, Mapping):
        raise ValueError("weights must be a mapping.")
    if calibrations is not None and not isinstance(calibrations, Mapping):
        raise ValueError("calibrations must be a mapping.")
    totals: dict[str, float] = {}
    denominators: dict[str, float] = {}
    for component, raw_scores in components.items():
        if not isinstance(component, str) or not component or len(component) > 100:
            raise ValueError("component names must be bounded strings.")
        scores = _score_map(raw_scores, component)
        weight = 1.0 if weights is None else _finite(weights.get(component, 0.0), "weight")
        if weight < 0.0 or weight > 1000.0:
            raise ValueError("weights must be non-negative and bounded.")
        if weight == 0.0:
            continue
        calibration = (
            ScoreCalibration()
            if calibrations is None
            else calibrations.get(component, ScoreCalibration())
        )
        if not isinstance(calibration, ScoreCalibration):
            raise ValueError("every calibration must be ScoreCalibration.")
        for candidate_id, score in scores.items():
            calibrated = calibration.apply(score)
            totals[candidate_id] = totals.get(candidate_id, 0.0) + weight * calibrated
            denominators[candidate_id] = denominators.get(candidate_id, 0.0) + weight
    return {
        candidate_id: totals[candidate_id] / denominators[candidate_id]
        for candidate_id in sorted(totals)
        if denominators[candidate_id] > 0.0
    }


def splade_sparse_similarity(
    query_weights: Mapping[str, Any],
    document_weights: Mapping[str, Any],
    *,
    normalize: bool = True,
) -> float:
    """Score SPLADE-style sparse term expansions with optional cosine normalization."""

    if not isinstance(normalize, bool):
        raise ValueError("normalize must be boolean.")
    query = _sparse_weights(query_weights, "query_weights")
    document = _sparse_weights(document_weights, "document_weights")
    if not query or not document:
        return 0.0
    dot = sum(weight * document.get(term, 0.0) for term, weight in query.items())
    if not normalize:
        return dot
    query_norm = math.sqrt(sum(weight * weight for weight in query.values()))
    document_norm = math.sqrt(sum(weight * weight for weight in document.values()))
    if query_norm == 0.0 or document_norm == 0.0:
        return 0.0
    return max(0.0, min(dot / (query_norm * document_norm), 1.0))


def cosine_similarity(left: Any, right: Any) -> float:
    """Return cosine similarity in [-1, 1] for finite, non-zero equal vectors."""

    left_vector = _vector(left, "left")
    right_vector = _vector(right, "right")
    if len(left_vector) != len(right_vector):
        raise ValueError("vector dimensions must match.")
    dot = sum(a * b for a, b in zip(left_vector, right_vector))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))
    result = dot / (left_norm * right_norm)
    return max(-1.0, min(result, 1.0))


def colbert_maxsim(query_vectors: Any, document_vectors: Any) -> float:
    """Compute normalized ColBERT MaxSim late interaction in [0, 1]."""

    query = _vectors(query_vectors, "query_vectors")
    document = _vectors(document_vectors, "document_vectors")
    if len(query[0]) != len(document[0]):
        raise ValueError("query and document vector dimensions must match.")
    maxima: list[float] = []
    for query_vector in query:
        maximum = max(cosine_similarity(query_vector, vector) for vector in document)
        maxima.append((maximum + 1.0) / 2.0)
    return max(0.0, min(sum(maxima) / len(maxima), 1.0))


def aggregate_multi_vector_scores(
    scores: Sequence[Any],
    *,
    mode: str = "max",
    top_n: int = 3,
) -> float:
    """Aggregate bounded multi-vector similarities using max, mean, or top-mean."""

    if isinstance(scores, (str, bytes, bytearray)):
        raise ValueError("scores must be a numeric sequence.")
    raw = list(scores[:_MAX_VECTORS]) if isinstance(scores, Sequence) else list(
        itertools.islice(iter(scores), _MAX_VECTORS + 1)
    )
    if not raw or len(raw) > _MAX_VECTORS:
        raise ValueError("scores are empty or exceed the vector limit.")
    values = [_unit(value, "multi-vector score") for value in raw]
    if mode == "max":
        return max(values)
    if mode == "mean":
        return sum(values) / len(values)
    if mode == "top_mean":
        count = min(_exact_int(top_n, "top_n", 1, _MAX_VECTORS), len(values))
        selected = sorted(values, reverse=True)[:count]
        return sum(selected) / len(selected)
    raise ValueError("mode must be max, mean, or top_mean.")


@dataclass(frozen=True)
class MatryoshkaSelection:
    dimensions: int
    demand: float
    available_dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        available = tuple(
            sorted(
                {
                    _exact_int(value, "available dimension", 1, _MAX_DIMENSIONS)
                    for value in self.available_dimensions
                }
            )
        )
        if not available:
            raise ValueError("at least one available dimension is required.")
        object.__setattr__(self, "available_dimensions", available)
        selected = _exact_int(self.dimensions, "dimensions", 1, _MAX_DIMENSIONS)
        if selected not in available:
            raise ValueError("selected dimensions must be available.")
        object.__setattr__(self, "dimensions", selected)
        object.__setattr__(self, "demand", _unit(self.demand, "demand"))


def select_matryoshka_dimensions(
    available_dimensions: Sequence[int],
    *,
    budget: float,
    query_complexity: float = 0.5,
    uncertainty: float = 0.5,
) -> MatryoshkaSelection:
    """Choose the smallest available dimension satisfying a bounded demand score."""

    if isinstance(available_dimensions, (str, bytes, bytearray)):
        raise ValueError("available_dimensions must be an integer sequence.")
    values = tuple(
        sorted(
            {
                _exact_int(value, "available dimension", 1, _MAX_DIMENSIONS)
                for value in available_dimensions
            }
        )
    )
    if not values:
        raise ValueError("at least one available dimension is required.")
    if len(values) > 64:
        raise ValueError("available_dimensions exceeds the option limit.")
    selected_budget = _unit(budget, "budget")
    complexity = _unit(query_complexity, "query_complexity")
    selected_uncertainty = _unit(uncertainty, "uncertainty")
    demand = 0.50 * selected_budget + 0.30 * complexity + 0.20 * selected_uncertainty
    minimum, maximum = values[0], values[-1]
    target = minimum + demand * (maximum - minimum)
    selected = next((value for value in values if value >= target), maximum)
    return MatryoshkaSelection(
        dimensions=selected,
        demand=demand,
        available_dimensions=values,
    )


class SparseExpansionScorer(Protocol):
    """Adapter contract for SPLADE-like query/document expansion scorers."""

    def query_weights(self, query: str) -> Mapping[str, float]: ...

    def document_weights(self, text: str) -> Mapping[str, float]: ...


class LateInteractionScorer(Protocol):
    """Adapter contract for ColBERT/BGE-M3-style token or multi-vector encoders."""

    def query_vectors(self, query: str) -> Sequence[Sequence[float]]: ...

    def document_vectors(self, text: str) -> Sequence[Sequence[float]]: ...


__all__ = [
    "LateInteractionScorer",
    "MatryoshkaSelection",
    "ScoreCalibration",
    "SparseExpansionScorer",
    "aggregate_multi_vector_scores",
    "calibrate_score",
    "calibrated_weighted_fusion",
    "colbert_maxsim",
    "cosine_similarity",
    "select_matryoshka_dimensions",
    "splade_sparse_similarity",
]
