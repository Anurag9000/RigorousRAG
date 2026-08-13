"""Uncertainty decomposition for stochastic RAG confidence estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping

_MAX_MEMBERS = 10_000


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def _probabilities(values: Iterable[Any], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric iterable.")
    result: list[float] = []
    try:
        iterator = iter(values)
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    for value in iterator:
        if len(result) >= _MAX_MEMBERS:
            raise ValueError(f"{label} exceeds the member limit.")
        result.append(_unit(value, label))
    if not result:
        raise ValueError(f"{label} must not be empty.")
    return tuple(result)


@dataclass(frozen=True)
class UncertaintyBreakdown:
    predictive_mean: float
    total: float
    aleatoric: float
    epistemic: float
    disagreement: float
    member_count: int


def decompose_binary_uncertainty(probabilities: Iterable[Any]) -> UncertaintyBreakdown:
    """Decompose Bernoulli predictive variance into aleatoric and epistemic terms.

    For ensemble/dropout member probabilities ``p_i``:
    total = p_bar(1-p_bar), aleatoric = mean(p_i(1-p_i)), and epistemic is the
    residual/ensemble variance. The identity is exact up to floating-point error.
    """

    values = _probabilities(probabilities, "probabilities")
    mean = fmean(values)
    aleatoric = fmean(value * (1.0 - value) for value in values)
    epistemic = fmean((value - mean) ** 2 for value in values)
    total = mean * (1.0 - mean)
    residual = max(0.0, total - aleatoric)
    epistemic = max(epistemic, residual) if abs(epistemic - residual) > 1e-12 else residual
    disagreement = max(values) - min(values)
    return UncertaintyBreakdown(
        predictive_mean=mean,
        total=total,
        aleatoric=aleatoric,
        epistemic=epistemic,
        disagreement=disagreement,
        member_count=len(values),
    )


@dataclass(frozen=True)
class RagUncertaintySignal:
    confidence: float
    retrieval_uncertainty: float
    generation_uncertainty: float
    evidence_conflict: float
    proof_gap: float
    aggregate_uncertainty: float
    should_abstain: bool


def combine_rag_uncertainty(
    *,
    retrieval_confidence: Any,
    generation_confidence: Any,
    evidence_conflict: Any = 0.0,
    proof_completeness: Any = 1.0,
    weights: Mapping[str, Any] | None = None,
    abstain_threshold: Any = 0.5,
) -> RagUncertaintySignal:
    """Combine calibrated component uncertainties into one bounded abstention signal."""

    retrieval = _unit(retrieval_confidence, "retrieval_confidence")
    generation = _unit(generation_confidence, "generation_confidence")
    conflict = _unit(evidence_conflict, "evidence_conflict")
    completeness = _unit(proof_completeness, "proof_completeness")
    threshold = _unit(abstain_threshold, "abstain_threshold")
    raw_weights = weights or {
        "retrieval": 0.30,
        "generation": 0.30,
        "conflict": 0.25,
        "proof_gap": 0.15,
    }
    if not isinstance(raw_weights, Mapping):
        raise ValueError("weights must be a mapping.")
    normalized: dict[str, float] = {}
    for name in ("retrieval", "generation", "conflict", "proof_gap"):
        value = raw_weights.get(name, 0.0)
        if isinstance(value, bool):
            raise ValueError("weights must be finite and non-negative.")
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weights must be finite and non-negative.") from exc
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError("weights must be finite and non-negative.")
        normalized[name] = parsed
    total_weight = sum(normalized.values())
    if total_weight <= 0.0:
        raise ValueError("at least one uncertainty weight must be positive.")
    retrieval_uncertainty = 1.0 - retrieval
    generation_uncertainty = 1.0 - generation
    proof_gap = 1.0 - completeness
    aggregate = (
        normalized["retrieval"] * retrieval_uncertainty
        + normalized["generation"] * generation_uncertainty
        + normalized["conflict"] * conflict
        + normalized["proof_gap"] * proof_gap
    ) / total_weight
    confidence = 1.0 - aggregate
    return RagUncertaintySignal(
        confidence=confidence,
        retrieval_uncertainty=retrieval_uncertainty,
        generation_uncertainty=generation_uncertainty,
        evidence_conflict=conflict,
        proof_gap=proof_gap,
        aggregate_uncertainty=aggregate,
        should_abstain=aggregate >= threshold,
    )


__all__ = [
    "RagUncertaintySignal",
    "UncertaintyBreakdown",
    "combine_rag_uncertainty",
    "decompose_binary_uncertainty",
]
