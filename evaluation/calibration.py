"""Calibration, selective-answering, and abstention policy evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


def _probability(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a probability.")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be in [0, 1].")
    return parsed


@dataclass(frozen=True)
class CalibrationExample:
    confidence: float
    correct: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        if not isinstance(self.correct, bool):
            raise ValueError("correct must be boolean.")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("weight must be finite and positive.")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class ConfidenceSignals:
    """Keep raw and evidence-derived confidence signals explicit rather than conflating them."""

    raw_retrieval: float
    citation_coverage: float
    self_consistency: float
    calibrated: float | None = None

    def __post_init__(self) -> None:
        for name in ("raw_retrieval", "citation_coverage", "self_consistency"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if self.calibrated is not None:
            object.__setattr__(self, "calibrated", _probability(self.calibrated, "calibrated"))


@dataclass(frozen=True)
class CorrectnessPolicy:
    min_answer_score: float = 0.5
    min_citation_support: float = 0.5
    require_both: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "min_answer_score", _probability(self.min_answer_score, "min_answer_score")
        )
        object.__setattr__(
            self,
            "min_citation_support",
            _probability(self.min_citation_support, "min_citation_support"),
        )

    def label(self, *, answer_score: float, citation_support: float) -> bool:
        answer_ok = _probability(answer_score, "answer_score") >= self.min_answer_score
        citation_ok = (
            _probability(citation_support, "citation_support") >= self.min_citation_support
        )
        return answer_ok and citation_ok if self.require_both else answer_ok or citation_ok


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    weight: float
    mean_confidence: float
    accuracy: float
    calibration_gap: float


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    risk: float
    accuracy: float
    accepted_count: int
    accepted_weight: float


@dataclass(frozen=True)
class ThresholdDecision:
    threshold: float
    expected_cost: float
    coverage: float
    risk: float
    accepted_count: int


def _examples(values: Iterable[CalibrationExample]) -> tuple[CalibrationExample, ...]:
    examples = tuple(values)
    if not examples:
        raise ValueError("at least one calibration example is required.")
    return examples


def reliability_bins(
    values: Iterable[CalibrationExample], *, bin_count: int = 10
) -> tuple[ReliabilityBin, ...]:
    examples = _examples(values)
    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 1:
        raise ValueError("bin_count must be a positive integer.")
    buckets: list[list[CalibrationExample]] = [[] for _ in range(bin_count)]
    for example in examples:
        index = min(int(example.confidence * bin_count), bin_count - 1)
        buckets[index].append(example)
    result = []
    for index, bucket in enumerate(buckets):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        total_weight = sum(item.weight for item in bucket)
        if total_weight:
            mean_confidence = sum(item.confidence * item.weight for item in bucket) / total_weight
            accuracy = sum(float(item.correct) * item.weight for item in bucket) / total_weight
        else:
            mean_confidence = 0.0
            accuracy = 0.0
        result.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                weight=total_weight,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                calibration_gap=abs(mean_confidence - accuracy) if total_weight else 0.0,
            )
        )
    return tuple(result)


def expected_calibration_error(
    values: Iterable[CalibrationExample], *, bin_count: int = 10
) -> float:
    bins = reliability_bins(values, bin_count=bin_count)
    total_weight = sum(item.weight for item in bins)
    return sum(item.weight * item.calibration_gap for item in bins) / total_weight


def brier_score(values: Iterable[CalibrationExample]) -> float:
    examples = _examples(values)
    total_weight = sum(item.weight for item in examples)
    return sum(
        item.weight * (item.confidence - float(item.correct)) ** 2 for item in examples
    ) / total_weight


def _selective_point(examples: Sequence[CalibrationExample], threshold: float) -> SelectivePoint:
    accepted = [item for item in examples if item.confidence >= threshold]
    total_weight = sum(item.weight for item in examples)
    accepted_weight = sum(item.weight for item in accepted)
    if accepted_weight:
        accuracy = sum(float(item.correct) * item.weight for item in accepted) / accepted_weight
        risk = 1.0 - accuracy
    else:
        accuracy = 1.0
        risk = 0.0
    return SelectivePoint(
        threshold=threshold,
        coverage=accepted_weight / total_weight,
        risk=risk,
        accuracy=accuracy,
        accepted_count=len(accepted),
        accepted_weight=accepted_weight,
    )


def risk_coverage_curve(values: Iterable[CalibrationExample]) -> tuple[SelectivePoint, ...]:
    examples = _examples(values)
    thresholds = sorted({item.confidence for item in examples}, reverse=True)
    points = [_selective_point(examples, math.nextafter(1.0, math.inf))]
    points.extend(_selective_point(examples, threshold) for threshold in thresholds)
    if thresholds[-1] > 0.0:
        points.append(_selective_point(examples, 0.0))
    return tuple(points)


def optimize_abstention_threshold(
    values: Iterable[CalibrationExample],
    *,
    incorrect_answer_cost: float = 1.0,
    abstention_cost: float = 0.2,
) -> ThresholdDecision:
    examples = _examples(values)
    incorrect_cost = float(incorrect_answer_cost)
    selected_abstention_cost = float(abstention_cost)
    if (
        not math.isfinite(incorrect_cost)
        or not math.isfinite(selected_abstention_cost)
        or incorrect_cost < 0.0
        or selected_abstention_cost < 0.0
    ):
        raise ValueError("answer and abstention costs must be finite and non-negative.")
    total_weight = sum(item.weight for item in examples)
    best: ThresholdDecision | None = None
    for point in risk_coverage_curve(examples):
        incorrect_weight = point.risk * point.accepted_weight
        abstained_weight = total_weight - point.accepted_weight
        expected = (
            incorrect_weight * incorrect_cost + abstained_weight * selected_abstention_cost
        ) / total_weight
        candidate = ThresholdDecision(
            threshold=point.threshold,
            expected_cost=expected,
            coverage=point.coverage,
            risk=point.risk,
            accepted_count=point.accepted_count,
        )
        if best is None or (candidate.expected_cost, -candidate.coverage, candidate.threshold) < (
            best.expected_cost,
            -best.coverage,
            best.threshold,
        ):
            best = candidate
    assert best is not None
    return best


class HistogramCalibrator:
    """Simple dependency-free empirical calibrator for held-out confidence labels."""

    def __init__(self, *, bin_count: int = 10, smoothing: float = 1.0) -> None:
        if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count < 1:
            raise ValueError("bin_count must be a positive integer.")
        smoothing = float(smoothing)
        if not math.isfinite(smoothing) or smoothing < 0.0:
            raise ValueError("smoothing must be finite and non-negative.")
        self.bin_count = bin_count
        self.smoothing = smoothing
        self._probabilities: tuple[float, ...] | None = None

    def fit(self, values: Iterable[CalibrationExample]) -> "HistogramCalibrator":
        examples = _examples(values)
        global_correct = sum(float(item.correct) * item.weight for item in examples)
        global_weight = sum(item.weight for item in examples)
        prior = global_correct / global_weight
        buckets: list[list[CalibrationExample]] = [[] for _ in range(self.bin_count)]
        for item in examples:
            buckets[min(int(item.confidence * self.bin_count), self.bin_count - 1)].append(item)
        probabilities = []
        for bucket in buckets:
            weight = sum(item.weight for item in bucket)
            correct = sum(float(item.correct) * item.weight for item in bucket)
            denominator = weight + self.smoothing
            probability = (
                (correct + self.smoothing * prior) / denominator if denominator else prior
            )
            probabilities.append(probability)
        self._probabilities = tuple(probabilities)
        return self

    def predict(self, confidence: float) -> float:
        if self._probabilities is None:
            raise RuntimeError("calibrator must be fitted before prediction.")
        probability = _probability(confidence, "confidence")
        index = min(int(probability * self.bin_count), self.bin_count - 1)
        return self._probabilities[index]


__all__ = [
    "CalibrationExample",
    "ConfidenceSignals",
    "CorrectnessPolicy",
    "HistogramCalibrator",
    "ReliabilityBin",
    "SelectivePoint",
    "ThresholdDecision",
    "brier_score",
    "expected_calibration_error",
    "optimize_abstention_threshold",
    "reliability_bins",
    "risk_coverage_curve",
]
