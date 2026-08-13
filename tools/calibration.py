"""Calibration, confidence aggregation, and selective-answering utilities.

The module is dependency-free so it can be used in unit tests, offline evaluation,
and production request paths without pulling in a numerical stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


def _probability(value: float, label: str = "probability") -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be a finite value in [0, 1].")
    return parsed


def _binary_label(value: object) -> int:
    if value in (0, False):
        return 0
    if value in (1, True):
        return 1
    raise ValueError("labels must contain only binary values.")


def _paired(
    confidences: Sequence[float],
    labels: Sequence[object],
) -> List[Tuple[float, int]]:
    if len(confidences) != len(labels):
        raise ValueError("confidences and labels must have the same length.")
    if not confidences:
        return []
    return [
        (_probability(confidence, "confidence"), _binary_label(label))
        for confidence, label in zip(confidences, labels)
    ]


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.mean_confidence - self.accuracy)


@dataclass(frozen=True)
class SelectivePoint:
    threshold: float
    coverage: float
    accuracy: float
    risk: float
    answered: int
    total: int


@dataclass(frozen=True)
class ThresholdDecision:
    threshold: float
    expected_cost: float
    false_positive_rate: float
    false_negative_rate: float
    coverage: float


@dataclass(frozen=True)
class ConfidenceDecision:
    confidence: float
    answer: bool
    threshold: float
    reason: str


def reliability_bins(
    confidences: Sequence[float],
    labels: Sequence[object],
    *,
    bins: int = 10,
) -> List[CalibrationBin]:
    """Return equal-width reliability bins over [0, 1]."""

    if isinstance(bins, bool) or int(bins) != bins or bins <= 0:
        raise ValueError("bins must be a positive integer.")
    pairs = _paired(confidences, labels)
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(int(bins))]
    for confidence, label in pairs:
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((confidence, label))

    output: List[CalibrationBin] = []
    for index, bucket in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        if bucket:
            mean_confidence = sum(item[0] for item in bucket) / len(bucket)
            accuracy = sum(item[1] for item in bucket) / len(bucket)
        else:
            mean_confidence = 0.0
            accuracy = 0.0
        output.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_confidence=mean_confidence,
                accuracy=accuracy,
            )
        )
    return output


def expected_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[object],
    *,
    bins: int = 10,
) -> float:
    pairs = _paired(confidences, labels)
    if not pairs:
        return 0.0
    bucketed = reliability_bins(confidences, labels, bins=bins)
    total = len(pairs)
    return sum((bucket.count / total) * bucket.gap for bucket in bucketed)


def maximum_calibration_error(
    confidences: Sequence[float],
    labels: Sequence[object],
    *,
    bins: int = 10,
) -> float:
    populated = [
        bucket.gap
        for bucket in reliability_bins(confidences, labels, bins=bins)
        if bucket.count
    ]
    return max(populated, default=0.0)


def brier_score(confidences: Sequence[float], labels: Sequence[object]) -> float:
    pairs = _paired(confidences, labels)
    if not pairs:
        return 0.0
    return sum((confidence - label) ** 2 for confidence, label in pairs) / len(pairs)


def log_loss(
    confidences: Sequence[float],
    labels: Sequence[object],
    *,
    epsilon: float = 1e-12,
) -> float:
    pairs = _paired(confidences, labels)
    if not pairs:
        return 0.0
    if epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError("epsilon must be in (0, 0.5).")
    total = 0.0
    for confidence, label in pairs:
        p = min(max(confidence, epsilon), 1.0 - epsilon)
        total -= label * math.log(p) + (1 - label) * math.log(1.0 - p)
    return total / len(pairs)


def selective_curve(
    confidences: Sequence[float],
    labels: Sequence[object],
    *,
    thresholds: Optional[Iterable[float]] = None,
) -> List[SelectivePoint]:
    """Measure quality/coverage as the system abstains below each threshold."""

    pairs = _paired(confidences, labels)
    total = len(pairs)
    if thresholds is None:
        candidates = sorted({0.0, 1.0, *(confidence for confidence, _ in pairs)})
    else:
        candidates = sorted({_probability(value, "threshold") for value in thresholds})
    output: List[SelectivePoint] = []
    for threshold in candidates:
        answered = [label for confidence, label in pairs if confidence >= threshold]
        count = len(answered)
        accuracy = (sum(answered) / count) if count else 1.0
        coverage = (count / total) if total else 0.0
        output.append(
            SelectivePoint(
                threshold=threshold,
                coverage=coverage,
                accuracy=accuracy,
                risk=1.0 - accuracy,
                answered=count,
                total=total,
            )
        )
    return output


def optimize_threshold(
    confidences: Sequence[float],
    labels: Sequence[object],
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 1.0,
    abstain_cost: float = 0.0,
) -> ThresholdDecision:
    """Choose an answer threshold under asymmetric error and abstention costs."""

    pairs = _paired(confidences, labels)
    for value, label in (
        (false_positive_cost, "false_positive_cost"),
        (false_negative_cost, "false_negative_cost"),
        (abstain_cost, "abstain_cost"),
    ):
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(f"{label} must be finite and non-negative.")
    if not pairs:
        return ThresholdDecision(1.0, 0.0, 0.0, 0.0, 0.0)

    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    candidates = sorted({0.0, 1.0, *(confidence for confidence, _ in pairs)})
    best: Optional[ThresholdDecision] = None

    for threshold in candidates:
        fp = fn = abstained = answered = 0
        for confidence, label in pairs:
            if confidence < threshold:
                abstained += 1
                if label:
                    fn += 1
                continue
            answered += 1
            if not label:
                fp += 1
        cost = (
            fp * false_positive_cost
            + fn * false_negative_cost
            + abstained * abstain_cost
        ) / len(pairs)
        decision = ThresholdDecision(
            threshold=threshold,
            expected_cost=cost,
            false_positive_rate=(fp / negatives) if negatives else 0.0,
            false_negative_rate=(fn / positives) if positives else 0.0,
            coverage=answered / len(pairs),
        )
        if best is None or (decision.expected_cost, -decision.coverage, decision.threshold) < (
            best.expected_cost,
            -best.coverage,
            best.threshold,
        ):
            best = decision
    assert best is not None
    return best


def confidence_from_signals(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    missing: str = "ignore",
) -> float:
    """Combine retrieval/citation/self-consistency confidence signals.

    Weights are normalized over available signals. Negative weights are rejected.
    """

    if missing not in {"ignore", "zero"}:
        raise ValueError("missing must be either 'ignore' or 'zero'.")
    if not signals:
        return 0.0

    chosen_weights = dict(weights or {key: 1.0 for key in signals})
    numerator = denominator = 0.0
    keys = set(chosen_weights) | (set(signals) if missing == "zero" else set())
    for key in keys:
        weight = float(chosen_weights.get(key, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("confidence weights must be finite and non-negative.")
        if key not in signals:
            if missing == "zero":
                denominator += weight
            continue
        value = _probability(signals[key], f"signal {key!r}")
        numerator += weight * value
        denominator += weight
    return numerator / denominator if denominator else 0.0


def selective_decision(
    confidence: float,
    *,
    threshold: float,
    minimum_citation_coverage: Optional[float] = None,
    citation_coverage: Optional[float] = None,
) -> ConfidenceDecision:
    confidence = _probability(confidence, "confidence")
    threshold = _probability(threshold, "threshold")
    if minimum_citation_coverage is not None:
        minimum_citation_coverage = _probability(
            minimum_citation_coverage, "minimum_citation_coverage"
        )
        if citation_coverage is None:
            return ConfidenceDecision(
                confidence, False, threshold, "citation coverage is unavailable"
            )
        citation_coverage = _probability(citation_coverage, "citation_coverage")
        if citation_coverage < minimum_citation_coverage:
            return ConfidenceDecision(
                confidence,
                False,
                threshold,
                "citation coverage is below the configured minimum",
            )
    if confidence < threshold:
        return ConfidenceDecision(
            confidence, False, threshold, "confidence is below the answer threshold"
        )
    return ConfidenceDecision(confidence, True, threshold, "confidence gate passed")
