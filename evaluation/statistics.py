"""Deterministic statistical significance and calibration tools for RAG evaluation."""

from __future__ import annotations

import math
import operator
import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_MAX_SAMPLES = 1_000_000
_MAX_RESAMPLES = 100_000


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _unit(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _values(values: Iterable[Any], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric iterable.")
    result: list[float] = []
    try:
        iterator = iter(values)
        for item in iterator:
            if len(result) >= _MAX_SAMPLES:
                raise ValueError(f"{label} exceeds the sample limit.")
            result.append(_finite(item, label))
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{label} is not safely iterable.") from exc
    if not result:
        raise ValueError(f"{label} must not be empty.")
    return tuple(result)


def _paired(left: Iterable[Any], right: Iterable[Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    first = _values(left, "left")
    second = _values(right, "right")
    if len(first) != len(second):
        raise ValueError("paired samples must have equal length.")
    return first, second


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values.")
    selected = _unit(probability, "probability")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = selected * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_difference: float
    confidence_low: float
    confidence_high: float
    probability_positive: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class PairedPermutationResult:
    mean_difference: float
    p_value_two_sided: float
    permutations: int
    seed: int


def paired_bootstrap_difference(
    baseline: Iterable[Any],
    candidate: Iterable[Any],
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired bootstrap CI for candidate-minus-baseline mean difference."""

    left, right = _paired(baseline, candidate)
    count = _integer(resamples, "resamples", 100, _MAX_RESAMPLES)
    selected_confidence = _unit(confidence, "confidence")
    if not 0.5 <= selected_confidence < 1.0:
        raise ValueError("confidence must be at least 0.5 and below 1.")
    selected_seed = _integer(seed, "seed", -(2**63), 2**63 - 1)
    differences = tuple(b - a for a, b in zip(left, right))
    observed = sum(differences) / len(differences)
    rng = random.Random(selected_seed)
    draws: list[float] = []
    positive = 0
    for _ in range(count):
        total = 0.0
        for _index in range(len(differences)):
            total += differences[rng.randrange(len(differences))]
        value = total / len(differences)
        draws.append(value)
        positive += value > 0.0
    draws.sort()
    alpha = (1.0 - selected_confidence) / 2.0
    return PairedBootstrapResult(
        mean_difference=observed,
        confidence_low=_quantile(draws, alpha),
        confidence_high=_quantile(draws, 1.0 - alpha),
        probability_positive=positive / count,
        resamples=count,
        seed=selected_seed,
    )


def paired_permutation_test(
    baseline: Iterable[Any],
    candidate: Iterable[Any],
    *,
    permutations: int = 5_000,
    seed: int = 0,
) -> PairedPermutationResult:
    """Two-sided paired sign-flip permutation test for mean metric differences."""

    left, right = _paired(baseline, candidate)
    count = _integer(permutations, "permutations", 100, _MAX_RESAMPLES)
    selected_seed = _integer(seed, "seed", -(2**63), 2**63 - 1)
    differences = tuple(b - a for a, b in zip(left, right))
    observed = sum(differences) / len(differences)
    rng = random.Random(selected_seed)
    extreme = 0
    threshold = abs(observed) - 1e-15
    for _ in range(count):
        randomized = sum(
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        ) / len(differences)
        extreme += abs(randomized) >= threshold
    p_value = (extreme + 1) / (count + 1)
    return PairedPermutationResult(
        mean_difference=observed,
        p_value_two_sided=p_value,
        permutations=count,
        seed=selected_seed,
    )


def brier_score(confidences: Iterable[Any], outcomes: Iterable[Any]) -> float:
    confidence_values, outcome_values = _paired(confidences, outcomes)
    confidence_values = tuple(_unit(value, "confidence") for value in confidence_values)
    outcome_values = tuple(_unit(value, "outcome") for value in outcome_values)
    return sum((confidence - outcome) ** 2 for confidence, outcome in zip(confidence_values, outcome_values)) / len(confidence_values)


def expected_calibration_error(
    confidences: Iterable[Any],
    outcomes: Iterable[Any],
    *,
    bins: int = 10,
) -> float:
    confidence_values, outcome_values = _paired(confidences, outcomes)
    confidence_values = tuple(_unit(value, "confidence") for value in confidence_values)
    outcome_values = tuple(_unit(value, "outcome") for value in outcome_values)
    count = _integer(bins, "bins", 1, 1_000)
    totals = [0 for _ in range(count)]
    confidence_sums = [0.0 for _ in range(count)]
    outcome_sums = [0.0 for _ in range(count)]
    for confidence, outcome in zip(confidence_values, outcome_values):
        index = min(int(confidence * count), count - 1)
        totals[index] += 1
        confidence_sums[index] += confidence
        outcome_sums[index] += outcome
    error = 0.0
    total_count = len(confidence_values)
    for population, confidence_sum, outcome_sum in zip(totals, confidence_sums, outcome_sums):
        if population:
            error += population / total_count * abs(
                confidence_sum / population - outcome_sum / population
            )
    return error


@dataclass(frozen=True)
class SelectiveRiskPoint:
    target_coverage: float
    achieved_coverage: float
    risk: float
    threshold: float


def selective_risk_curve(
    confidences: Iterable[Any],
    losses: Iterable[Any],
    *,
    coverages: Sequence[float] = (1.0, 0.9, 0.75, 0.5, 0.25),
) -> tuple[SelectiveRiskPoint, ...]:
    """Measure loss after abstaining on the lowest-confidence examples."""

    confidence_values, loss_values = _paired(confidences, losses)
    rows = sorted(
        (
            (_unit(confidence, "confidence"), _unit(loss, "loss"), index)
            for index, (confidence, loss) in enumerate(zip(confidence_values, loss_values))
        ),
        key=lambda value: (-value[0], value[2]),
    )
    if isinstance(coverages, (str, bytes, bytearray)) or not coverages:
        raise ValueError("coverages must be a non-empty sequence.")
    result: list[SelectiveRiskPoint] = []
    for raw_coverage in coverages:
        coverage = _unit(raw_coverage, "coverage")
        if coverage <= 0.0:
            raise ValueError("coverage must be greater than zero.")
        retained = max(1, int(math.ceil(len(rows) * coverage)))
        selected = rows[:retained]
        result.append(
            SelectiveRiskPoint(
                target_coverage=coverage,
                achieved_coverage=retained / len(rows),
                risk=sum(row[1] for row in selected) / retained,
                threshold=selected[-1][0],
            )
        )
    return tuple(result)


__all__ = [
    "PairedBootstrapResult",
    "PairedPermutationResult",
    "SelectiveRiskPoint",
    "brier_score",
    "expected_calibration_error",
    "paired_bootstrap_difference",
    "paired_permutation_test",
    "selective_risk_curve",
]
