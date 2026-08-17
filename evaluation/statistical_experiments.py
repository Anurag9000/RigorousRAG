"""Deterministic paired statistics and promotion primitives for retrieval experiments.

The repository already records retrieval and route experiments.  This module owns the
statistics needed to turn repeated paired measurements into governed evidence: paired
bootstrap confidence intervals, paired randomisation/permutation tests, standardized
paired effects, Holm family-wise correction, Benjamini-Hochberg FDR correction and
multi-metric promotion decisions.  Callers must supply measurements; importing this
module does not run any benchmark or model.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_OBSERVATIONS = 5_000_000
_MAX_RESAMPLES = 1_000_000


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: Any, label: str, *, inclusive_zero: bool = True) -> float:
    result = _finite(value, label)
    lower_ok = result >= 0.0 if inclusive_zero else result > 0.0
    if not lower_ok or result > 1.0:
        raise ValueError(f"{label} must be within its probability range")
    return result


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _paired(baseline: Sequence[Any], candidate: Sequence[Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be non-empty aligned sequences")
    if len(baseline) > _MAX_OBSERVATIONS:
        raise ValueError("paired sample exceeds the observation limit")
    left = tuple(_finite(value, "baseline observation") for value in baseline)
    right = tuple(_finite(value, "candidate observation") for value in candidate)
    return left, right


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires values")
    p = min(max(probability, 0.0), 1.0)
    position = p * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class Alternative(str, Enum):
    TWO_SIDED = "two_sided"
    GREATER = "greater"
    LESS = "less"


@dataclass(frozen=True)
class BootstrapDifference:
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    ci_lower: float
    ci_upper: float
    confidence: float
    resamples: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_lower > 0.0 or self.ci_upper < 0.0


@dataclass(frozen=True)
class PermutationResult:
    observed_difference: float
    p_value: float
    alternative: Alternative
    resamples: int
    seed: int


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    direction: MetricDirection
    baseline_mean: float
    candidate_mean: float
    raw_delta: float
    improvement: float
    paired_effect: float | None
    p_value: float
    adjusted_p_value: float | None = None
    ci_lower_improvement: float | None = None
    ci_upper_improvement: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _identifier(self.metric, "metric", 200))
        if not isinstance(self.direction, MetricDirection):
            object.__setattr__(self, "direction", MetricDirection(self.direction))
        for name in (
            "baseline_mean",
            "candidate_mean",
            "raw_delta",
            "improvement",
            "p_value",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.paired_effect is not None:
            object.__setattr__(self, "paired_effect", _finite(self.paired_effect, "paired_effect"))
        _probability(self.p_value, "p_value")
        if self.adjusted_p_value is not None:
            object.__setattr__(
                self,
                "adjusted_p_value",
                _probability(self.adjusted_p_value, "adjusted_p_value"),
            )
        for name in ("ci_lower_improvement", "ci_upper_improvement"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if (self.ci_lower_improvement is None) != (self.ci_upper_improvement is None):
            raise ValueError("both confidence interval bounds must be supplied")
        if self.ci_lower_improvement is not None and self.ci_lower_improvement > self.ci_upper_improvement:
            raise ValueError("confidence interval is inverted")


@dataclass(frozen=True)
class MetricPromotionRule:
    metric: str
    direction: MetricDirection
    minimum_improvement: float = 0.0
    alpha: float = 0.05
    require_ci_above_threshold: bool = True
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _identifier(self.metric, "metric", 200))
        if not isinstance(self.direction, MetricDirection):
            object.__setattr__(self, "direction", MetricDirection(self.direction))
        object.__setattr__(
            self,
            "minimum_improvement",
            _finite(self.minimum_improvement, "minimum_improvement"),
        )
        alpha = _probability(self.alpha, "alpha", inclusive_zero=False)
        if alpha >= 1.0:
            raise ValueError("alpha must be less than 1")
        object.__setattr__(self, "alpha", alpha)
        if not isinstance(self.require_ci_above_threshold, bool) or not isinstance(self.required, bool):
            raise ValueError("promotion flags must be boolean")


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reasons: tuple[str, ...]
    comparisons: tuple[MetricComparison, ...]
    correction: str


def paired_bootstrap_difference(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapDifference:
    """Percentile paired-bootstrap CI for candidate - baseline mean difference."""

    left, right = _paired(baseline, candidate)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or not 100 <= resamples <= _MAX_RESAMPLES:
        raise ValueError("resamples must be between 100 and 1,000,000")
    selected_confidence = _probability(confidence, "confidence", inclusive_zero=False)
    if selected_confidence >= 1.0:
        raise ValueError("confidence must be less than 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be a non-negative 63-bit integer")
    differences = tuple(candidate_value - baseline_value for baseline_value, candidate_value in zip(left, right))
    observed = statistics.fmean(differences)
    generator = random.Random(seed)
    count = len(differences)
    bootstrapped: list[float] = []
    for _ in range(resamples):
        bootstrapped.append(statistics.fmean(differences[generator.randrange(count)] for _ in range(count)))
    bootstrapped.sort()
    tail = (1.0 - selected_confidence) / 2.0
    return BootstrapDifference(
        baseline_mean=statistics.fmean(left),
        candidate_mean=statistics.fmean(right),
        mean_difference=observed,
        ci_lower=_quantile(bootstrapped, tail),
        ci_upper=_quantile(bootstrapped, 1.0 - tail),
        confidence=selected_confidence,
        resamples=resamples,
        seed=seed,
    )


def paired_permutation_test(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    resamples: int = 20_000,
    alternative: Alternative = Alternative.TWO_SIDED,
    seed: int = 0,
) -> PermutationResult:
    """Paired randomisation test by independently swapping each pair under H0."""

    left, right = _paired(baseline, candidate)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or not 100 <= resamples <= _MAX_RESAMPLES:
        raise ValueError("resamples must be between 100 and 1,000,000")
    if not isinstance(alternative, Alternative):
        alternative = Alternative(alternative)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be a non-negative 63-bit integer")
    differences = tuple(candidate_value - baseline_value for baseline_value, candidate_value in zip(left, right))
    observed = statistics.fmean(differences)
    generator = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        randomized = statistics.fmean(value if generator.getrandbits(1) else -value for value in differences)
        if alternative == Alternative.TWO_SIDED:
            extreme += abs(randomized) >= abs(observed)
        elif alternative == Alternative.GREATER:
            extreme += randomized >= observed
        else:
            extreme += randomized <= observed
    # Add-one correction prevents reporting zero from a finite Monte-Carlo sample.
    p_value = (extreme + 1.0) / (resamples + 1.0)
    return PermutationResult(observed, p_value, alternative, resamples, seed)


def paired_standardized_effect(baseline: Sequence[Any], candidate: Sequence[Any]) -> float | None:
    """Return Cohen's dz, or ``None`` when the paired SD is undefined or exactly zero."""

    left, right = _paired(baseline, candidate)
    differences = [candidate_value - baseline_value for baseline_value, candidate_value in zip(left, right)]
    if len(differences) < 2:
        return None
    deviation = statistics.stdev(differences)
    if deviation == 0.0:
        return None
    return statistics.fmean(differences) / deviation


def holm_adjust(p_values: Mapping[str, Any]) -> dict[str, float]:
    """Holm step-down family-wise error adjustment, keyed without reordering output."""

    if not p_values or len(p_values) > 100_000:
        raise ValueError("p_values must be a non-empty bounded mapping")
    cleaned = {str(key): _probability(value, f"p[{key}]") for key, value in p_values.items()}
    ordered = sorted(cleaned.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for index, (key, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return {key: adjusted[key] for key in cleaned}


def benjamini_hochberg_adjust(p_values: Mapping[str, Any]) -> dict[str, float]:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values."""

    if not p_values or len(p_values) > 100_000:
        raise ValueError("p_values must be a non-empty bounded mapping")
    cleaned = {str(key): _probability(value, f"p[{key}]") for key, value in p_values.items()}
    ordered = sorted(cleaned.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        key, value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, value * count / rank, 1.0)
        adjusted[key] = running
    return {key: adjusted[key] for key in cleaned}


def compare_metric(
    metric: str,
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    *,
    direction: MetricDirection,
    bootstrap_resamples: int = 10_000,
    permutation_resamples: int = 20_000,
    seed: int = 0,
) -> MetricComparison:
    """Create one paired comparison on a common improvement-positive scale."""

    selected_direction = MetricDirection(direction)
    bootstrap = paired_bootstrap_difference(
        baseline,
        candidate,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    permutation = paired_permutation_test(
        baseline,
        candidate,
        resamples=permutation_resamples,
        seed=seed,
    )
    raw_delta = bootstrap.mean_difference
    sign = 1.0 if selected_direction == MetricDirection.HIGHER_IS_BETTER else -1.0
    raw_effect = paired_standardized_effect(baseline, candidate)
    paired_effect = None if raw_effect is None else raw_effect * sign
    return MetricComparison(
        metric=metric,
        direction=selected_direction,
        baseline_mean=bootstrap.baseline_mean,
        candidate_mean=bootstrap.candidate_mean,
        raw_delta=raw_delta,
        improvement=sign * raw_delta,
        paired_effect=paired_effect,
        p_value=permutation.p_value,
        ci_lower_improvement=min(sign * bootstrap.ci_lower, sign * bootstrap.ci_upper),
        ci_upper_improvement=max(sign * bootstrap.ci_lower, sign * bootstrap.ci_upper),
    )


def apply_multiplicity(
    comparisons: Sequence[MetricComparison],
    *,
    method: str = "holm",
) -> tuple[MetricComparison, ...]:
    if not comparisons or len(comparisons) > 100_000:
        raise ValueError("comparisons must be a non-empty bounded sequence")
    values = {comparison.metric: comparison.p_value for comparison in comparisons}
    if len(values) != len(comparisons):
        raise ValueError("metric names must be unique")
    if method == "holm":
        adjusted = holm_adjust(values)
    elif method in {"bh", "benjamini_hochberg"}:
        adjusted = benjamini_hochberg_adjust(values)
    elif method == "none":
        adjusted = values
    else:
        raise ValueError("method must be holm, bh, benjamini_hochberg, or none")
    return tuple(
        MetricComparison(
            metric=value.metric,
            direction=value.direction,
            baseline_mean=value.baseline_mean,
            candidate_mean=value.candidate_mean,
            raw_delta=value.raw_delta,
            improvement=value.improvement,
            paired_effect=value.paired_effect,
            p_value=value.p_value,
            adjusted_p_value=adjusted[value.metric],
            ci_lower_improvement=value.ci_lower_improvement,
            ci_upper_improvement=value.ci_upper_improvement,
        )
        for value in comparisons
    )


def decide_promotion(
    comparisons: Sequence[MetricComparison],
    rules: Sequence[MetricPromotionRule],
    *,
    correction: str = "holm",
) -> PromotionDecision:
    """Require all mandatory metric gates after multiplicity correction."""

    adjusted = apply_multiplicity(comparisons, method=correction)
    by_metric = {comparison.metric: comparison for comparison in adjusted}
    reasons: list[str] = []
    promoted = True
    for rule in rules:
        if not isinstance(rule, MetricPromotionRule):
            raise ValueError("rules must contain MetricPromotionRule values")
        comparison = by_metric.get(rule.metric)
        if comparison is None:
            if rule.required:
                promoted = False
                reasons.append(f"missing required metric {rule.metric}")
            continue
        if comparison.direction != rule.direction:
            raise ValueError(f"metric direction mismatch for {rule.metric}")
        failures: list[str] = []
        if comparison.improvement < rule.minimum_improvement:
            failures.append(
                f"improvement {comparison.improvement:.6g} < required {rule.minimum_improvement:.6g}"
            )
        adjusted_p = comparison.adjusted_p_value if comparison.adjusted_p_value is not None else comparison.p_value
        if adjusted_p > rule.alpha:
            failures.append(f"adjusted p {adjusted_p:.6g} > alpha {rule.alpha:.6g}")
        if rule.require_ci_above_threshold:
            if comparison.ci_lower_improvement is None or comparison.ci_lower_improvement < rule.minimum_improvement:
                failures.append("confidence interval does not clear the required improvement")
        if failures and rule.required:
            promoted = False
        if failures:
            reasons.append(f"{rule.metric}: " + "; ".join(failures))
    if promoted and not reasons:
        reasons.append("all required statistical promotion gates passed")
    return PromotionDecision(promoted, tuple(reasons), adjusted, correction)


__all__ = [
    "Alternative",
    "BootstrapDifference",
    "MetricComparison",
    "MetricDirection",
    "MetricPromotionRule",
    "PermutationResult",
    "PromotionDecision",
    "apply_multiplicity",
    "benjamini_hochberg_adjust",
    "compare_metric",
    "decide_promotion",
    "holm_adjust",
    "paired_bootstrap_difference",
    "paired_permutation_test",
    "paired_standardized_effect",
]
