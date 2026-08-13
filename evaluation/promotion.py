"""Statistical multi-metric promotion decisions for retrieval and RAG experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evaluation.multiple_comparisons import holm_adjust, noninferiority_gate
from evaluation.statistics import paired_bootstrap_difference, paired_permutation_test


@dataclass(frozen=True)
class MetricPromotionRule:
    name: str
    higher_is_better: bool = True
    noninferiority_margin: float = 0.0
    require_significance: bool = False


@dataclass(frozen=True)
class MetricPromotionResult:
    name: str
    mean_difference: float
    confidence_low: float
    confidence_high: float
    raw_p_value: float
    adjusted_p_value: float
    noninferior: bool
    significant: bool
    passed: bool


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    metrics: tuple[MetricPromotionResult, ...]


def evaluate_promotion(
    baseline: Mapping[str, Sequence[float]],
    candidate: Mapping[str, Sequence[float]],
    rules: Sequence[MetricPromotionRule],
    *,
    confidence: float = 0.95,
    alpha: float = 0.05,
    resamples: int = 3000,
    permutations: int = 5000,
    seed: int = 0,
) -> PromotionDecision:
    """Require every governed metric to pass CI, significance, and direction rules."""

    if not rules:
        raise ValueError("at least one promotion rule is required.")
    names = [rule.name for rule in rules]
    if len(names) != len(set(names)):
        raise ValueError("promotion rule names must be unique.")

    bootstrap = {}
    permutations_by_metric = {}
    raw_p_values = {}
    for offset, rule in enumerate(rules):
        if rule.name not in baseline or rule.name not in candidate:
            raise KeyError(f"missing governed metric {rule.name!r}.")
        base_values = baseline[rule.name]
        candidate_values = candidate[rule.name]
        if rule.higher_is_better:
            base_for_test = base_values
            candidate_for_test = candidate_values
        else:
            base_for_test = tuple(-float(value) for value in base_values)
            candidate_for_test = tuple(-float(value) for value in candidate_values)
        bootstrap[rule.name] = paired_bootstrap_difference(
            base_for_test,
            candidate_for_test,
            resamples=resamples,
            confidence=confidence,
            seed=seed + offset,
        )
        permutations_by_metric[rule.name] = paired_permutation_test(
            base_for_test,
            candidate_for_test,
            permutations=permutations,
            seed=seed + offset,
        )
        raw_p_values[rule.name] = permutations_by_metric[rule.name].p_value_two_sided

    adjusted = {
        row.name: row.adjusted_p_value for row in holm_adjust(raw_p_values, alpha=alpha)
    }
    results = []
    for rule in rules:
        interval = bootstrap[rule.name]
        gate = noninferiority_gate(
            estimate=interval.mean_difference,
            confidence_low=interval.confidence_low,
            confidence_high=interval.confidence_high,
            margin=rule.noninferiority_margin,
            higher_is_better=True,
        )
        corrected = adjusted[rule.name]
        significant = corrected <= alpha
        passed = gate.passed and (significant if rule.require_significance else True)
        results.append(
            MetricPromotionResult(
                name=rule.name,
                mean_difference=interval.mean_difference,
                confidence_low=interval.confidence_low,
                confidence_high=interval.confidence_high,
                raw_p_value=raw_p_values[rule.name],
                adjusted_p_value=corrected,
                noninferior=gate.passed,
                significant=significant,
                passed=passed,
            )
        )
    return PromotionDecision(all(result.passed for result in results), tuple(results))
