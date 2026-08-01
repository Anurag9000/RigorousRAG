"""Paired confidence-interval and practical-effect gates for migration reports."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from tools.migration_benchmark import DeltaInterval, PromotionBenchmarkResult
from tools.migration_promotion import PromotionReport
from tools.migration_types import digest, exact_integer, identifier, timestamp

_METRICS = (
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "support_recall",
    "citation_precision",
    "abstention_accuracy",
)


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite probability.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite probability.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be a finite probability.")
    return parsed


def _optional_gain(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _probability(value, label)


def _sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class StatisticalGatePolicy:
    policy_id: str = "paired-noninferiority-v1"
    min_repeated_runs: int = 3
    min_seed_count: int = 3
    minimum_confidence_level: float = 0.95
    max_recall_regression: float = 0.01
    max_ndcg_regression: float = 0.01
    max_mrr_regression: float = 0.01
    max_support_recall_regression: float = 0.01
    max_citation_precision_regression: float = 0.0
    max_abstention_accuracy_regression: float = 0.01
    minimum_recall_gain: float | None = None
    minimum_ndcg_gain: float | None = None
    minimum_mrr_gain: float | None = None
    minimum_support_recall_gain: float | None = None
    minimum_citation_precision_gain: float | None = None
    minimum_abstention_accuracy_gain: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", identifier(self.policy_id, "policy_id", 128))
        object.__setattr__(
            self,
            "min_repeated_runs",
            exact_integer(self.min_repeated_runs, "min_repeated_runs", 1, 10_000),
        )
        object.__setattr__(
            self,
            "min_seed_count",
            exact_integer(self.min_seed_count, "min_seed_count", 1, 10_000),
        )
        object.__setattr__(
            self,
            "minimum_confidence_level",
            _probability(self.minimum_confidence_level, "minimum_confidence_level"),
        )
        if self.minimum_confidence_level <= 0.0:
            raise ValueError("minimum_confidence_level must be positive.")
        for name in (
            "max_recall_regression",
            "max_ndcg_regression",
            "max_mrr_regression",
            "max_support_recall_regression",
            "max_citation_precision_regression",
            "max_abstention_accuracy_regression",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in (
            "minimum_recall_gain",
            "minimum_ndcg_gain",
            "minimum_mrr_gain",
            "minimum_support_recall_gain",
            "minimum_citation_precision_gain",
            "minimum_abstention_accuracy_gain",
        ):
            object.__setattr__(self, name, _optional_gain(getattr(self, name), name))

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class MetricStatisticalAssessment:
    mean_delta: float
    lower_bound: float
    upper_bound: float
    sample_count: int
    noninferiority_margin: float
    practical_gain_threshold: float | None
    noninferior: bool
    practical_gain_satisfied: bool | None

    def __post_init__(self) -> None:
        for name in ("mean_delta", "lower_bound", "upper_bound"):
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise ValueError(f"{name} must be finite and bounded.")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be finite and bounded.") from exc
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and bounded.")
            object.__setattr__(self, name, value)
        if not self.lower_bound <= self.mean_delta <= self.upper_bound:
            raise ValueError("statistical interval ordering is invalid.")
        object.__setattr__(
            self,
            "sample_count",
            exact_integer(self.sample_count, "sample_count", 1, 10_000),
        )
        object.__setattr__(
            self,
            "noninferiority_margin",
            _probability(self.noninferiority_margin, "noninferiority_margin"),
        )
        object.__setattr__(
            self,
            "practical_gain_threshold",
            _optional_gain(self.practical_gain_threshold, "practical_gain_threshold"),
        )
        if not isinstance(self.noninferior, bool):
            raise ValueError("noninferior must be boolean.")
        if self.practical_gain_satisfied is not None and not isinstance(
            self.practical_gain_satisfied, bool
        ):
            raise ValueError("practical_gain_satisfied must be boolean or null.")
        if (
            self.practical_gain_threshold is None
            and self.practical_gain_satisfied is not None
        ):
            raise ValueError("practical gain result requires a configured threshold.")
        if (
            self.practical_gain_threshold is not None
            and self.practical_gain_satisfied is None
        ):
            raise ValueError("configured practical gain requires a result.")


@dataclass(frozen=True)
class StatisticalGateAssessment:
    task_id: str
    validation_digest: str
    benchmark_fingerprint: str
    evidence_digest: str
    policy_id: str
    policy_digest: str
    confidence_level: float
    repeated_runs: int
    seed_count: int
    decision: str
    reason_codes: tuple[str, ...]
    metrics: Mapping[str, MetricStatisticalAssessment]
    evaluated_at: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        for name in (
            "validation_digest",
            "benchmark_fingerprint",
            "evidence_digest",
            "policy_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(self, "policy_id", identifier(self.policy_id, "policy_id", 128))
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        object.__setattr__(
            self,
            "repeated_runs",
            exact_integer(self.repeated_runs, "repeated_runs", 1, 10_000),
        )
        object.__setattr__(
            self,
            "seed_count",
            exact_integer(self.seed_count, "seed_count", 1, 10_000),
        )
        if self.decision not in {"passed", "blocked"}:
            raise ValueError("statistical decision is invalid.")
        reasons = tuple(identifier(item, "reason_code", 200) for item in self.reason_codes)
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("statistical reason codes must be unique and sorted.")
        object.__setattr__(self, "reason_codes", reasons)
        if self.decision == "passed" and reasons:
            raise ValueError("passed statistical assessments may not contain reasons.")
        if self.decision == "blocked" and not reasons:
            raise ValueError("blocked statistical assessments require reasons.")
        if not isinstance(self.metrics, Mapping) or set(self.metrics) != set(_METRICS):
            raise ValueError("statistical metrics must contain exactly the required metrics.")
        cleaned: dict[str, MetricStatisticalAssessment] = {}
        for name in _METRICS:
            value = self.metrics[name]
            if not isinstance(value, MetricStatisticalAssessment):
                raise ValueError("statistical metric assessment is invalid.")
            cleaned[name] = value
        object.__setattr__(self, "metrics", cleaned)
        object.__setattr__(self, "evaluated_at", timestamp(self.evaluated_at, "evaluated_at"))

    @property
    def assessment_digest(self) -> str:
        stable = asdict(self)
        stable.pop("evaluated_at", None)
        return _sha256(stable)


def _margins(policy: StatisticalGatePolicy) -> dict[str, float]:
    return {
        "recall_at_k": policy.max_recall_regression,
        "ndcg_at_k": policy.max_ndcg_regression,
        "mrr": policy.max_mrr_regression,
        "support_recall": policy.max_support_recall_regression,
        "citation_precision": policy.max_citation_precision_regression,
        "abstention_accuracy": policy.max_abstention_accuracy_regression,
    }


def _gains(policy: StatisticalGatePolicy) -> dict[str, float | None]:
    return {
        "recall_at_k": policy.minimum_recall_gain,
        "ndcg_at_k": policy.minimum_ndcg_gain,
        "mrr": policy.minimum_mrr_gain,
        "support_recall": policy.minimum_support_recall_gain,
        "citation_precision": policy.minimum_citation_precision_gain,
        "abstention_accuracy": policy.minimum_abstention_accuracy_gain,
    }


def evaluate_statistical_gate(
    benchmark: PromotionBenchmarkResult,
    *,
    policy: StatisticalGatePolicy | None = None,
    now: float | None = None,
) -> StatisticalGateAssessment:
    if not isinstance(benchmark, PromotionBenchmarkResult):
        raise ValueError("benchmark must be PromotionBenchmarkResult.")
    selected = policy or StatisticalGatePolicy()
    if not isinstance(selected, StatisticalGatePolicy):
        raise ValueError("policy must be StatisticalGatePolicy.")
    evidence = benchmark.evidence
    reasons: set[str] = set()
    if evidence.repeated_runs < selected.min_repeated_runs:
        reasons.add("paired_runs_below_minimum")
    if evidence.seed_count < selected.min_seed_count:
        reasons.add("paired_seed_count_below_minimum")
    if evidence.confidence_interval_level < selected.minimum_confidence_level:
        reasons.add("paired_confidence_level_below_minimum")

    metrics: dict[str, MetricStatisticalAssessment] = {}
    margins = _margins(selected)
    gains = _gains(selected)
    for name in _METRICS:
        interval = benchmark.delta_intervals[name]
        if not isinstance(interval, DeltaInterval):
            raise ValueError("benchmark delta interval is invalid.")
        margin = margins[name]
        gain = gains[name]
        noninferior = interval.lower >= -margin
        practical = None if gain is None else interval.lower >= gain
        if not noninferior:
            reasons.add(f"paired_{name}_noninferiority_failed")
        if practical is False:
            reasons.add(f"paired_{name}_practical_gain_failed")
        metrics[name] = MetricStatisticalAssessment(
            mean_delta=interval.mean,
            lower_bound=interval.lower,
            upper_bound=interval.upper,
            sample_count=interval.sample_count,
            noninferiority_margin=margin,
            practical_gain_threshold=gain,
            noninferior=noninferior,
            practical_gain_satisfied=practical,
        )

    ordered = tuple(sorted(reasons))
    return StatisticalGateAssessment(
        task_id=evidence.task_id,
        validation_digest=evidence.validation_digest,
        benchmark_fingerprint=evidence.benchmark_fingerprint,
        evidence_digest=evidence.evidence_digest,
        policy_id=selected.policy_id,
        policy_digest=selected.policy_digest,
        confidence_level=evidence.confidence_interval_level,
        repeated_runs=evidence.repeated_runs,
        seed_count=evidence.seed_count,
        decision="blocked" if ordered else "passed",
        reason_codes=ordered,
        metrics=metrics,
        evaluated_at=time.time() if now is None else timestamp(now, "evaluated_at"),
    )


def attach_statistical_assessment(
    report: PromotionReport,
    assessment: StatisticalGateAssessment,
) -> PromotionReport:
    if not isinstance(report, PromotionReport):
        raise ValueError("report must be PromotionReport.")
    if not isinstance(assessment, StatisticalGateAssessment):
        raise ValueError("assessment must be StatisticalGateAssessment.")
    if (
        report.task_id != assessment.task_id
        or report.validation_digest != assessment.validation_digest
        or report.benchmark_fingerprint != assessment.benchmark_fingerprint
        or report.evidence_digest != assessment.evidence_digest
    ):
        raise RuntimeError("statistical assessment does not match the promotion report.")
    reasons = tuple(sorted(set(report.reason_codes) | set(assessment.reason_codes)))
    composite_evidence_digest = _sha256(
        {
            "promotion_evidence_digest": report.evidence_digest,
            "statistical_assessment_digest": assessment.assessment_digest,
        }
    )
    composite_policy_digest = _sha256(
        {
            "promotion_policy_digest": report.policy_digest,
            "statistical_policy_digest": assessment.policy_digest,
        }
    )
    return replace(
        report,
        decision="blocked" if reasons else "eligible",
        reason_codes=reasons,
        evidence_digest=composite_evidence_digest,
        policy_id="paired-promotion-v1",
        policy_digest=composite_policy_digest,
    )


def statistical_policy_from_mapping(value: Mapping[str, Any]) -> StatisticalGatePolicy:
    if not isinstance(value, Mapping):
        raise ValueError("statistical policy must be an object.")
    allowed = set(StatisticalGatePolicy.__dataclass_fields__)
    if set(value) - allowed:
        raise ValueError("statistical policy contains unsupported fields.")
    return StatisticalGatePolicy(**dict(value))


__all__ = [
    "MetricStatisticalAssessment",
    "StatisticalGateAssessment",
    "StatisticalGatePolicy",
    "attach_statistical_assessment",
    "evaluate_statistical_gate",
    "statistical_policy_from_mapping",
]
