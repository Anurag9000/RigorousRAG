"""Governed promotion, distribution-shift and rollback decisions for adaptive routing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tools.adaptive_route_experiments import ROUTES, RouteBenchmarkReport

_MAX_CASES = 10_000


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


def _nonnegative(value: Any, label: str, maximum: float = 1_000_000_000.0) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= maximum:
        raise ValueError(f"{label} must be non-negative and bounded.")
    return result


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def route_distribution(report: RouteBenchmarkReport) -> dict[str, float]:
    """Return the selected-route distribution for one bounded benchmark report."""

    if not isinstance(report, RouteBenchmarkReport):
        raise ValueError("report must be RouteBenchmarkReport.")
    if report.case_count != len(report.cases) or report.case_count > _MAX_CASES:
        raise ValueError("route benchmark case accounting is invalid.")
    counts = {route: 0 for route in ROUTES}
    for case in report.cases:
        if case.selected_route not in counts:
            raise ValueError("route benchmark contains an unsupported selected route.")
        counts[case.selected_route] += 1
    denominator = max(report.case_count, 1)
    return {route: counts[route] / denominator for route in ROUTES}


def jensen_shannon_divergence(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> float:
    """Compute base-2 Jensen-Shannon divergence for normalized categorical maps."""

    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("distributions must be mappings.")
    keys = sorted(set(left) | set(right))
    if not keys or len(keys) > 1000:
        raise ValueError("distributions are empty or exceed the category limit.")
    left_values = []
    right_values = []
    for key in keys:
        if not isinstance(key, str) or not key or len(key) > 200:
            raise ValueError("distribution categories are invalid.")
        left_values.append(_nonnegative(left.get(key, 0.0), "left probability", 1.0))
        right_values.append(_nonnegative(right.get(key, 0.0), "right probability", 1.0))
    left_total, right_total = sum(left_values), sum(right_values)
    if left_total <= 0.0 or right_total <= 0.0:
        raise ValueError("distributions must each contain positive mass.")
    p = [value / left_total for value in left_values]
    q = [value / right_total for value in right_values]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]

    def divergence(values: list[float]) -> float:
        return sum(
            value * math.log2(value / middle)
            for value, middle in zip(values, midpoint)
            if value > 0.0 and middle > 0.0
        )

    return max(0.0, min((divergence(p) + divergence(q)) / 2.0, 1.0))


@dataclass(frozen=True)
class AdaptivePolicyComparison:
    baseline_policy_id: str
    candidate_policy_id: str
    case_count: int
    success_rate_delta: float
    route_accuracy_delta: float
    regret_delta: float
    cost_delta: float
    latency_delta_ms: float
    route_shift_jsd: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "baseline_policy_id",
            _identifier(self.baseline_policy_id, "baseline_policy_id"),
        )
        object.__setattr__(
            self,
            "candidate_policy_id",
            _identifier(self.candidate_policy_id, "candidate_policy_id"),
        )
        if isinstance(self.case_count, bool) or not isinstance(self.case_count, int):
            raise ValueError("case_count must be an integer.")
        if not 1 <= self.case_count <= _MAX_CASES:
            raise ValueError("case_count is outside the supported range.")
        for name in (
            "success_rate_delta",
            "route_accuracy_delta",
            "regret_delta",
            "cost_delta",
            "latency_delta_ms",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "route_shift_jsd",
            _unit(self.route_shift_jsd, "route_shift_jsd"),
        )

    @property
    def comparison_digest(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class AdaptivePolicyGate:
    min_success_rate_delta: float = 0.0
    min_route_accuracy_delta: float = -0.02
    max_regret_delta: float = 0.01
    max_cost_delta: float = 50.0
    max_latency_delta_ms: float = 250.0
    max_route_shift_jsd: float = 0.20

    def __post_init__(self) -> None:
        for name in (
            "min_success_rate_delta",
            "min_route_accuracy_delta",
            "max_regret_delta",
            "max_cost_delta",
            "max_latency_delta_ms",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "max_route_shift_jsd",
            _unit(self.max_route_shift_jsd, "max_route_shift_jsd"),
        )


@dataclass(frozen=True)
class AdaptivePolicyDecision:
    comparison: AdaptivePolicyComparison
    gate: AdaptivePolicyGate
    decision: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.comparison, AdaptivePolicyComparison):
            raise ValueError("comparison must be AdaptivePolicyComparison.")
        if not isinstance(self.gate, AdaptivePolicyGate):
            raise ValueError("gate must be AdaptivePolicyGate.")
        if self.decision not in {"eligible", "hold", "rollback"}:
            raise ValueError("adaptive policy decision is unsupported.")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(reason, str) or not reason or len(reason) > 200
            for reason in self.reasons
        ):
            raise ValueError("adaptive policy decision reasons are invalid.")
        if self.decision == "eligible" and self.reasons:
            raise ValueError("eligible policy decisions may not contain failure reasons.")
        if self.decision != "eligible" and not self.reasons:
            raise ValueError("held or rollback decisions require reasons.")

    @property
    def decision_digest(self) -> str:
        return _digest(
            {
                "comparison_digest": self.comparison.comparison_digest,
                "gate": asdict(self.gate),
                "decision": self.decision,
                "reasons": self.reasons,
            }
        )


def compare_adaptive_policies(
    *,
    baseline_policy_id: str,
    candidate_policy_id: str,
    baseline: RouteBenchmarkReport,
    candidate: RouteBenchmarkReport,
) -> AdaptivePolicyComparison:
    """Compare paired benchmark reports and refuse unpaired case sets."""

    if not isinstance(baseline, RouteBenchmarkReport) or not isinstance(
        candidate, RouteBenchmarkReport
    ):
        raise ValueError("baseline and candidate must be RouteBenchmarkReport values.")
    baseline_ids = tuple(case.case_id for case in baseline.cases)
    candidate_ids = tuple(case.case_id for case in candidate.cases)
    if baseline_ids != candidate_ids or not baseline_ids:
        raise ValueError("policy comparison requires identical ordered benchmark cases.")
    if baseline.case_count != len(baseline_ids) or candidate.case_count != len(candidate_ids):
        raise ValueError("policy benchmark case accounting is invalid.")
    return AdaptivePolicyComparison(
        baseline_policy_id=baseline_policy_id,
        candidate_policy_id=candidate_policy_id,
        case_count=len(baseline_ids),
        success_rate_delta=candidate.selected_success_rate - baseline.selected_success_rate,
        route_accuracy_delta=(
            candidate.route_selection_accuracy - baseline.route_selection_accuracy
        ),
        regret_delta=candidate.mean_regret - baseline.mean_regret,
        cost_delta=(
            candidate.mean_selected_cost_units - baseline.mean_selected_cost_units
        ),
        latency_delta_ms=(
            candidate.mean_selected_latency_ms - baseline.mean_selected_latency_ms
        ),
        route_shift_jsd=jensen_shannon_divergence(
            route_distribution(baseline), route_distribution(candidate)
        ),
    )


def _violations(
    comparison: AdaptivePolicyComparison,
    gate: AdaptivePolicyGate,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if comparison.success_rate_delta < gate.min_success_rate_delta:
        reasons.append("success_rate_regressed")
    if comparison.route_accuracy_delta < gate.min_route_accuracy_delta:
        reasons.append("route_accuracy_regressed")
    if comparison.regret_delta > gate.max_regret_delta:
        reasons.append("regret_increased")
    if comparison.cost_delta > gate.max_cost_delta:
        reasons.append("cost_increased")
    if comparison.latency_delta_ms > gate.max_latency_delta_ms:
        reasons.append("latency_increased")
    if comparison.route_shift_jsd > gate.max_route_shift_jsd:
        reasons.append("route_distribution_shifted")
    return tuple(reasons)


def evaluate_adaptive_policy_promotion(
    comparison: AdaptivePolicyComparison,
    *,
    gate: AdaptivePolicyGate | None = None,
) -> AdaptivePolicyDecision:
    selected_gate = gate or AdaptivePolicyGate()
    if not isinstance(comparison, AdaptivePolicyComparison):
        raise ValueError("comparison must be AdaptivePolicyComparison.")
    if not isinstance(selected_gate, AdaptivePolicyGate):
        raise ValueError("gate must be AdaptivePolicyGate.")
    reasons = _violations(comparison, selected_gate)
    return AdaptivePolicyDecision(
        comparison=comparison,
        gate=selected_gate,
        decision="hold" if reasons else "eligible",
        reasons=reasons,
    )


def evaluate_adaptive_policy_rollback(
    comparison: AdaptivePolicyComparison,
    *,
    gate: AdaptivePolicyGate | None = None,
) -> AdaptivePolicyDecision:
    """Recommend rollback when an online/promoted candidate breaches the same gates."""

    selected_gate = gate or AdaptivePolicyGate()
    if not isinstance(comparison, AdaptivePolicyComparison):
        raise ValueError("comparison must be AdaptivePolicyComparison.")
    if not isinstance(selected_gate, AdaptivePolicyGate):
        raise ValueError("gate must be AdaptivePolicyGate.")
    reasons = _violations(comparison, selected_gate)
    return AdaptivePolicyDecision(
        comparison=comparison,
        gate=selected_gate,
        decision="rollback" if reasons else "eligible",
        reasons=reasons,
    )


__all__ = [
    "AdaptivePolicyComparison",
    "AdaptivePolicyDecision",
    "AdaptivePolicyGate",
    "compare_adaptive_policies",
    "evaluate_adaptive_policy_promotion",
    "evaluate_adaptive_policy_rollback",
    "jensen_shannon_divergence",
    "route_distribution",
]
