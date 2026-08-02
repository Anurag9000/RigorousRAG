"""Historical regression gates for evidence-graph benchmark reports."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Mapping

from tools.evidence_graph_rag_benchmark import (
    GraphRAGBenchmarkReport,
    GraphRAGBenchmarkRunReport,
)
from tools.evidence_graph_rag_evaluation import GraphRAGAggregate

_METRICS = (
    "macro_node_f1",
    "macro_document_f1",
    "macro_edge_f1",
    "complete_required_path_rate",
    "mean_lineage_completeness",
    "abstention_accuracy",
)


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and positive.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive.")
    return result


def _finite(value: Any, label: str, minimum: float = -1.0, maximum: float = 1.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and bounded.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and bounded.") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{label} must be finite and bounded.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PairedMetricInterval:
    mean_delta: float
    lower: float
    upper: float
    confidence_level: float
    pair_count: int

    def __post_init__(self) -> None:
        for name in ("mean_delta", "lower", "upper"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        if self.confidence_level <= 0:
            raise ValueError("confidence_level must be positive.")
        object.__setattr__(self, "pair_count", _integer(self.pair_count, "pair_count", 1, 10_000))
        if self.lower > self.mean_delta or self.upper < self.mean_delta:
            raise ValueError("paired interval must contain its mean.")


@dataclass(frozen=True)
class GraphRAGRegressionPolicy:
    policy_id: str = "graph-rag-conservative-v1"
    min_run_count: int = 3
    min_seed_count: int = 3
    min_cases_per_run: int = 10
    confidence_level: float = 0.95
    minimum_node_f1: float = 0.60
    minimum_document_f1: float = 0.70
    minimum_edge_f1: float = 0.50
    minimum_complete_path_rate: float = 0.50
    minimum_lineage_completeness: float = 0.95
    minimum_abstention_accuracy: float = 0.80
    max_node_f1_regression: float = 0.02
    max_document_f1_regression: float = 0.02
    max_edge_f1_regression: float = 0.03
    max_complete_path_regression: float = 0.03
    max_lineage_regression: float = 0.0
    max_abstention_regression: float = 0.02
    max_work_ratio: float = 1.50

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id", 200))
        for name in ("min_run_count", "min_seed_count", "min_cases_per_run"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 1, 1_000_000),
            )
        object.__setattr__(
            self,
            "confidence_level",
            _probability(self.confidence_level, "confidence_level"),
        )
        if self.confidence_level <= 0:
            raise ValueError("confidence_level must be positive.")
        for name in (
            "minimum_node_f1",
            "minimum_document_f1",
            "minimum_edge_f1",
            "minimum_complete_path_rate",
            "minimum_lineage_completeness",
            "minimum_abstention_accuracy",
            "max_node_f1_regression",
            "max_document_f1_regression",
            "max_edge_f1_regression",
            "max_complete_path_regression",
            "max_lineage_regression",
            "max_abstention_regression",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        object.__setattr__(self, "max_work_ratio", _positive(self.max_work_ratio, "max_work_ratio"))

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GraphRAGRegressionReport:
    benchmark_fingerprint: str
    baseline_report_digest: str
    candidate_report_digest: str
    policy_id: str
    policy_digest: str
    decision: str
    reason_codes: tuple[str, ...]
    aggregate_deltas: Mapping[str, float]
    paired_intervals: Mapping[str, PairedMetricInterval]
    work_ratio: float | None

    def __post_init__(self) -> None:
        for name in (
            "benchmark_fingerprint",
            "baseline_report_digest",
            "candidate_report_digest",
            "policy_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, "policy_id", 200))
        if self.decision not in {"eligible", "blocked"}:
            raise ValueError("decision is unsupported.")
        reasons = tuple(sorted(set(_identifier(value, "reason_code", 200) for value in self.reason_codes)))
        if self.decision == "eligible" and reasons:
            raise ValueError("eligible reports may not contain blocking reasons.")
        if self.decision == "blocked" and not reasons:
            raise ValueError("blocked reports require reason codes.")
        object.__setattr__(self, "reason_codes", reasons)
        if not isinstance(self.aggregate_deltas, Mapping) or set(self.aggregate_deltas) != set(_METRICS):
            raise ValueError("aggregate_deltas must contain every quality metric.")
        object.__setattr__(
            self,
            "aggregate_deltas",
            {name: _finite(self.aggregate_deltas[name], f"aggregate_deltas.{name}") for name in _METRICS},
        )
        if not isinstance(self.paired_intervals, Mapping) or set(self.paired_intervals) != set(_METRICS):
            raise ValueError("paired_intervals must contain every quality metric.")
        if any(not isinstance(self.paired_intervals[name], PairedMetricInterval) for name in _METRICS):
            raise ValueError("paired_intervals contain unsupported values.")
        object.__setattr__(self, "paired_intervals", dict(self.paired_intervals))
        if self.work_ratio is not None:
            object.__setattr__(self, "work_ratio", _positive(self.work_ratio, "work_ratio"))

    @property
    def report_digest(self) -> str:
        return _sha256(asdict(self))


def _interval(values: list[float], confidence_level: float) -> PairedMetricInterval:
    if not values:
        raise ValueError("paired intervals require values.")
    mean = statistics.fmean(values)
    if len(values) == 1:
        margin = 0.0
    else:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        z_score = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
        margin = z_score * standard_error
    return PairedMetricInterval(
        mean_delta=max(-1.0, min(1.0, mean)),
        lower=max(-1.0, min(1.0, mean - margin)),
        upper=max(-1.0, min(1.0, mean + margin)),
        confidence_level=confidence_level,
        pair_count=len(values),
    )


def evaluate_graph_rag_regression(
    baseline: GraphRAGBenchmarkReport,
    candidate: GraphRAGBenchmarkReport,
    policy: GraphRAGRegressionPolicy | None = None,
) -> GraphRAGRegressionReport:
    if not isinstance(baseline, GraphRAGBenchmarkReport) or not isinstance(
        candidate, GraphRAGBenchmarkReport
    ):
        raise ValueError("baseline and candidate must be GraphRAGBenchmarkReport.")
    selected = policy or GraphRAGRegressionPolicy()
    reasons: list[str] = []
    if baseline.benchmark_fingerprint != candidate.benchmark_fingerprint:
        raise ValueError("baseline and candidate benchmark fingerprints differ.")
    if (
        baseline.run_count != candidate.run_count
        or baseline.case_count_per_run != candidate.case_count_per_run
    ):
        raise ValueError("baseline and candidate benchmark dimensions differ.")
    baseline_runs = {(value.run_id, value.seed): value for value in baseline.run_reports}
    candidate_runs = {(value.run_id, value.seed): value for value in candidate.run_reports}
    if baseline_runs.keys() != candidate_runs.keys():
        raise ValueError("baseline and candidate run identities differ.")
    for key in baseline_runs:
        if baseline_runs[key].run_contract_digest != candidate_runs[key].run_contract_digest:
            raise ValueError("baseline and candidate run contracts differ.")
    if candidate.run_count < selected.min_run_count:
        reasons.append("run_count_below_minimum")
    if candidate.seed_count < selected.min_seed_count:
        reasons.append("seed_count_below_minimum")
    if candidate.case_count_per_run < selected.min_cases_per_run:
        reasons.append("case_count_below_minimum")

    floors = {
        "macro_node_f1": selected.minimum_node_f1,
        "macro_document_f1": selected.minimum_document_f1,
        "macro_edge_f1": selected.minimum_edge_f1,
        "complete_required_path_rate": selected.minimum_complete_path_rate,
        "mean_lineage_completeness": selected.minimum_lineage_completeness,
        "abstention_accuracy": selected.minimum_abstention_accuracy,
    }
    margins = {
        "macro_node_f1": selected.max_node_f1_regression,
        "macro_document_f1": selected.max_document_f1_regression,
        "macro_edge_f1": selected.max_edge_f1_regression,
        "complete_required_path_rate": selected.max_complete_path_regression,
        "mean_lineage_completeness": selected.max_lineage_regression,
        "abstention_accuracy": selected.max_abstention_regression,
    }
    deltas: dict[str, float] = {}
    intervals: dict[str, PairedMetricInterval] = {}
    for metric in _METRICS:
        candidate_value = float(getattr(candidate.aggregate, metric))
        baseline_value = float(getattr(baseline.aggregate, metric))
        deltas[metric] = candidate_value - baseline_value
        if candidate_value < floors[metric]:
            reasons.append(f"{metric}_below_floor")
        paired = [
            float(getattr(candidate_runs[key].aggregate, metric))
            - float(getattr(baseline_runs[key].aggregate, metric))
            for key in sorted(baseline_runs)
        ]
        interval = _interval(paired, selected.confidence_level)
        intervals[metric] = interval
        if interval.lower < -margins[metric]:
            reasons.append(f"{metric}_noninferiority_failed")

    baseline_work = baseline.aggregate.mean_estimated_work_units
    candidate_work = candidate.aggregate.mean_estimated_work_units
    if baseline_work == 0:
        work_ratio = 1.0 if candidate_work == 0 else None
    else:
        work_ratio = candidate_work / baseline_work
    if work_ratio is None or work_ratio > selected.max_work_ratio:
        reasons.append("estimated_work_ratio_exceeds_limit")
    unique_reasons = tuple(sorted(set(reasons)))
    return GraphRAGRegressionReport(
        benchmark_fingerprint=candidate.benchmark_fingerprint,
        baseline_report_digest=baseline.report_digest,
        candidate_report_digest=candidate.report_digest,
        policy_id=selected.policy_id,
        policy_digest=selected.policy_digest,
        decision="eligible" if not unique_reasons else "blocked",
        reason_codes=unique_reasons,
        aggregate_deltas=deltas,
        paired_intervals=intervals,
        work_ratio=work_ratio,
    )


def _aggregate_from_mapping(value: Mapping[str, Any]) -> GraphRAGAggregate:
    if not isinstance(value, Mapping):
        raise ValueError("aggregate must be an object.")
    return GraphRAGAggregate(**value)


def report_from_mapping(value: Mapping[str, Any]) -> GraphRAGBenchmarkReport:
    allowed = {
        "benchmark_id",
        "benchmark_fingerprint",
        "run_count",
        "seed_count",
        "case_count_per_run",
        "run_reports",
        "aggregate",
        "schema_version",
        "report_digest",
        "contains_raw_query",
        "contains_evidence_text",
    }
    if not isinstance(value, Mapping) or not set(value) <= allowed or not {
        "benchmark_id",
        "benchmark_fingerprint",
        "run_count",
        "seed_count",
        "case_count_per_run",
        "run_reports",
        "aggregate",
        "schema_version",
    } <= set(value):
        raise ValueError("graph RAG benchmark report schema is invalid.")
    if value.get("contains_raw_query") not in (None, False) or value.get(
        "contains_evidence_text"
    ) not in (None, False):
        raise ValueError("benchmark reports may not claim raw-query or evidence text.")
    if not isinstance(value["run_reports"], list):
        raise ValueError("run_reports must be a JSON array.")
    runs = tuple(
        GraphRAGBenchmarkRunReport(
            run_id=item["run_id"],
            seed=item["seed"],
            aggregate=_aggregate_from_mapping(item["aggregate"]),
            run_contract_digest=item["run_contract_digest"],
            run_result_digest=item["run_result_digest"],
        )
        for item in value["run_reports"]
    )
    report = GraphRAGBenchmarkReport(
        benchmark_id=value["benchmark_id"],
        benchmark_fingerprint=value["benchmark_fingerprint"],
        run_count=value["run_count"],
        seed_count=value["seed_count"],
        case_count_per_run=value["case_count_per_run"],
        run_reports=runs,
        aggregate=_aggregate_from_mapping(value["aggregate"]),
        schema_version=value["schema_version"],
    )
    if value.get("report_digest") is not None and value["report_digest"] != report.report_digest:
        raise ValueError("benchmark report digest is invalid.")
    return report


def policy_from_mapping(value: Mapping[str, Any]) -> GraphRAGRegressionPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("policy must be an object.")
    allowed = set(GraphRAGRegressionPolicy.__dataclass_fields__)
    if not set(value) <= allowed:
        raise ValueError("regression policy contains unknown fields.")
    return GraphRAGRegressionPolicy(**dict(value))


__all__ = [
    "GraphRAGRegressionPolicy",
    "GraphRAGRegressionReport",
    "PairedMetricInterval",
    "evaluate_graph_rag_regression",
    "policy_from_mapping",
    "report_from_mapping",
]
