"""Governed paired benchmark producer for migration promotion evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from tools.migration_promotion import (
    PromotionEvidence,
    ResourceMetrics,
    RetrievalMetrics,
)
from tools.migration_types import digest, exact_integer, identifier

_MAX_RUNS = 10_000
_MAX_CASES = 100_000
_MAX_RANKED = 10_000
_Z_95 = 1.959963984540054
_METRICS = (
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "support_recall",
    "citation_precision",
    "abstention_accuracy",
)


def _bounded_ids(value: Any, label: str, maximum: int) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence of identifiers.")
    try:
        items = tuple(value)
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    if len(items) > maximum:
        raise ValueError(f"{label} exceeds the identifier limit.")
    normalized = tuple(identifier(item, label, 256) for item in items)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} may not contain duplicates.")
    return normalized


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BenchmarkCase:
    query_id: str
    relevant_ids: tuple[str, ...]
    current_ranked_ids: tuple[str, ...]
    shadow_ranked_ids: tuple[str, ...]
    support_total: int
    current_support_found: int
    shadow_support_found: int
    current_citation_count: int
    current_valid_citation_count: int
    shadow_citation_count: int
    shadow_valid_citation_count: int
    should_abstain: bool
    current_abstained: bool
    shadow_abstained: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", identifier(self.query_id, "query_id", 256))
        object.__setattr__(
            self,
            "relevant_ids",
            _bounded_ids(self.relevant_ids, "relevant_ids", _MAX_RANKED),
        )
        object.__setattr__(
            self,
            "current_ranked_ids",
            _bounded_ids(self.current_ranked_ids, "current_ranked_ids", _MAX_RANKED),
        )
        object.__setattr__(
            self,
            "shadow_ranked_ids",
            _bounded_ids(self.shadow_ranked_ids, "shadow_ranked_ids", _MAX_RANKED),
        )
        for name in (
            "support_total",
            "current_support_found",
            "shadow_support_found",
            "current_citation_count",
            "current_valid_citation_count",
            "shadow_citation_count",
            "shadow_valid_citation_count",
        ):
            object.__setattr__(
                self,
                name,
                exact_integer(getattr(self, name), name, 0, 100_000_000),
            )
        if self.current_support_found > self.support_total:
            raise ValueError("current_support_found exceeds support_total.")
        if self.shadow_support_found > self.support_total:
            raise ValueError("shadow_support_found exceeds support_total.")
        if self.current_valid_citation_count > self.current_citation_count:
            raise ValueError("current valid citations exceed citation count.")
        if self.shadow_valid_citation_count > self.shadow_citation_count:
            raise ValueError("shadow valid citations exceed citation count.")
        for name in ("should_abstain", "current_abstained", "shadow_abstained"):
            object.__setattr__(self, name, _boolean(getattr(self, name), name))
        if not self.relevant_ids and not self.should_abstain:
            raise ValueError("non-abstention cases require relevant identifiers.")

    @property
    def contract_identity(self) -> tuple[Any, ...]:
        return (
            self.query_id,
            self.relevant_ids,
            self.support_total,
            self.should_abstain,
        )


@dataclass(frozen=True)
class BenchmarkRun:
    seed: int
    cases: tuple[BenchmarkCase, ...]
    current_resources: ResourceMetrics
    shadow_resources: ResourceMetrics

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", exact_integer(self.seed, "seed", 0, 2**31 - 1))
        if isinstance(self.cases, (str, bytes, bytearray)):
            raise ValueError("cases must be a sequence.")
        try:
            cases = tuple(self.cases)
        except Exception as exc:
            raise ValueError("cases must be safely iterable.") from exc
        if not cases or len(cases) > _MAX_CASES:
            raise ValueError("cases must contain a bounded non-empty sequence.")
        if any(not isinstance(case, BenchmarkCase) for case in cases):
            raise ValueError("every case must be BenchmarkCase.")
        if len({case.query_id for case in cases}) != len(cases):
            raise ValueError("query_id values must be unique within a run.")
        object.__setattr__(self, "cases", cases)
        if not isinstance(self.current_resources, ResourceMetrics):
            raise ValueError("current_resources must be ResourceMetrics.")
        if not isinstance(self.shadow_resources, ResourceMetrics):
            raise ValueError("shadow_resources must be ResourceMetrics.")


@dataclass(frozen=True)
class PromotionBenchmarkFixture:
    task_id: str
    validation_digest: str
    source_sequence: int
    source_content_sha256: str
    vector_count: int
    sparse_count: int
    rank_cutoff: int
    runs: tuple[BenchmarkRun, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(
            self,
            "validation_digest",
            digest(self.validation_digest, "validation_digest"),
        )
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "source_content_sha256",
            digest(self.source_content_sha256, "source_content_sha256"),
        )
        object.__setattr__(
            self,
            "vector_count",
            exact_integer(self.vector_count, "vector_count", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "sparse_count",
            exact_integer(self.sparse_count, "sparse_count", 1, 100_000_000),
        )
        object.__setattr__(
            self,
            "rank_cutoff",
            exact_integer(self.rank_cutoff, "rank_cutoff", 1, _MAX_RANKED),
        )
        if isinstance(self.runs, (str, bytes, bytearray)):
            raise ValueError("runs must be a sequence.")
        try:
            runs = tuple(self.runs)
        except Exception as exc:
            raise ValueError("runs must be safely iterable.") from exc
        if not runs or len(runs) > _MAX_RUNS:
            raise ValueError("runs must contain a bounded non-empty sequence.")
        if any(not isinstance(run, BenchmarkRun) for run in runs):
            raise ValueError("every run must be BenchmarkRun.")
        baseline = tuple(case.contract_identity for case in runs[0].cases)
        for run in runs[1:]:
            if tuple(case.contract_identity for case in run.cases) != baseline:
                raise ValueError("all runs must use the same ordered benchmark contract.")
        object.__setattr__(self, "runs", runs)

    @property
    def benchmark_fingerprint(self) -> str:
        contract = {
            "rank_cutoff": self.rank_cutoff,
            "runs": [
                {
                    "seed": run.seed,
                    "cases": [
                        {
                            "query_id": case.query_id,
                            "relevant_ids": list(case.relevant_ids),
                            "support_total": case.support_total,
                            "should_abstain": case.should_abstain,
                        }
                        for case in run.cases
                    ],
                }
                for run in self.runs
            ],
        }
        return _canonical_digest(contract)


@dataclass(frozen=True)
class MetricInterval:
    mean: float
    lower: float
    upper: float
    sample_count: int

    def __post_init__(self) -> None:
        for name in ("mean", "lower", "upper"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("metric interval values must be finite probabilities.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "sample_count",
            exact_integer(self.sample_count, "sample_count", 1, _MAX_RUNS),
        )
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("metric interval ordering is invalid.")


@dataclass(frozen=True)
class DeltaInterval:
    mean: float
    lower: float
    upper: float
    sample_count: int

    def __post_init__(self) -> None:
        for name in ("mean", "lower", "upper"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("delta interval values must be finite and bounded.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "sample_count",
            exact_integer(self.sample_count, "sample_count", 1, _MAX_RUNS),
        )
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("delta interval ordering is invalid.")


@dataclass(frozen=True)
class PromotionBenchmarkResult:
    evidence: PromotionEvidence
    current_intervals: Mapping[str, MetricInterval]
    shadow_intervals: Mapping[str, MetricInterval]
    delta_intervals: Mapping[str, DeltaInterval]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, PromotionEvidence):
            raise ValueError("evidence must be PromotionEvidence.")
        for name, value in (
            ("current_intervals", self.current_intervals),
            ("shadow_intervals", self.shadow_intervals),
            ("delta_intervals", self.delta_intervals),
        ):
            if not isinstance(value, Mapping) or set(value) != set(_METRICS):
                raise ValueError(f"{name} must contain exactly the benchmark metrics.")
            expected_type = DeltaInterval if name == "delta_intervals" else MetricInterval
            if any(not isinstance(item, expected_type) for item in value.values()):
                raise ValueError(f"{name} values have an invalid interval type.")


def _recall(relevant: tuple[str, ...], ranked: tuple[str, ...], cutoff: int) -> float:
    if not relevant:
        return 1.0
    found = len(set(relevant) & set(ranked[:cutoff]))
    return found / len(relevant)


def _mrr(relevant: tuple[str, ...], ranked: tuple[str, ...], cutoff: int) -> float:
    if not relevant:
        return 1.0
    targets = set(relevant)
    for position, item in enumerate(ranked[:cutoff], start=1):
        if item in targets:
            return 1.0 / position
    return 0.0


def _ndcg(relevant: tuple[str, ...], ranked: tuple[str, ...], cutoff: int) -> float:
    if not relevant:
        return 1.0
    targets = set(relevant)
    dcg = sum(
        (1.0 / math.log2(position + 1))
        for position, item in enumerate(ranked[:cutoff], start=1)
        if item in targets
    )
    ideal_hits = min(len(relevant), cutoff)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return 0.0 if ideal == 0.0 else dcg / ideal


def _run_metrics(run: BenchmarkRun, cutoff: int, *, shadow: bool) -> RetrievalMetrics:
    retrieval_cases = [case for case in run.cases if case.relevant_ids]
    ranked_name = "shadow_ranked_ids" if shadow else "current_ranked_ids"
    recalls = [
        _recall(case.relevant_ids, getattr(case, ranked_name), cutoff)
        for case in retrieval_cases
    ]
    ndcgs = [
        _ndcg(case.relevant_ids, getattr(case, ranked_name), cutoff)
        for case in retrieval_cases
    ]
    mrrs = [
        _mrr(case.relevant_ids, getattr(case, ranked_name), cutoff)
        for case in retrieval_cases
    ]
    support_found = sum(
        case.shadow_support_found if shadow else case.current_support_found
        for case in run.cases
    )
    support_total = sum(case.support_total for case in run.cases)
    citation_count = sum(
        case.shadow_citation_count if shadow else case.current_citation_count
        for case in run.cases
    )
    valid_citations = sum(
        case.shadow_valid_citation_count if shadow else case.current_valid_citation_count
        for case in run.cases
    )
    abstention_correct = sum(
        (case.shadow_abstained if shadow else case.current_abstained)
        == case.should_abstain
        for case in run.cases
    )
    return RetrievalMetrics(
        query_count=len(run.cases),
        recall_at_k=fmean(recalls) if recalls else 1.0,
        ndcg_at_k=fmean(ndcgs) if ndcgs else 1.0,
        mrr=fmean(mrrs) if mrrs else 1.0,
        support_recall=(support_found / support_total if support_total else 1.0),
        citation_precision=(valid_citations / citation_count if citation_count else 0.0),
        abstention_accuracy=abstention_correct / len(run.cases),
    )


def _interval(values: Sequence[float]) -> MetricInterval:
    if not values:
        raise ValueError("interval values may not be empty.")
    mean = fmean(values)
    if len(values) == 1:
        lower = upper = mean
    else:
        margin = _Z_95 * stdev(values) / math.sqrt(len(values))
        lower = mean - margin
        upper = mean + margin
    return MetricInterval(
        mean=min(1.0, max(0.0, mean)),
        lower=min(1.0, max(0.0, lower)),
        upper=min(1.0, max(0.0, upper)),
        sample_count=len(values),
    )


def _delta_interval(values: Sequence[float]) -> DeltaInterval:
    if not values:
        raise ValueError("delta interval values may not be empty.")
    mean = fmean(values)
    if len(values) == 1:
        lower = upper = mean
    else:
        margin = _Z_95 * stdev(values) / math.sqrt(len(values))
        lower = mean - margin
        upper = mean + margin
    return DeltaInterval(
        mean=min(1.0, max(-1.0, mean)),
        lower=min(1.0, max(-1.0, lower)),
        upper=min(1.0, max(-1.0, upper)),
        sample_count=len(values),
    )


def _aggregate_resources(runs: tuple[BenchmarkRun, ...], *, shadow: bool) -> ResourceMetrics:
    selected = [run.shadow_resources if shadow else run.current_resources for run in runs]
    return ResourceMetrics(
        p95_latency_ms=max(item.p95_latency_ms for item in selected),
        peak_memory_bytes=max(item.peak_memory_bytes for item in selected),
        index_bytes=max(item.index_bytes for item in selected),
        estimated_cost_units=fmean(item.estimated_cost_units for item in selected),
    )


def run_promotion_benchmark(fixture: PromotionBenchmarkFixture) -> PromotionBenchmarkResult:
    if not isinstance(fixture, PromotionBenchmarkFixture):
        raise ValueError("fixture must be PromotionBenchmarkFixture.")
    current_runs = tuple(
        _run_metrics(run, fixture.rank_cutoff, shadow=False) for run in fixture.runs
    )
    shadow_runs = tuple(
        _run_metrics(run, fixture.rank_cutoff, shadow=True) for run in fixture.runs
    )

    def aggregate(name: str, values: tuple[RetrievalMetrics, ...]) -> float:
        return fmean(getattr(item, name) for item in values)

    current_quality = RetrievalMetrics(
        query_count=current_runs[0].query_count,
        **{name: aggregate(name, current_runs) for name in _METRICS},
    )
    shadow_quality = RetrievalMetrics(
        query_count=shadow_runs[0].query_count,
        **{name: aggregate(name, shadow_runs) for name in _METRICS},
    )
    current_intervals = {
        name: _interval([getattr(item, name) for item in current_runs])
        for name in _METRICS
    }
    shadow_intervals = {
        name: _interval([getattr(item, name) for item in shadow_runs])
        for name in _METRICS
    }
    delta_intervals = {
        name: _delta_interval(
            [
                getattr(shadow_item, name) - getattr(current_item, name)
                for current_item, shadow_item in zip(current_runs, shadow_runs, strict=True)
            ]
        )
        for name in _METRICS
    }
    evidence = PromotionEvidence(
        task_id=fixture.task_id,
        validation_digest=fixture.validation_digest,
        benchmark_fingerprint=fixture.benchmark_fingerprint,
        source_sequence=fixture.source_sequence,
        source_content_sha256=fixture.source_content_sha256,
        vector_count=fixture.vector_count,
        sparse_count=fixture.sparse_count,
        repeated_runs=len(fixture.runs),
        seed_count=len({run.seed for run in fixture.runs}),
        confidence_interval_level=0.95,
        current_quality=current_quality,
        shadow_quality=shadow_quality,
        current_resources=_aggregate_resources(fixture.runs, shadow=False),
        shadow_resources=_aggregate_resources(fixture.runs, shadow=True),
    )
    return PromotionBenchmarkResult(
        evidence=evidence,
        current_intervals=current_intervals,
        shadow_intervals=shadow_intervals,
        delta_intervals=delta_intervals,
    )


def fixture_from_mapping(value: Mapping[str, Any]) -> PromotionBenchmarkFixture:
    if not isinstance(value, Mapping):
        raise ValueError("benchmark fixture must be an object.")
    expected = {
        "task_id",
        "validation_digest",
        "source_sequence",
        "source_content_sha256",
        "vector_count",
        "sparse_count",
        "rank_cutoff",
        "runs",
    }
    if set(value) != expected:
        raise ValueError("benchmark fixture fields are incomplete or unsupported.")
    raw_runs = value["runs"]
    if isinstance(raw_runs, (str, bytes, bytearray)):
        raise ValueError("runs must be a sequence.")
    runs: list[BenchmarkRun] = []
    for raw_run in raw_runs:
        if not isinstance(raw_run, Mapping) or set(raw_run) != {
            "seed",
            "cases",
            "current_resources",
            "shadow_resources",
        }:
            raise ValueError("benchmark run fields are invalid.")
        cases: list[BenchmarkCase] = []
        for raw_case in raw_run["cases"]:
            if not isinstance(raw_case, Mapping):
                raise ValueError("benchmark cases must be objects.")
            cases.append(BenchmarkCase(**dict(raw_case)))
        runs.append(
            BenchmarkRun(
                seed=raw_run["seed"],
                cases=tuple(cases),
                current_resources=ResourceMetrics(**raw_run["current_resources"]),
                shadow_resources=ResourceMetrics(**raw_run["shadow_resources"]),
            )
        )
    return PromotionBenchmarkFixture(
        task_id=value["task_id"],
        validation_digest=value["validation_digest"],
        source_sequence=value["source_sequence"],
        source_content_sha256=value["source_content_sha256"],
        vector_count=value["vector_count"],
        sparse_count=value["sparse_count"],
        rank_cutoff=value["rank_cutoff"],
        runs=tuple(runs),
    )


__all__ = [
    "BenchmarkCase",
    "BenchmarkRun",
    "DeltaInterval",
    "MetricInterval",
    "PromotionBenchmarkFixture",
    "PromotionBenchmarkResult",
    "fixture_from_mapping",
    "run_promotion_benchmark",
]
