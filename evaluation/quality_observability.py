"""Deterministic, privacy-safe observability contract for retrieval and RAG quality.

Metric algorithms remain in their owning evaluation modules. This module only normalizes
their scalar outputs, binds immutable provenance, evaluates SLOs, compares like-for-like
snapshots, and writes canonical machine-readable artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evaluation.benchmark_suite import BenchmarkSuiteResult
from evaluation.conformal_retrieval import SelectiveRiskMetrics
from evaluation.drift import DriftReport
from evaluation.efficiency import LatencySummary
from evaluation.resource_measurement import ResourceUsage
from evaluation.semantic_support import CitationSupportMetrics, SemanticMetrics

SCHEMA_VERSION = 1
_MAX_METRICS = 100_000
_MAX_SLOS = 10_000
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@:/+-]{0,299}$")
_DIRECTIONS = frozenset({"higher", "lower", "neutral"})
_COMPARATORS = frozenset({">=", "<=", ">", "<"})
_SAFE_TAGS = frozenset(
    {
        "stage", "route", "language", "domain", "split", "cohort", "variant",
        "metric_family", "k", "provider", "device", "precision",
    }
)
_SAFE_TAG_SUFFIXES = ("_id", "_digest", "_sha256", "_version")
_CONTENT_FRAGMENTS = (
    "query", "prompt", "answer", "content", "text", "passage", "evidence",
    "document", "chunk_text",
)


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    value = value.strip()
    if not value or len(value) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{label} is invalid.")
    return value


def _metric_name(value: Any, label: str = "metric name") -> str:
    value = _identifier(value, label, 300)
    if not _NAME.fullmatch(value):
        raise ValueError(f"{label} contains unsupported characters.")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def _count(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return value


def _sha256(value: Any, label: str) -> str:
    value = _identifier(value, label, 64).lower()
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _tag_pairs(
    value: Mapping[str, str] | Sequence[tuple[str, str]] | None,
    *,
    label: str = "tags",
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        rows = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rows = tuple(value)
    else:
        raise ValueError(f"{label} must be a mapping or key/value sequence.")
    if len(rows) > 100:
        raise ValueError(f"{label} exceeds the dimension limit.")
    cleaned: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) or len(row) != 2:
            raise ValueError(f"{label} entries must be key/value pairs.")
        key = _identifier(row[0], f"{label} key", 100).lower()
        if any(fragment in key for fragment in _CONTENT_FRAGMENTS):
            raise ValueError(f"{label} key {key!r} may expose raw content.")
        if key not in _SAFE_TAGS and not key.endswith(_SAFE_TAG_SUFFIXES):
            raise ValueError(f"{label} key {key!r} is not an approved non-content dimension.")
        selected = _identifier(row[1], f"{label} value", 256)
        if key in cleaned and cleaned[key] != selected:
            raise ValueError(f"{label} contains conflicting duplicate keys.")
        cleaned[key] = selected
    return tuple(sorted(cleaned.items()))


@dataclass(frozen=True)
class QualityWindow:
    started_at_unix: float
    ended_at_unix: float
    generated_at_unix: float

    def __post_init__(self) -> None:
        start = _finite(self.started_at_unix, "started_at_unix")
        end = _finite(self.ended_at_unix, "ended_at_unix")
        generated = _finite(self.generated_at_unix, "generated_at_unix")
        if min(start, end, generated) < 0.0:
            raise ValueError("quality-window timestamps must be non-negative.")
        if end < start:
            raise ValueError("ended_at_unix must be >= started_at_unix.")
        if generated < end:
            raise ValueError("generated_at_unix must be >= ended_at_unix.")
        object.__setattr__(self, "started_at_unix", start)
        object.__setattr__(self, "ended_at_unix", end)
        object.__setattr__(self, "generated_at_unix", generated)

    def to_dict(self) -> dict[str, float]:
        return {
            "started_at_unix": self.started_at_unix,
            "ended_at_unix": self.ended_at_unix,
            "generated_at_unix": self.generated_at_unix,
        }


@dataclass(frozen=True)
class QualityProvenance:
    run_id: str
    system_id: str
    domain_id: str
    dataset_manifest_digest: str
    split_digest: str
    evaluation_contract_digest: str
    code_revision: str
    retrieval_stack_digest: str | None = None
    model_digest: str | None = None
    environment_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "system_id", "domain_id", "code_revision"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        for name in ("dataset_manifest_digest", "split_digest", "evaluation_contract_digest"):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        for name in ("retrieval_stack_digest", "model_digest", "environment_digest"):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else _sha256(value, name))

    @property
    def comparison_scope_digest(self) -> str:
        return canonical_digest(
            {
                "system_id": self.system_id,
                "domain_id": self.domain_id,
                "dataset_manifest_digest": self.dataset_manifest_digest,
                "split_digest": self.split_digest,
                "evaluation_contract_digest": self.evaluation_contract_digest,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "system_id": self.system_id,
            "domain_id": self.domain_id,
            "dataset_manifest_digest": self.dataset_manifest_digest,
            "split_digest": self.split_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "code_revision": self.code_revision,
            "retrieval_stack_digest": self.retrieval_stack_digest,
            "model_digest": self.model_digest,
            "environment_digest": self.environment_digest,
            "comparison_scope_digest": self.comparison_scope_digest,
        }


@dataclass(frozen=True)
class MetricObservation:
    name: str
    value: float
    direction: str
    unit: str
    sample_count: int
    source: str
    tags: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _metric_name(self.name))
        object.__setattr__(self, "value", _finite(self.value, f"{self.name} value"))
        direction = _identifier(self.direction, "direction", 20).lower()
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}.")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "unit", _identifier(self.unit, "unit", 100))
        object.__setattr__(self, "sample_count", _count(self.sample_count, "sample_count"))
        object.__setattr__(self, "source", _identifier(self.source, "source", 300))
        object.__setattr__(self, "tags", _tag_pairs(self.tags))

    @property
    def identity(self) -> str:
        # Source is deliberately excluded: the public metric contract is name + dimensions.
        return canonical_digest({"name": self.name, "tags": list(self.tags)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "direction": self.direction,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "source": self.source,
            "tags": dict(self.tags),
            "identity": self.identity,
        }


@dataclass(frozen=True)
class QualitySnapshot:
    window: QualityWindow
    provenance: QualityProvenance
    metrics: tuple[MetricObservation, ...]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
        if not isinstance(self.window, QualityWindow):
            raise ValueError("window must be QualityWindow.")
        if not isinstance(self.provenance, QualityProvenance):
            raise ValueError("provenance must be QualityProvenance.")
        metrics = tuple(self.metrics)
        if len(metrics) > _MAX_METRICS or any(not isinstance(row, MetricObservation) for row in metrics):
            raise ValueError("metrics must be a bounded MetricObservation sequence.")
        metrics = tuple(sorted(metrics, key=lambda row: (row.name, row.tags)))
        identities = [row.identity for row in metrics]
        if len(identities) != len(set(identities)):
            raise ValueError(
                "snapshot contains duplicate metric identities; aggregate in the owning "
                "metric module before observability export."
            )
        object.__setattr__(self, "metrics", metrics)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "window": self.window.to_dict(),
            "provenance": self.provenance.to_dict(),
            "metrics": [row.to_dict() for row in self.metrics],
        }

    @property
    def snapshot_digest(self) -> str:
        return canonical_digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        value = self._payload()
        value["snapshot_digest"] = self.snapshot_digest
        return value


@dataclass(frozen=True)
class QualitySLO:
    name: str
    metric_name: str
    comparator: str
    threshold: float
    required: bool = True
    tag_match: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "SLO name", 300))
        object.__setattr__(self, "metric_name", _metric_name(self.metric_name))
        comparator = _identifier(self.comparator, "comparator", 2)
        if comparator not in _COMPARATORS:
            raise ValueError(f"comparator must be one of {sorted(_COMPARATORS)}.")
        object.__setattr__(self, "comparator", comparator)
        object.__setattr__(self, "threshold", _finite(self.threshold, "threshold"))
        if not isinstance(self.required, bool):
            raise ValueError("required must be boolean.")
        object.__setattr__(self, "tag_match", _tag_pairs(self.tag_match, label="tag_match"))


@dataclass(frozen=True)
class QualitySLOResult:
    name: str
    status: str
    passed: bool
    metric_name: str
    comparator: str
    threshold: float
    observed_value: float | None
    metric_identity: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "passed": self.passed,
            "metric_name": self.metric_name,
            "comparator": self.comparator,
            "threshold": self.threshold,
            "observed_value": self.observed_value,
            "metric_identity": self.metric_identity,
        }


@dataclass(frozen=True)
class QualityDashboard:
    snapshot: QualitySnapshot
    slo_results: tuple[QualitySLOResult, ...]

    @property
    def healthy(self) -> bool:
        return all(row.passed for row in self.slo_results)

    @property
    def failed_slos(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.slo_results if not row.passed)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": self.snapshot.snapshot_digest,
            "comparison_scope_digest": self.snapshot.provenance.comparison_scope_digest,
            "window": self.snapshot.window.to_dict(),
            "provenance": self.snapshot.provenance.to_dict(),
            "metric_count": len(self.snapshot.metrics),
            "metrics": [row.to_dict() for row in self.snapshot.metrics],
            "slo_count": len(self.slo_results),
            "slo_results": [row.to_dict() for row in self.slo_results],
            "healthy": self.healthy,
            "failed_slos": list(self.failed_slos),
        }

    @property
    def dashboard_digest(self) -> str:
        return canonical_digest(self._payload())

    def to_dict(self) -> dict[str, Any]:
        value = self._payload()
        value["dashboard_digest"] = self.dashboard_digest
        return value


@dataclass(frozen=True)
class MetricDelta:
    name: str
    tags: tuple[tuple[str, str], ...]
    direction: str
    unit: str
    baseline_value: float | None
    current_value: float | None
    absolute_delta: float | None
    relative_delta: float | None
    normalized_delta: float | None
    state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tags": dict(self.tags),
            "direction": self.direction,
            "unit": self.unit,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "normalized_delta": self.normalized_delta,
            "state": self.state,
        }


@dataclass(frozen=True)
class QualityComparison:
    baseline_snapshot_digest: str
    current_snapshot_digest: str
    comparison_scope_digest: str
    deltas: tuple[MetricDelta, ...]

    @property
    def improved_count(self) -> int:
        return sum(row.state == "improved" for row in self.deltas)

    @property
    def regressed_count(self) -> int:
        return sum(row.state == "regressed" for row in self.deltas)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "baseline_snapshot_digest": self.baseline_snapshot_digest,
            "current_snapshot_digest": self.current_snapshot_digest,
            "comparison_scope_digest": self.comparison_scope_digest,
            "improved_count": self.improved_count,
            "regressed_count": self.regressed_count,
            "deltas": [row.to_dict() for row in self.deltas],
        }
        value["comparison_digest"] = canonical_digest(value)
        return value


def _matches(metric: MetricObservation, slo: QualitySLO) -> bool:
    if metric.name != slo.metric_name:
        return False
    tags = dict(metric.tags)
    return all(tags.get(key) == value for key, value in slo.tag_match)


def _threshold_passes(value: float, comparator: str, threshold: float) -> bool:
    if comparator == ">=":
        return value >= threshold
    if comparator == "<=":
        return value <= threshold
    if comparator == ">":
        return value > threshold
    if comparator == "<":
        return value < threshold
    raise AssertionError("validated comparator was lost")


def evaluate_quality_slos(
    snapshot: QualitySnapshot,
    slos: Iterable[QualitySLO],
) -> tuple[QualitySLOResult, ...]:
    if not isinstance(snapshot, QualitySnapshot):
        raise ValueError("snapshot must be QualitySnapshot.")
    slos = tuple(slos)
    if len(slos) > _MAX_SLOS or any(not isinstance(row, QualitySLO) for row in slos):
        raise ValueError("slos must be a bounded QualitySLO sequence.")
    if len({row.name for row in slos}) != len(slos):
        raise ValueError("SLO names must be unique.")
    output: list[QualitySLOResult] = []
    for slo in slos:
        matches = [metric for metric in snapshot.metrics if _matches(metric, slo)]
        if not matches:
            output.append(
                QualitySLOResult(
                    slo.name, "missing" if slo.required else "optional_missing",
                    not slo.required, slo.metric_name, slo.comparator, slo.threshold, None, None,
                )
            )
            continue
        if len(matches) != 1:
            output.append(
                QualitySLOResult(
                    slo.name, "ambiguous", False, slo.metric_name, slo.comparator,
                    slo.threshold, None, None,
                )
            )
            continue
        metric = matches[0]
        passed = _threshold_passes(metric.value, slo.comparator, slo.threshold)
        output.append(
            QualitySLOResult(
                slo.name, "passed" if passed else "failed", passed, slo.metric_name,
                slo.comparator, slo.threshold, metric.value, metric.identity,
            )
        )
    return tuple(output)


def build_quality_dashboard(
    snapshot: QualitySnapshot,
    slos: Iterable[QualitySLO] = (),
) -> QualityDashboard:
    return QualityDashboard(snapshot, evaluate_quality_slos(snapshot, slos))


def compare_quality_snapshots(
    baseline: QualitySnapshot,
    current: QualitySnapshot,
    *,
    tolerance: float = 0.0,
) -> QualityComparison:
    """Direction-aware comparison with immutable like-for-like evaluation scope."""

    if not isinstance(baseline, QualitySnapshot) or not isinstance(current, QualitySnapshot):
        raise ValueError("baseline and current must be QualitySnapshot values.")
    tolerance = _finite(tolerance, "tolerance")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    scope = baseline.provenance.comparison_scope_digest
    if scope != current.provenance.comparison_scope_digest:
        raise ValueError("quality snapshots are not in the same comparison scope.")
    base = {row.identity: row for row in baseline.metrics}
    candidate = {row.identity: row for row in current.metrics}
    deltas: list[MetricDelta] = []
    for identity in sorted(set(base) | set(candidate)):
        left, right = base.get(identity), candidate.get(identity)
        selected = right or left
        assert selected is not None
        if left is None or right is None:
            deltas.append(
                MetricDelta(
                    selected.name, selected.tags, selected.direction, selected.unit,
                    None if left is None else left.value,
                    None if right is None else right.value,
                    None, None, None, "new" if left is None else "missing",
                )
            )
            continue
        if left.direction != right.direction or left.unit != right.unit:
            raise ValueError(f"metric contract changed for {left.name!r}; direction/unit must match.")
        absolute = right.value - left.value
        relative = None if left.value == 0.0 else absolute / abs(left.value)
        normalized = absolute if left.direction == "higher" else -absolute if left.direction == "lower" else None
        if left.direction == "neutral":
            state = "unchanged" if abs(absolute) <= tolerance else "changed"
        elif normalized is not None and normalized > tolerance:
            state = "improved"
        elif normalized is not None and normalized < -tolerance:
            state = "regressed"
        else:
            state = "unchanged"
        deltas.append(
            MetricDelta(
                left.name, left.tags, left.direction, left.unit, left.value, right.value,
                absolute, relative, normalized, state,
            )
        )
    deltas.sort(key=lambda row: (row.name, row.tags))
    return QualityComparison(baseline.snapshot_digest, current.snapshot_digest, scope, tuple(deltas))


def observations_from_mapping(
    namespace: str,
    metrics: Mapping[str, Any],
    *,
    sample_count: int,
    source: str,
    directions: Mapping[str, str],
    units: Mapping[str, str] | None = None,
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    """Normalize scalar metrics without guessing an unknown metric's direction."""

    namespace = _metric_name(namespace, "namespace")
    sample_count = _count(sample_count, "sample_count")
    if not isinstance(metrics, Mapping) or len(metrics) > _MAX_METRICS:
        raise ValueError("metrics must be a bounded mapping.")
    if not isinstance(directions, Mapping):
        raise ValueError("directions must be a mapping.")
    if any(not isinstance(name, str) for name in metrics):
        raise ValueError("metric names must be strings.")
    units = units or {}
    pairs = _tag_pairs(tags)
    output = []
    for name in sorted(metrics):
        if name not in directions:
            raise ValueError(f"direction is required for metric {name!r}.")
        if metrics[name] is None:
            continue
        output.append(
            MetricObservation(
                f"{namespace}.{_metric_name(name)}", _finite(metrics[name], name),
                directions[name], units.get(name, "ratio"), sample_count, source, pairs,
            )
        )
    return tuple(output)


_RETRIEVAL_PREFIXES = ("precision@", "recall@", "hit_rate@", "mrr@", "map@", "ndcg@")
_GENERATION_DIRECTIONS = {"rouge_l": "higher", "chrf": "higher", "unsupported_claim_rate": "lower"}


def observations_from_retrieval_metrics(
    metrics: Mapping[str, Any],
    *,
    sample_count: int,
    source: str = "evaluation.retrieval",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    directions = {}
    for name in metrics:
        if not isinstance(name, str) or not name.lower().startswith(_RETRIEVAL_PREFIXES):
            raise ValueError(
                f"retrieval metric {name!r} has no registered direction; "
                "use observations_from_mapping with an explicit direction."
            )
        directions[name] = "higher"
    return observations_from_mapping(
        "retrieval", metrics, sample_count=sample_count, source=source,
        directions=directions, tags=tags,
    )


def observations_from_generation_metrics(
    metrics: Mapping[str, Any],
    *,
    sample_count: int,
    source: str = "evaluation.generation_metrics",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    directions = {}
    for name in metrics:
        if name not in _GENERATION_DIRECTIONS:
            raise ValueError(
                f"generation metric {name!r} has no registered direction; "
                "use observations_from_mapping with an explicit direction."
            )
        directions[name] = _GENERATION_DIRECTIONS[name]
    return observations_from_mapping(
        "generation", metrics, sample_count=sample_count, source=source,
        directions=directions, tags=tags,
    )


def observations_from_benchmark_suite(
    result: BenchmarkSuiteResult,
    *,
    source: str = "evaluation.benchmark_suite",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    """Export benchmark aggregates only; row queries/answers are never serialized."""

    if not isinstance(result, BenchmarkSuiteResult):
        raise ValueError("result must be BenchmarkSuiteResult.")
    retrieval: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    latency: dict[str, Any] = {}
    for name, value in result.aggregate.items():
        if name == "retrieval_latency_ms":
            latency["retrieval_mean_ms"] = value
        elif name == "generation_latency_ms":
            latency["generation_mean_ms"] = value
        elif isinstance(name, str) and name.lower().startswith(_RETRIEVAL_PREFIXES):
            retrieval[name] = value
        elif name in _GENERATION_DIRECTIONS:
            generation[name] = value
        else:
            raise ValueError(f"benchmark aggregate metric {name!r} has no observability contract.")
    count = len(result.rows)
    output = list(observations_from_retrieval_metrics(retrieval, sample_count=count, source=source, tags=tags))
    output.extend(observations_from_generation_metrics(generation, sample_count=count, source=source, tags=tags))
    if latency:
        output.extend(
            observations_from_mapping(
                "benchmark_latency", latency, sample_count=count, source=source,
                directions={name: "lower" for name in latency},
                units={name: "ms" for name in latency}, tags=tags,
            )
        )
    return tuple(output)


def observations_from_semantic_metrics(
    semantic: SemanticMetrics,
    citation: CitationSupportMetrics,
    *,
    source: str = "evaluation.semantic_support",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    if not isinstance(semantic, SemanticMetrics) or not isinstance(citation, CitationSupportMetrics):
        raise ValueError("semantic/citation must use semantic_support result types.")
    semantic_values = {
        "coverage": semantic.coverage,
        "accuracy_on_covered": semantic.accuracy_on_covered,
        "entailment_recall": semantic.entailment_recall,
        "contradiction_recall": semantic.contradiction_recall,
        "contradiction_false_negative_rate": semantic.contradiction_false_negative_rate,
        "multiclass_brier": semantic.multiclass_brier,
        "expected_calibration_error": semantic.expected_calibration_error,
    }
    semantic_directions = {
        "coverage": "higher", "accuracy_on_covered": "higher", "entailment_recall": "higher",
        "contradiction_recall": "higher", "contradiction_false_negative_rate": "lower",
        "multiclass_brier": "lower", "expected_calibration_error": "lower",
    }
    citation_values = {
        "claim_coverage": citation.claim_coverage,
        "mean_best_entailment": citation.mean_best_entailment,
        "supported_claim_rate": citation.supported_claim_rate,
        "contradicted_claim_rate": citation.contradicted_claim_rate,
        "unsupported_claim_rate": citation.unsupported_claim_rate,
    }
    citation_directions = {
        "claim_coverage": "higher", "mean_best_entailment": "higher",
        "supported_claim_rate": "higher", "contradicted_claim_rate": "lower",
        "unsupported_claim_rate": "lower",
    }
    output = list(
        observations_from_mapping(
            "semantic", semantic_values, sample_count=semantic.count, source=source,
            directions=semantic_directions, tags=tags,
        )
    )
    output.extend(
        observations_from_mapping(
            "citation", citation_values, sample_count=citation.claim_count, source=source,
            directions=citation_directions, tags=tags,
        )
    )
    pairs = _tag_pairs(tags)
    output.extend(
        (
            MetricObservation("semantic.example_count", semantic.count, "neutral", "count", semantic.count, source, pairs),
            MetricObservation("citation.claim_count", citation.claim_count, "neutral", "count", citation.claim_count, source, pairs),
            MetricObservation("citation.citation_count", citation.citation_count, "neutral", "count", citation.claim_count, source, pairs),
        )
    )
    return tuple(output)


def observations_from_selective_risk(
    metrics: SelectiveRiskMetrics,
    *,
    source: str = "evaluation.conformal_retrieval",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    if not isinstance(metrics, SelectiveRiskMetrics):
        raise ValueError("metrics must be SelectiveRiskMetrics.")
    output = list(
        observations_from_mapping(
            "selective",
            {
                "coverage": metrics.coverage,
                "error_rate_on_covered": metrics.error_rate_on_covered,
                "abstention_rate": metrics.abstention_rate,
            },
            sample_count=metrics.total, source=source,
            directions={"coverage": "higher", "error_rate_on_covered": "lower", "abstention_rate": "neutral"},
            tags=tags,
        )
    )
    pairs = _tag_pairs(tags)
    output.extend(
        (
            MetricObservation("selective.total", metrics.total, "neutral", "count", metrics.total, source, pairs),
            MetricObservation("selective.covered", metrics.covered, "neutral", "count", metrics.total, source, pairs),
        )
    )
    return tuple(output)


def observations_from_latency_summary(
    summary: LatencySummary,
    *,
    source: str = "evaluation.efficiency",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    if not isinstance(summary, LatencySummary):
        raise ValueError("summary must be LatencySummary.")
    pairs = _tag_pairs(tags)
    output = [MetricObservation("latency.count", summary.count, "neutral", "count", summary.count, source, pairs)]
    for name in ("mean_ms", "median_ms", "p95_ms", "p99_ms", "minimum_ms", "maximum_ms"):
        output.append(MetricObservation(f"latency.{name}", getattr(summary, name), "lower", "ms", summary.count, source, pairs))
    return tuple(output)


def observations_from_resource_usage(
    usage: ResourceUsage,
    *,
    sample_count: int = 1,
    source: str = "evaluation.resource_measurement",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    if not isinstance(usage, ResourceUsage):
        raise ValueError("usage must be ResourceUsage.")
    count = _count(sample_count, "sample_count", 1)
    pairs = _tag_pairs(tags)
    output = [
        MetricObservation("resource.wall_ms", usage.wall_ms, "lower", "ms", count, source, pairs),
        MetricObservation("resource.cpu_ms", usage.cpu_ms, "lower", "ms", count, source, pairs),
        MetricObservation("resource.python_peak_allocated_bytes", usage.python_peak_allocated_bytes, "lower", "bytes", count, source, pairs),
        MetricObservation("resource.prompt_tokens", usage.provider.prompt_tokens, "lower", "tokens", count, source, pairs),
        MetricObservation("resource.completion_tokens", usage.provider.completion_tokens, "lower", "tokens", count, source, pairs),
        MetricObservation("resource.cost_units", usage.provider.cost_units, "lower", "cost_units", count, source, pairs),
    ]
    if usage.process_peak_rss_bytes is not None:
        output.append(MetricObservation("resource.process_peak_rss_bytes", usage.process_peak_rss_bytes, "lower", "bytes", count, source, pairs))
    return tuple(output)


def observations_from_drift_report(
    report: DriftReport,
    *,
    sample_count: int,
    source: str = "evaluation.drift",
    tags: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> tuple[MetricObservation, ...]:
    if not isinstance(report, DriftReport):
        raise ValueError("report must be DriftReport.")
    count = _count(sample_count, "sample_count")
    pairs = _tag_pairs(tags)
    values = (
        ("drift.score_psi", report.score_psi, "score"),
        ("drift.route_jsd", report.route_jsd, "score"),
        ("drift.calibration_shift", report.calibration_shift, "ratio"),
        ("drift.latency_relative", report.latency_relative, "ratio"),
        ("drift.cost_relative", report.cost_relative, "ratio"),
        ("drift.alert_count", len(report.alerts), "count"),
    )
    return tuple(
        MetricObservation(name, value, "lower", unit, count, source, pairs)
        for name, value, unit in values
    )


def _redirecting(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse)


def _atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    if not destination.name:
        raise ValueError("output path must name a file.")
    parent = destination.parent if str(destination.parent) else Path(".")
    parent.mkdir(parents=True, exist_ok=True)
    for component in (parent, *parent.parents):
        if _redirecting(component):
            raise ValueError("output path may not traverse a symbolic link or reparse point.")
    if _redirecting(destination):
        raise ValueError("output path may not be a symbolic link or reparse point.")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(parent))
        temporary = Path(name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(_canonical_bytes(payload) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        os.replace(temporary, destination)
        temporary = None
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass
        try:
            directory_fd = os.open(parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                try:
                    os.fsync(directory_fd)
                except OSError:
                    pass
            finally:
                os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return destination


def write_quality_snapshot(path: str | os.PathLike[str], snapshot: QualitySnapshot) -> Path:
    if not isinstance(snapshot, QualitySnapshot):
        raise ValueError("snapshot must be QualitySnapshot.")
    return _atomic_write_json(path, snapshot.to_dict())


def write_quality_dashboard(path: str | os.PathLike[str], dashboard: QualityDashboard) -> Path:
    if not isinstance(dashboard, QualityDashboard):
        raise ValueError("dashboard must be QualityDashboard.")
    return _atomic_write_json(path, dashboard.to_dict())


__all__ = [
    "MetricDelta", "MetricObservation", "QualityComparison", "QualityDashboard",
    "QualityProvenance", "QualitySLO", "QualitySLOResult", "QualitySnapshot",
    "QualityWindow", "SCHEMA_VERSION", "build_quality_dashboard", "canonical_digest",
    "compare_quality_snapshots", "evaluate_quality_slos",
    "observations_from_benchmark_suite", "observations_from_drift_report",
    "observations_from_generation_metrics", "observations_from_latency_summary",
    "observations_from_mapping", "observations_from_resource_usage",
    "observations_from_retrieval_metrics", "observations_from_selective_risk",
    "observations_from_semantic_metrics", "write_quality_dashboard",
    "write_quality_snapshot",
]
