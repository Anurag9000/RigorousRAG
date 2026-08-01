"""Governed, non-mutating promotion decisions for validated migration shadows."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from tools.migration_types import digest, exact_integer, identifier, timestamp
from tools.security import normalize_owner_id

_SCHEMA_VERSION = 1
_METRIC_NAMES = (
    "recall_at_k",
    "ndcg_at_k",
    "mrr",
    "support_recall",
    "citation_precision",
    "abstention_accuracy",
)
_RESOURCE_NAMES = (
    "p95_latency_ms",
    "peak_memory_bytes",
    "index_bytes",
    "estimated_cost_units",
)


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be a finite number between 0 and 1.")
    return parsed


def _nonnegative_float(value: Any, label: str, maximum: float = 1e18) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= maximum:
        raise ValueError(f"{label} must be a finite non-negative number.")
    return parsed


def _positive_float(value: Any, label: str, maximum: float = 1e9) -> float:
    parsed = _nonnegative_float(value, label, maximum)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class RetrievalMetrics:
    query_count: int
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    support_recall: float
    citation_precision: float
    abstention_accuracy: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "query_count",
            exact_integer(self.query_count, "query_count", 1, 10_000_000),
        )
        for name in _METRIC_NAMES:
            object.__setattr__(self, name, _probability(getattr(self, name), name))


@dataclass(frozen=True)
class ResourceMetrics:
    p95_latency_ms: float
    peak_memory_bytes: int
    index_bytes: int
    estimated_cost_units: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "p95_latency_ms",
            _nonnegative_float(self.p95_latency_ms, "p95_latency_ms"),
        )
        object.__setattr__(
            self,
            "peak_memory_bytes",
            exact_integer(
                self.peak_memory_bytes,
                "peak_memory_bytes",
                0,
                2**63 - 1,
            ),
        )
        object.__setattr__(
            self,
            "index_bytes",
            exact_integer(self.index_bytes, "index_bytes", 0, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "estimated_cost_units",
            _nonnegative_float(
                self.estimated_cost_units,
                "estimated_cost_units",
            ),
        )


@dataclass(frozen=True)
class PromotionEvidence:
    task_id: str
    validation_digest: str
    benchmark_fingerprint: str
    source_sequence: int
    source_content_sha256: str
    vector_count: int
    sparse_count: int
    repeated_runs: int
    seed_count: int
    confidence_interval_level: float
    current_quality: RetrievalMetrics
    shadow_quality: RetrievalMetrics
    current_resources: ResourceMetrics
    shadow_resources: ResourceMetrics

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(
            self,
            "validation_digest",
            digest(self.validation_digest, "validation_digest"),
        )
        object.__setattr__(
            self,
            "benchmark_fingerprint",
            digest(self.benchmark_fingerprint, "benchmark_fingerprint"),
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
            "repeated_runs",
            exact_integer(self.repeated_runs, "repeated_runs", 1, 10_000),
        )
        object.__setattr__(
            self,
            "seed_count",
            exact_integer(self.seed_count, "seed_count", 1, 10_000),
        )
        object.__setattr__(
            self,
            "confidence_interval_level",
            _probability(self.confidence_interval_level, "confidence_interval_level"),
        )
        if self.confidence_interval_level <= 0.0:
            raise ValueError("confidence_interval_level must be positive.")
        if not isinstance(self.current_quality, RetrievalMetrics):
            raise ValueError("current_quality must be RetrievalMetrics.")
        if not isinstance(self.shadow_quality, RetrievalMetrics):
            raise ValueError("shadow_quality must be RetrievalMetrics.")
        if not isinstance(self.current_resources, ResourceMetrics):
            raise ValueError("current_resources must be ResourceMetrics.")
        if not isinstance(self.shadow_resources, ResourceMetrics):
            raise ValueError("shadow_resources must be ResourceMetrics.")
        if self.current_quality.query_count != self.shadow_quality.query_count:
            raise ValueError("current and shadow quality must use the same query count.")

    @property
    def evidence_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class PromotionPolicy:
    policy_id: str = "conservative-v1"
    min_query_count: int = 50
    min_repeated_runs: int = 3
    min_seed_count: int = 3
    min_confidence_interval_level: float = 0.95
    require_equal_vector_sparse_counts: bool = True
    minimum_recall_at_k: float = 0.70
    minimum_ndcg_at_k: float = 0.60
    minimum_mrr: float = 0.60
    minimum_support_recall: float = 0.70
    minimum_citation_precision: float = 0.90
    minimum_abstention_accuracy: float = 0.80
    max_recall_regression: float = 0.01
    max_ndcg_regression: float = 0.01
    max_mrr_regression: float = 0.01
    max_support_recall_regression: float = 0.01
    max_citation_precision_regression: float = 0.0
    max_abstention_accuracy_regression: float = 0.01
    max_latency_ratio: float = 1.50
    max_memory_ratio: float = 1.50
    max_storage_ratio: float = 2.00
    max_cost_ratio: float = 1.50

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", identifier(self.policy_id, "policy_id", 128))
        object.__setattr__(
            self,
            "min_query_count",
            exact_integer(self.min_query_count, "min_query_count", 1, 10_000_000),
        )
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
            "min_confidence_interval_level",
            _probability(
                self.min_confidence_interval_level,
                "min_confidence_interval_level",
            ),
        )
        if self.min_confidence_interval_level <= 0.0:
            raise ValueError("min_confidence_interval_level must be positive.")
        if not isinstance(self.require_equal_vector_sparse_counts, bool):
            raise ValueError("require_equal_vector_sparse_counts must be boolean.")
        for name in (
            "minimum_recall_at_k",
            "minimum_ndcg_at_k",
            "minimum_mrr",
            "minimum_support_recall",
            "minimum_citation_precision",
            "minimum_abstention_accuracy",
            "max_recall_regression",
            "max_ndcg_regression",
            "max_mrr_regression",
            "max_support_recall_regression",
            "max_citation_precision_regression",
            "max_abstention_accuracy_regression",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in (
            "max_latency_ratio",
            "max_memory_ratio",
            "max_storage_ratio",
            "max_cost_ratio",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))

    @property
    def policy_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class PromotionReport:
    task_id: str
    owner_id: str
    doc_id: str
    source_sequence: int
    source_profile_fingerprint: str
    target_profile_fingerprint: str
    validation_digest: str
    benchmark_fingerprint: str
    evidence_digest: str
    policy_id: str
    policy_digest: str
    decision: str
    reason_codes: tuple[str, ...]
    quality_deltas: Mapping[str, float]
    resource_ratios: Mapping[str, float | None]
    evaluated_at: float
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", identifier(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "source_sequence",
            exact_integer(self.source_sequence, "source_sequence", 1, 2**63 - 1),
        )
        for name in (
            "source_profile_fingerprint",
            "target_profile_fingerprint",
            "validation_digest",
            "benchmark_fingerprint",
            "evidence_digest",
            "policy_digest",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        object.__setattr__(self, "policy_id", identifier(self.policy_id, "policy_id", 128))
        if self.decision not in {"eligible", "blocked"}:
            raise ValueError("promotion decision is invalid.")
        if not isinstance(self.reason_codes, tuple):
            object.__setattr__(self, "reason_codes", tuple(self.reason_codes))
        reasons = tuple(identifier(item, "reason_code", 200) for item in self.reason_codes)
        if reasons != tuple(sorted(set(reasons))):
            raise ValueError("reason_codes must be unique and sorted.")
        object.__setattr__(self, "reason_codes", reasons)
        if self.decision == "eligible" and reasons:
            raise ValueError("eligible reports may not contain blocking reasons.")
        if self.decision == "blocked" and not reasons:
            raise ValueError("blocked reports require at least one reason.")
        object.__setattr__(
            self,
            "quality_deltas",
            _validated_float_mapping(self.quality_deltas, _METRIC_NAMES, "quality_deltas"),
        )
        object.__setattr__(
            self,
            "resource_ratios",
            _validated_ratio_mapping(self.resource_ratios),
        )
        object.__setattr__(self, "evaluated_at", timestamp(self.evaluated_at, "evaluated_at"))
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("promotion report schema is unsupported.")

    @property
    def report_digest(self) -> str:
        stable = asdict(self)
        stable.pop("evaluated_at", None)
        return _sha256(stable)


def _validated_float_mapping(
    value: Mapping[str, Any],
    names: tuple[str, ...],
    label: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise ValueError(f"{label} must contain exactly the required metrics.")
    result: dict[str, float] = {}
    for name in names:
        raw = value[name]
        if isinstance(raw, bool):
            raise ValueError(f"{label}.{name} must be finite.")
        try:
            parsed = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label}.{name} must be finite.") from exc
        if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
            raise ValueError(f"{label}.{name} must be finite and bounded.")
        result[name] = parsed
    return result


def _validated_ratio_mapping(value: Mapping[str, Any]) -> dict[str, float | None]:
    if not isinstance(value, Mapping) or set(value) != set(_RESOURCE_NAMES):
        raise ValueError("resource_ratios must contain exactly the required resources.")
    result: dict[str, float | None] = {}
    for name in _RESOURCE_NAMES:
        raw = value[name]
        if raw is None:
            result[name] = None
            continue
        result[name] = _nonnegative_float(raw, f"resource_ratios.{name}")
    return result


def _ratio(current: float | int, shadow: float | int) -> float | None:
    baseline = float(current)
    candidate = float(shadow)
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else None
    return candidate / baseline


def _task_manifest_matches(task: Any, manifest: Any) -> bool:
    return bool(
        getattr(task, "task_id", None) == getattr(manifest, "task_id", None)
        and getattr(task, "owner_id", None) == getattr(manifest, "owner_id", None)
        and getattr(task, "doc_id", None) == getattr(manifest, "doc_id", None)
        and getattr(task, "source_sequence", None)
        == getattr(manifest, "source_sequence", None)
        and getattr(task, "source_profile_fingerprint", None)
        == getattr(manifest, "source_profile_fingerprint", None)
        and getattr(task, "target_profile_fingerprint", None)
        == getattr(manifest, "target_profile_fingerprint", None)
    )


def evaluate_promotion(
    *,
    task: Any,
    manifest: Any,
    generation: Any,
    evidence: PromotionEvidence,
    policy: PromotionPolicy | None = None,
    now: float | None = None,
) -> PromotionReport:
    """Return a deterministic non-mutating eligibility report for one shadow."""

    selected = policy or PromotionPolicy()
    if not isinstance(evidence, PromotionEvidence):
        raise ValueError("evidence must be PromotionEvidence.")
    if not isinstance(selected, PromotionPolicy):
        raise ValueError("policy must be PromotionPolicy.")
    if getattr(task, "state", None) != "validated":
        raise ValueError("promotion evaluation requires a validated task.")
    if not _task_manifest_matches(task, manifest):
        raise RuntimeError("migration task and shadow manifest do not match.")
    validation_digest = digest(
        getattr(manifest, "validation_digest", None),
        "manifest validation_digest",
    )
    if getattr(task, "validation_digest", None) != validation_digest:
        raise RuntimeError("migration journal and shadow validation digests differ.")
    if evidence.task_id != task.task_id or evidence.validation_digest != validation_digest:
        raise RuntimeError("promotion evidence does not identify the validated shadow.")

    reasons: set[str] = set()
    if generation is None:
        reasons.add("source_generation_missing")
    else:
        if getattr(generation, "state", None) not in {"active", "restored"}:
            reasons.add("source_generation_inactive")
        if getattr(generation, "sequence", None) != task.source_sequence:
            reasons.add("source_generation_sequence_changed")
        if (
            getattr(generation, "profile_fingerprint", None)
            != task.source_profile_fingerprint
        ):
            reasons.add("source_profile_changed")
        if getattr(generation, "content_sha256", None) != evidence.source_content_sha256:
            reasons.add("source_content_changed")
    if evidence.source_sequence != task.source_sequence:
        reasons.add("evidence_source_sequence_mismatch")
    if evidence.source_content_sha256 != getattr(manifest, "content_sha256", None):
        reasons.add("evidence_content_hash_mismatch")
    if evidence.vector_count != getattr(manifest, "vector_count", None):
        reasons.add("evidence_vector_count_mismatch")
    if evidence.sparse_count != getattr(manifest, "sparse_count", None):
        reasons.add("evidence_sparse_count_mismatch")
    if (
        selected.require_equal_vector_sparse_counts
        and evidence.vector_count != evidence.sparse_count
    ):
        reasons.add("vector_sparse_count_mismatch")
    if evidence.current_quality.query_count < selected.min_query_count:
        reasons.add("benchmark_query_count_below_minimum")
    if evidence.repeated_runs < selected.min_repeated_runs:
        reasons.add("benchmark_repeated_runs_below_minimum")
    if evidence.seed_count < selected.min_seed_count:
        reasons.add("benchmark_seed_count_below_minimum")
    if evidence.confidence_interval_level < selected.min_confidence_interval_level:
        reasons.add("benchmark_confidence_level_below_minimum")

    quality_deltas: dict[str, float] = {}
    floors = {
        "recall_at_k": selected.minimum_recall_at_k,
        "ndcg_at_k": selected.minimum_ndcg_at_k,
        "mrr": selected.minimum_mrr,
        "support_recall": selected.minimum_support_recall,
        "citation_precision": selected.minimum_citation_precision,
        "abstention_accuracy": selected.minimum_abstention_accuracy,
    }
    regressions = {
        "recall_at_k": selected.max_recall_regression,
        "ndcg_at_k": selected.max_ndcg_regression,
        "mrr": selected.max_mrr_regression,
        "support_recall": selected.max_support_recall_regression,
        "citation_precision": selected.max_citation_precision_regression,
        "abstention_accuracy": selected.max_abstention_accuracy_regression,
    }
    for name in _METRIC_NAMES:
        current = getattr(evidence.current_quality, name)
        shadow = getattr(evidence.shadow_quality, name)
        delta = shadow - current
        quality_deltas[name] = delta
        if shadow < floors[name]:
            reasons.add(f"{name}_below_floor")
        if delta < -regressions[name]:
            reasons.add(f"{name}_regression_exceeds_limit")

    resource_ratios = {
        "p95_latency_ms": _ratio(
            evidence.current_resources.p95_latency_ms,
            evidence.shadow_resources.p95_latency_ms,
        ),
        "peak_memory_bytes": _ratio(
            evidence.current_resources.peak_memory_bytes,
            evidence.shadow_resources.peak_memory_bytes,
        ),
        "index_bytes": _ratio(
            evidence.current_resources.index_bytes,
            evidence.shadow_resources.index_bytes,
        ),
        "estimated_cost_units": _ratio(
            evidence.current_resources.estimated_cost_units,
            evidence.shadow_resources.estimated_cost_units,
        ),
    }
    ratio_limits = {
        "p95_latency_ms": selected.max_latency_ratio,
        "peak_memory_bytes": selected.max_memory_ratio,
        "index_bytes": selected.max_storage_ratio,
        "estimated_cost_units": selected.max_cost_ratio,
    }
    for name, ratio in resource_ratios.items():
        if ratio is None:
            reasons.add(f"{name}_has_zero_baseline")
        elif ratio > ratio_limits[name]:
            reasons.add(f"{name}_ratio_exceeds_limit")

    ordered = tuple(sorted(reasons))
    return PromotionReport(
        task_id=task.task_id,
        owner_id=task.owner_id,
        doc_id=task.doc_id,
        source_sequence=task.source_sequence,
        source_profile_fingerprint=task.source_profile_fingerprint,
        target_profile_fingerprint=task.target_profile_fingerprint,
        validation_digest=validation_digest,
        benchmark_fingerprint=evidence.benchmark_fingerprint,
        evidence_digest=evidence.evidence_digest,
        policy_id=selected.policy_id,
        policy_digest=selected.policy_digest,
        decision="blocked" if ordered else "eligible",
        reason_codes=ordered,
        quality_deltas=quality_deltas,
        resource_ratios=resource_ratios,
        evaluated_at=timestamp(time.time() if now is None else now, "evaluated_at"),
    )


def evidence_from_mapping(value: Mapping[str, Any]) -> PromotionEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("promotion evidence must be an object.")
    expected = {
        "task_id",
        "validation_digest",
        "benchmark_fingerprint",
        "source_sequence",
        "source_content_sha256",
        "vector_count",
        "sparse_count",
        "repeated_runs",
        "seed_count",
        "confidence_interval_level",
        "current_quality",
        "shadow_quality",
        "current_resources",
        "shadow_resources",
    }
    if set(value) != expected:
        raise ValueError("promotion evidence fields are incomplete or unsupported.")
    return PromotionEvidence(
        task_id=value["task_id"],
        validation_digest=value["validation_digest"],
        benchmark_fingerprint=value["benchmark_fingerprint"],
        source_sequence=value["source_sequence"],
        source_content_sha256=value["source_content_sha256"],
        vector_count=value["vector_count"],
        sparse_count=value["sparse_count"],
        repeated_runs=value["repeated_runs"],
        seed_count=value["seed_count"],
        confidence_interval_level=value["confidence_interval_level"],
        current_quality=RetrievalMetrics(**value["current_quality"]),
        shadow_quality=RetrievalMetrics(**value["shadow_quality"]),
        current_resources=ResourceMetrics(**value["current_resources"]),
        shadow_resources=ResourceMetrics(**value["shadow_resources"]),
    )


def policy_from_mapping(value: Mapping[str, Any]) -> PromotionPolicy:
    if not isinstance(value, Mapping):
        raise ValueError("promotion policy must be an object.")
    allowed = set(PromotionPolicy.__dataclass_fields__)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("promotion policy contains unsupported fields.")
    return PromotionPolicy(**dict(value))


__all__ = [
    "PromotionEvidence",
    "PromotionPolicy",
    "PromotionReport",
    "ResourceMetrics",
    "RetrievalMetrics",
    "evaluate_promotion",
    "evidence_from_mapping",
    "policy_from_mapping",
]
