"""Framework-neutral learned query-planning artifacts and deterministic normalizers.

The runtime can use these immutable artifacts without embedding a training framework:

* a linear softmax domain classifier with explicit feature schema and fallback policy;
* pairwise/listwise objectives for learning query-plan rankings;
* auditable query-plan candidates with cost/risk/features;
* deterministic entity alias resolution that refuses ambiguous silent linking; and
* temporal normalization into explicit closed/open ISO date intervals, including
  relative phrases only when a reference date is supplied.

Training data construction and optimizer execution intentionally live outside this
module.  The loss functions and artifact contracts are present so training can be
implemented reproducibly without changing production inference semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_FEATURES = 100_000
_MAX_LABELS = 10_000
_MAX_CANDIDATES = 10_000
_EPS = 1e-12


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _text(value: Any, label: str, maximum: int = 100_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or "\x00" in result:
        raise ValueError(f"{label} is empty or too long")
    return result


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


def _probability(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("softmax requires logits")
    maximum = max(logits)
    exponents = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponents)
    return tuple(value / denominator for value in exponents)


@dataclass(frozen=True)
class FeatureVector:
    schema: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.schema or len(self.schema) > _MAX_FEATURES or len(self.schema) != len(self.values):
            raise ValueError("feature schema and values must be non-empty and aligned")
        cleaned = tuple(_identifier(value, "feature name", 300) for value in self.schema)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "schema", cleaned)
        object.__setattr__(self, "values", tuple(_finite(value, "feature value") for value in self.values))


@dataclass(frozen=True)
class DomainPrediction:
    label: str
    probability: float
    probabilities: Mapping[str, float]
    fallback_used: bool
    artifact_digest: str


@dataclass(frozen=True)
class LinearDomainClassifier:
    labels: tuple[str, ...]
    feature_schema: tuple[str, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    fallback_label: str
    minimum_confidence: float = 0.55
    training_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.labels or len(self.labels) > _MAX_LABELS:
            raise ValueError("labels must be non-empty and bounded")
        labels = tuple(_identifier(value, "domain label", 300) for value in self.labels)
        if len(set(labels)) != len(labels):
            raise ValueError("domain labels must be unique")
        object.__setattr__(self, "labels", labels)
        if not self.feature_schema or len(self.feature_schema) > _MAX_FEATURES:
            raise ValueError("feature_schema must be non-empty and bounded")
        schema = tuple(_identifier(value, "feature name", 300) for value in self.feature_schema)
        if len(set(schema)) != len(schema):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "feature_schema", schema)
        if len(self.weights) != len(labels) or len(self.bias) != len(labels):
            raise ValueError("classifier dimensions do not match labels")
        cleaned_weights: list[tuple[float, ...]] = []
        for row in self.weights:
            if len(row) != len(schema):
                raise ValueError("classifier weight row does not match feature schema")
            cleaned_weights.append(tuple(_finite(value, "classifier weight") for value in row))
        object.__setattr__(self, "weights", tuple(cleaned_weights))
        object.__setattr__(self, "bias", tuple(_finite(value, "classifier bias") for value in self.bias))
        fallback = _identifier(self.fallback_label, "fallback_label", 300)
        if fallback not in labels:
            raise ValueError("fallback_label must be one of the classifier labels")
        object.__setattr__(self, "fallback_label", fallback)
        confidence = _probability(self.minimum_confidence, "minimum_confidence")
        object.__setattr__(self, "minimum_confidence", confidence)
        if self.training_manifest_digest is not None:
            digest = _identifier(self.training_manifest_digest, "training_manifest_digest", 64).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("training_manifest_digest must be SHA-256")
            object.__setattr__(self, "training_manifest_digest", digest)

    @property
    def artifact_digest(self) -> str:
        return _digest(asdict(self))

    def logits(self, vector: FeatureVector) -> tuple[float, ...]:
        if vector.schema != self.feature_schema:
            raise ValueError("feature schema mismatch")
        return tuple(
            bias + sum(weight * feature for weight, feature in zip(row, vector.values))
            for row, bias in zip(self.weights, self.bias)
        )

    def predict(self, vector: FeatureVector) -> DomainPrediction:
        probabilities = _softmax(self.logits(vector))
        best_index = max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))
        selected_probability = probabilities[best_index]
        fallback_used = selected_probability < self.minimum_confidence
        selected_label = self.fallback_label if fallback_used else self.labels[best_index]
        mapping = {label: probability for label, probability in zip(self.labels, probabilities)}
        return DomainPrediction(
            label=selected_label,
            probability=mapping[selected_label],
            probabilities=mapping,
            fallback_used=fallback_used,
            artifact_digest=self.artifact_digest,
        )


def domain_cross_entropy(logits: Sequence[Any], target_index: int) -> float:
    """Reference multinomial cross entropy for domain-classifier training adapters."""

    if not logits or len(logits) > _MAX_LABELS:
        raise ValueError("logits must be non-empty and bounded")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < len(logits):
        raise ValueError("target_index is invalid")
    values = tuple(_finite(value, "domain logit") for value in logits)
    probabilities = _softmax(values)
    return -math.log(max(probabilities[target_index], _EPS))


class PlanKind(str, Enum):
    DIRECT = "direct"
    HYBRID = "hybrid"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    STEP_BACK = "step_back"
    MULTI_HOP = "multi_hop"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"
    SCIENTIFIC = "scientific"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class QueryPlanCandidate:
    plan_id: str
    kind: PlanKind
    retrievers: tuple[str, ...]
    rerankers: tuple[str, ...] = ()
    decomposition: tuple[str, ...] = ()
    feature_values: Mapping[str, float] = field(default_factory=dict)
    estimated_latency_ms: float = 0.0
    estimated_cost_units: float = 0.0
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _identifier(self.plan_id, "plan_id", 500))
        if not isinstance(self.kind, PlanKind):
            object.__setattr__(self, "kind", PlanKind(self.kind))
        if not self.retrievers or len(self.retrievers) > 100:
            raise ValueError("at least one bounded retriever is required")
        object.__setattr__(self, "retrievers", tuple(_identifier(value, "retriever", 500) for value in self.retrievers))
        object.__setattr__(self, "rerankers", tuple(_identifier(value, "reranker", 500) for value in self.rerankers))
        object.__setattr__(self, "decomposition", tuple(_text(value, "decomposition step", 10_000) for value in self.decomposition))
        if len(self.feature_values) > _MAX_FEATURES:
            raise ValueError("feature_values is too large")
        cleaned: dict[str, float] = {}
        for key, value in self.feature_values.items():
            cleaned[_identifier(key, "plan feature", 300)] = _finite(value, "plan feature value")
        object.__setattr__(self, "feature_values", cleaned)
        for name in ("estimated_latency_ms", "estimated_cost_units", "risk_score"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.risk_score > 1.0:
            raise ValueError("risk_score must be at most 1")


@dataclass(frozen=True)
class LinearPlanRanker:
    feature_schema: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float = 0.0
    latency_penalty: float = 0.0
    cost_penalty: float = 0.0
    risk_penalty: float = 0.0
    training_manifest_digest: str | None = None

    def __post_init__(self) -> None:
        schema = tuple(_identifier(value, "plan feature", 300) for value in self.feature_schema)
        if not schema or len(schema) != len(self.weights) or len(schema) > _MAX_FEATURES:
            raise ValueError("plan ranker schema and weights must be aligned")
        if len(set(schema)) != len(schema):
            raise ValueError("plan feature schema must be unique")
        object.__setattr__(self, "feature_schema", schema)
        object.__setattr__(self, "weights", tuple(_finite(value, "plan weight") for value in self.weights))
        object.__setattr__(self, "bias", _finite(self.bias, "plan bias"))
        for name in ("latency_penalty", "cost_penalty", "risk_penalty"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.training_manifest_digest is not None:
            digest = _identifier(self.training_manifest_digest, "training_manifest_digest", 64).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("training_manifest_digest must be SHA-256")
            object.__setattr__(self, "training_manifest_digest", digest)

    @property
    def artifact_digest(self) -> str:
        return _digest(asdict(self))

    def score(self, candidate: QueryPlanCandidate) -> float:
        values = tuple(candidate.feature_values.get(name, 0.0) for name in self.feature_schema)
        score = self.bias + sum(weight * value for weight, value in zip(self.weights, values))
        score -= self.latency_penalty * candidate.estimated_latency_ms
        score -= self.cost_penalty * candidate.estimated_cost_units
        score -= self.risk_penalty * candidate.risk_score
        return score

    def rank(
        self,
        candidates: Sequence[QueryPlanCandidate],
        *,
        max_latency_ms: float | None = None,
        max_cost_units: float | None = None,
        max_risk: float | None = None,
    ) -> tuple[QueryPlanCandidate, ...]:
        if not candidates or len(candidates) > _MAX_CANDIDATES:
            raise ValueError("candidates must be non-empty and bounded")
        latency_limit = math.inf if max_latency_ms is None else _finite(max_latency_ms, "max_latency_ms")
        cost_limit = math.inf if max_cost_units is None else _finite(max_cost_units, "max_cost_units")
        risk_limit = 1.0 if max_risk is None else _probability(max_risk, "max_risk")
        if latency_limit < 0.0 or cost_limit < 0.0:
            raise ValueError("plan budgets must be non-negative")
        eligible = [
            candidate
            for candidate in candidates
            if candidate.estimated_latency_ms <= latency_limit
            and candidate.estimated_cost_units <= cost_limit
            and candidate.risk_score <= risk_limit
        ]
        if not eligible:
            return ()
        return tuple(sorted(eligible, key=lambda candidate: (-self.score(candidate), candidate.plan_id)))


def pairwise_plan_loss(preferred_score: Any, rejected_score: Any, *, margin: float = 0.0) -> float:
    preferred = _finite(preferred_score, "preferred_score")
    rejected = _finite(rejected_score, "rejected_score")
    selected_margin = _finite(margin, "margin")
    if selected_margin < 0.0:
        raise ValueError("margin must be non-negative")
    value = selected_margin - preferred + rejected
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def listwise_plan_loss(scores: Sequence[Any], utilities: Sequence[Any], *, temperature: float = 1.0) -> float:
    if not scores or len(scores) != len(utilities) or len(scores) > _MAX_CANDIDATES:
        raise ValueError("scores and utilities must be aligned bounded sequences")
    selected_temperature = _finite(temperature, "temperature")
    if selected_temperature <= 0.0:
        raise ValueError("temperature must be positive")
    model = _softmax(tuple(_finite(value, "score") / selected_temperature for value in scores))
    labels = tuple(_finite(value, "utility") for value in utilities)
    minimum = min(labels)
    shifted = tuple(value - minimum for value in labels)
    total = sum(shifted)
    if total <= 0.0:
        raise ValueError("utilities must contain at least two distinct values")
    target = tuple(value / total for value in shifted)
    return -sum(expected * math.log(max(observed, _EPS)) for expected, observed in zip(target, model))


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    entity_type: str
    provenance_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "entity_id", _identifier(self.entity_id, "entity_id", 500))
        object.__setattr__(self, "canonical_name", _text(self.canonical_name, "canonical_name", 2_000))
        if len(self.aliases) > 10_000:
            raise ValueError("aliases are too numerous")
        object.__setattr__(self, "aliases", tuple(_text(value, "alias", 2_000) for value in self.aliases))
        object.__setattr__(self, "entity_type", _identifier(self.entity_type, "entity_type", 500))
        object.__setattr__(self, "provenance_ids", tuple(_identifier(value, "provenance_id", 1_000) for value in self.provenance_ids))


@dataclass(frozen=True)
class EntityResolution:
    mention: str
    entity_id: str | None
    candidate_ids: tuple[str, ...]
    ambiguous: bool
    reason: str


class EntityAliasResolver:
    """Exact normalized alias resolver; ambiguity is surfaced rather than guessed away."""

    def __init__(self, records: Sequence[EntityRecord]) -> None:
        if not records or len(records) > 1_000_000:
            raise ValueError("records must be non-empty and bounded")
        self._records = {record.entity_id: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("entity_ids must be unique")
        index: dict[str, set[str]] = {}
        for record in records:
            for alias in (record.canonical_name, *record.aliases):
                index.setdefault(self._normalize(alias), set()).add(record.entity_id)
        self._index = {key: tuple(sorted(values)) for key, values in index.items()}

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(_text(text, "entity mention", 2_000).casefold().split())

    def resolve(self, mention: str, *, expected_type: str | None = None) -> EntityResolution:
        selected = _text(mention, "mention", 2_000)
        candidates = list(self._index.get(self._normalize(selected), ()))
        if expected_type is not None:
            expected = _identifier(expected_type, "expected_type", 500)
            candidates = [candidate for candidate in candidates if self._records[candidate].entity_type == expected]
        if not candidates:
            return EntityResolution(selected, None, (), False, "no exact normalized alias match")
        if len(candidates) > 1:
            return EntityResolution(selected, None, tuple(candidates), True, "ambiguous alias requires explicit disambiguation")
        return EntityResolution(selected, candidates[0], tuple(candidates), False, "unique exact normalized alias match")


class IntervalPrecision(str, Enum):
    DAY = "day"
    MONTH = "month"
    YEAR = "year"
    RANGE = "range"
    RELATIVE = "relative"


@dataclass(frozen=True)
class TemporalInterval:
    start: date
    end_exclusive: date
    precision: IntervalPrecision
    source_text: str
    reference_date: date | None = None

    def __post_init__(self) -> None:
        if self.end_exclusive <= self.start:
            raise ValueError("temporal interval must have positive duration")
        if not isinstance(self.precision, IntervalPrecision):
            object.__setattr__(self, "precision", IntervalPrecision(self.precision))
        object.__setattr__(self, "source_text", _text(self.source_text, "source_text", 2_000))


_ISO_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_MONTH = re.compile(r"^(\d{4})-(\d{2})$")
_YEAR = re.compile(r"^(\d{4})$")
_YEAR_RANGE = re.compile(r"^(\d{4})\s*(?:-|–|—|to)\s*(\d{4})$", re.IGNORECASE)
_RELATIVE_DAYS = re.compile(r"^(last|next)\s+(\d+)\s+days?$", re.IGNORECASE)


def _next_month(year: int, month: int) -> date:
    return date(year + (month == 12), 1 if month == 12 else month + 1, 1)


def normalize_temporal_expression(text: str, *, reference_date: date | None = None) -> TemporalInterval:
    """Normalize a deliberately conservative set of temporal expressions.

    Unsupported/ambiguous natural-language dates raise instead of silently choosing a
    locale or timezone interpretation.  Upstream learned/LLM normalization may propose
    an ISO value, but that proposal should pass through this deterministic validator.
    """

    source = _text(text, "temporal expression", 2_000)
    normalized = " ".join(source.casefold().split())
    day_match = _ISO_DAY.fullmatch(normalized)
    if day_match:
        start = date(*(int(part) for part in day_match.groups()))
        return TemporalInterval(start, start + timedelta(days=1), IntervalPrecision.DAY, source)
    month_match = _ISO_MONTH.fullmatch(normalized)
    if month_match:
        year, month = (int(part) for part in month_match.groups())
        start = date(year, month, 1)
        return TemporalInterval(start, _next_month(year, month), IntervalPrecision.MONTH, source)
    year_match = _YEAR.fullmatch(normalized)
    if year_match:
        year = int(year_match.group(1))
        return TemporalInterval(date(year, 1, 1), date(year + 1, 1, 1), IntervalPrecision.YEAR, source)
    range_match = _YEAR_RANGE.fullmatch(normalized)
    if range_match:
        first, last = (int(part) for part in range_match.groups())
        if last < first:
            raise ValueError("year range is reversed")
        return TemporalInterval(date(first, 1, 1), date(last + 1, 1, 1), IntervalPrecision.RANGE, source)
    if normalized in {"today", "yesterday", "tomorrow"}:
        if reference_date is None:
            raise ValueError("relative temporal expressions require reference_date")
        offsets = {"yesterday": -1, "today": 0, "tomorrow": 1}
        start = reference_date + timedelta(days=offsets[normalized])
        return TemporalInterval(start, start + timedelta(days=1), IntervalPrecision.RELATIVE, source, reference_date)
    relative_match = _RELATIVE_DAYS.fullmatch(normalized)
    if relative_match:
        if reference_date is None:
            raise ValueError("relative temporal expressions require reference_date")
        direction, count_text = relative_match.groups()
        count = int(count_text)
        if not 1 <= count <= 100_000:
            raise ValueError("relative day count is outside supported bounds")
        if direction.casefold() == "last":
            start = reference_date - timedelta(days=count)
            end = reference_date
        else:
            start = reference_date + timedelta(days=1)
            end = start + timedelta(days=count)
        return TemporalInterval(start, end, IntervalPrecision.RELATIVE, source, reference_date)
    raise ValueError("unsupported or ambiguous temporal expression; provide an explicit ISO date/range")


__all__ = [
    "DomainPrediction",
    "EntityAliasResolver",
    "EntityRecord",
    "EntityResolution",
    "FeatureVector",
    "IntervalPrecision",
    "LinearDomainClassifier",
    "LinearPlanRanker",
    "PlanKind",
    "QueryPlanCandidate",
    "TemporalInterval",
    "domain_cross_entropy",
    "listwise_plan_loss",
    "normalize_temporal_expression",
    "pairwise_plan_loss",
]
