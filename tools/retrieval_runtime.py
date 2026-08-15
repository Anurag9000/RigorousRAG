"""Shared retrieval control plane: filters, query variants, budgets, fusion and cascades.

The module operates on already owner-authorized candidates.  It contains no model
weights, downloads or training loop; learned implementations plug into the explicit
policy/transform/reranker protocols while deterministic fallbacks remain available.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from tools.hybrid_retrieval import RetrievalCandidate
from tools.reranking import Reranker

_MAX_FILTER_NODES = 128
_MAX_FILTER_VALUES = 256
_MAX_QUERY_CHARS = 20_000
_MAX_VARIANTS = 32
_MAX_CANDIDATES = 2_000
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._:/+-][A-Za-z0-9]+)*")
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9-]{1,15}\b")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _bounded_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is empty or exceeds {maximum} characters")
    return cleaned


def _finite(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return parsed


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return parsed


def _metadata_path(metadata: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = metadata
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return False, None
        current = current[component]
    return True, current


@dataclass(frozen=True)
class FilterExpression:
    """Closed, backend-neutral metadata filter AST."""

    op: str
    field: str = ""
    value: Any = None
    values: tuple[Any, ...] = ()
    children: tuple["FilterExpression", ...] = ()

    def __post_init__(self) -> None:
        selected = self.op.strip().lower() if isinstance(self.op, str) else ""
        allowed = {"eq", "ne", "lt", "lte", "gt", "gte", "in", "contains", "exists", "and", "or", "not"}
        if selected not in allowed:
            raise ValueError("unsupported filter operation")
        object.__setattr__(self, "op", selected)
        if selected in {"and", "or", "not"}:
            if self.field or self.values or self.value is not None:
                raise ValueError("logical filter nodes may contain only children")
            expected_min = 1
            expected_max = 1 if selected == "not" else 32
            if not expected_min <= len(self.children) <= expected_max:
                raise ValueError("logical filter has an invalid child count")
        else:
            path = _bounded_text(self.field, "filter field", 300)
            if any(part in {"__class__", "__dict__", "__globals__"} or part.startswith("_") for part in path.split(".")):
                raise ValueError("filter field is forbidden")
            object.__setattr__(self, "field", path)
            if self.children:
                raise ValueError("leaf filter nodes may not contain children")
            if selected == "in":
                if not 1 <= len(self.values) <= _MAX_FILTER_VALUES:
                    raise ValueError("in filter values are empty or exceed the limit")
            elif self.values:
                raise ValueError("values are only valid for in filters")

    @property
    def node_count(self) -> int:
        return 1 + sum(child.node_count for child in self.children)

    @property
    def fingerprint(self) -> str:
        if self.node_count > _MAX_FILTER_NODES:
            raise ValueError("filter AST exceeds the node limit")
        return hashlib.sha256(_canonical_json(asdict(self))).hexdigest()

    def evaluate(self, metadata: Mapping[str, Any]) -> bool:
        if self.node_count > _MAX_FILTER_NODES:
            raise ValueError("filter AST exceeds the node limit")
        if not isinstance(metadata, Mapping):
            return False
        if self.op == "and":
            return all(child.evaluate(metadata) for child in self.children)
        if self.op == "or":
            return any(child.evaluate(metadata) for child in self.children)
        if self.op == "not":
            return not self.children[0].evaluate(metadata)
        exists, actual = _metadata_path(metadata, self.field)
        if self.op == "exists":
            return exists is bool(self.value if self.value is not None else True)
        if not exists:
            return False
        try:
            if self.op == "eq":
                return actual == self.value
            if self.op == "ne":
                return actual != self.value
            if self.op == "in":
                return actual in self.values
            if self.op == "contains":
                if isinstance(actual, str) and isinstance(self.value, str):
                    return self.value.casefold() in actual.casefold()
                if isinstance(actual, (list, tuple, set, frozenset)):
                    return self.value in actual
                return False
            if self.op in {"lt", "lte", "gt", "gte"}:
                left = _finite(actual, "metadata value")
                right = _finite(self.value, "filter value")
                if self.op == "lt":
                    return left < right
                if self.op == "lte":
                    return left <= right
                if self.op == "gt":
                    return left > right
                return left >= right
        except (TypeError, ValueError, OverflowError):
            return False
        return False


def filter_candidates(
    candidates: Iterable[RetrievalCandidate],
    expression: FilterExpression | None,
    *,
    limit: int = 500,
) -> tuple[RetrievalCandidate, ...]:
    maximum = _integer(limit, "limit", 1, _MAX_CANDIDATES)
    output: list[RetrievalCandidate] = []
    for candidate in candidates:
        if len(output) >= maximum:
            break
        if not isinstance(candidate, RetrievalCandidate):
            continue
        if expression is None or expression.evaluate(candidate.metadata):
            output.append(candidate)
    return tuple(output)


class QueryTransformProvider(Protocol):
    def transform(self, query: str, *, mode: str, context: Mapping[str, Any]) -> Sequence[str]: ...


@dataclass(frozen=True)
class QueryVariant:
    text: str
    strategy: str
    parent_index: int = 0
    generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _bounded_text(self.text, "query variant", _MAX_QUERY_CHARS))
        object.__setattr__(self, "strategy", _bounded_text(self.strategy, "strategy", 64).lower())
        object.__setattr__(self, "parent_index", _integer(self.parent_index, "parent_index", 0, _MAX_VARIANTS - 1))
        if not isinstance(self.generated, bool):
            raise ValueError("generated must be boolean")


@dataclass(frozen=True)
class QueryTransformPlan:
    original_query: str
    variants: tuple[QueryVariant, ...]
    fingerprint: str


def build_query_transform_plan(
    query: str,
    *,
    acronym_map: Mapping[str, str] | None = None,
    synonym_map: Mapping[str, Sequence[str]] | None = None,
    provider: QueryTransformProvider | None = None,
    provider_modes: Sequence[str] = (),
    context: Mapping[str, Any] | None = None,
    max_variants: int = 12,
) -> QueryTransformPlan:
    original = _bounded_text(query, "query", _MAX_QUERY_CHARS)
    maximum = _integer(max_variants, "max_variants", 1, _MAX_VARIANTS)
    variants: list[QueryVariant] = [QueryVariant(original, "original")]
    seen = {original.casefold()}

    def add(text: str, strategy: str, generated: bool = False) -> None:
        if len(variants) >= maximum:
            return
        cleaned = _bounded_text(text, "query variant", _MAX_QUERY_CHARS)
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            variants.append(QueryVariant(cleaned, strategy, generated=generated))

    if acronym_map:
        replacements = []
        for acronym in _ACRONYM_RE.findall(original):
            expansion = acronym_map.get(acronym) or acronym_map.get(acronym.casefold())
            if isinstance(expansion, str) and expansion.strip():
                replacements.append((acronym, _bounded_text(expansion, "acronym expansion", 500)))
        if replacements:
            expanded = original
            for acronym, expansion in replacements:
                expanded = re.sub(rf"\b{re.escape(acronym)}\b", f"{acronym} ({expansion})", expanded)
            add(expanded, "acronym_expansion")

    if synonym_map:
        tokens = _TOKEN_RE.findall(original)
        additions: list[str] = []
        for token in tokens:
            values = synonym_map.get(token) or synonym_map.get(token.casefold()) or ()
            for value in list(values)[:4]:
                if isinstance(value, str) and value.strip():
                    additions.append(_bounded_text(value, "synonym", 200))
        if additions:
            add(f"{original} {' '.join(dict.fromkeys(additions))}", "synonym_expansion")

    if provider is not None:
        safe_context = dict(context or {})
        for raw_mode in list(provider_modes)[:8]:
            mode = _bounded_text(raw_mode, "provider mode", 64).lower()
            if mode not in {"rewrite", "multi_query", "hyde", "terminology", "citation_chase", "pseudo_relevance"}:
                raise ValueError("unsupported provider transform mode")
            if len(variants) >= maximum:
                break
            try:
                produced = provider.transform(original, mode=mode, context=safe_context)
            except Exception:
                continue
            if isinstance(produced, (str, bytes, bytearray)):
                continue
            for text in list(produced)[: maximum - len(variants)]:
                if isinstance(text, str):
                    add(text, mode, generated=True)

    payload = {"original_query": original, "variants": [asdict(item) for item in variants]}
    return QueryTransformPlan(original, tuple(variants), hashlib.sha256(_canonical_json(payload)).hexdigest())


@dataclass(frozen=True)
class BudgetLimits:
    max_wall_ms: float = 30_000.0
    max_calls: int = 32
    max_input_tokens: int = 100_000
    max_output_tokens: int = 50_000
    max_cost: float = 10.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_wall_ms", _finite(self.max_wall_ms, "max_wall_ms", minimum=1.0, maximum=86_400_000.0))
        object.__setattr__(self, "max_calls", _integer(self.max_calls, "max_calls", 1, 100_000))
        object.__setattr__(self, "max_input_tokens", _integer(self.max_input_tokens, "max_input_tokens", 0, 10**9))
        object.__setattr__(self, "max_output_tokens", _integer(self.max_output_tokens, "max_output_tokens", 0, 10**9))
        object.__setattr__(self, "max_cost", _finite(self.max_cost, "max_cost", minimum=0.0, maximum=1_000_000.0))


@dataclass(frozen=True)
class BudgetSnapshot:
    elapsed_ms: float
    calls: int
    input_tokens: int
    output_tokens: int
    cost: float
    exhausted_reasons: tuple[str, ...]

    @property
    def exhausted(self) -> bool:
        return bool(self.exhausted_reasons)


class RuntimeBudget:
    """Monotonic accounting ledger shared across retrieval/planning/reranking stages."""

    def __init__(self, limits: BudgetLimits, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not isinstance(limits, BudgetLimits):
            raise TypeError("limits must be BudgetLimits")
        self.limits = limits
        self._clock = clock
        self._started = clock()
        self._calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cost = 0.0

    def snapshot(self) -> BudgetSnapshot:
        elapsed = max(0.0, (self._clock() - self._started) * 1000.0)
        reasons: list[str] = []
        if elapsed >= self.limits.max_wall_ms:
            reasons.append("wall_time")
        if self._calls >= self.limits.max_calls:
            reasons.append("calls")
        if self._input_tokens >= self.limits.max_input_tokens and self.limits.max_input_tokens >= 0:
            reasons.append("input_tokens")
        if self._output_tokens >= self.limits.max_output_tokens and self.limits.max_output_tokens >= 0:
            reasons.append("output_tokens")
        if self._cost >= self.limits.max_cost:
            reasons.append("cost")
        return BudgetSnapshot(elapsed, self._calls, self._input_tokens, self._output_tokens, self._cost, tuple(reasons))

    def reserve(self, *, calls: int = 1, input_tokens: int = 0, output_tokens: int = 0, cost: float = 0.0) -> None:
        add_calls = _integer(calls, "calls", 0, 100_000)
        add_input = _integer(input_tokens, "input_tokens", 0, 10**9)
        add_output = _integer(output_tokens, "output_tokens", 0, 10**9)
        add_cost = _finite(cost, "cost", minimum=0.0, maximum=1_000_000.0)
        current = self.snapshot()
        if current.elapsed_ms >= self.limits.max_wall_ms:
            raise RuntimeError("runtime budget exhausted: wall_time")
        if self._calls + add_calls > self.limits.max_calls:
            raise RuntimeError("runtime budget exhausted: calls")
        if self._input_tokens + add_input > self.limits.max_input_tokens:
            raise RuntimeError("runtime budget exhausted: input_tokens")
        if self._output_tokens + add_output > self.limits.max_output_tokens:
            raise RuntimeError("runtime budget exhausted: output_tokens")
        if self._cost + add_cost > self.limits.max_cost:
            raise RuntimeError("runtime budget exhausted: cost")
        self._calls += add_calls
        self._input_tokens += add_input
        self._output_tokens += add_output
        self._cost += add_cost

    def allocate_fraction(self, fraction: float) -> BudgetLimits:
        selected = _finite(fraction, "fraction", minimum=0.0, maximum=1.0)
        snapshot = self.snapshot()
        return BudgetLimits(
            max_wall_ms=max(1.0, (self.limits.max_wall_ms - snapshot.elapsed_ms) * selected),
            max_calls=max(1, int((self.limits.max_calls - snapshot.calls) * selected)),
            max_input_tokens=max(0, int((self.limits.max_input_tokens - snapshot.input_tokens) * selected)),
            max_output_tokens=max(0, int((self.limits.max_output_tokens - snapshot.output_tokens) * selected)),
            max_cost=max(0.0, (self.limits.max_cost - snapshot.cost) * selected),
        )


@dataclass(frozen=True)
class CandidateFeatures:
    candidate_id: str
    scores: Mapping[str, float]
    source_id: str = ""
    document_id: str = ""
    modality: str = "text"
    freshness: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _bounded_text(self.candidate_id, "candidate_id", 500))
        if not isinstance(self.scores, Mapping) or len(self.scores) > 64:
            raise ValueError("scores must be a bounded mapping")
        normalized: dict[str, float] = {}
        for name, value in self.scores.items():
            normalized[_bounded_text(name, "score name", 100)] = _finite(value, "score", minimum=0.0, maximum=1.0)
        object.__setattr__(self, "scores", normalized)
        if self.source_id:
            object.__setattr__(self, "source_id", _bounded_text(self.source_id, "source_id", 500))
        if self.document_id:
            object.__setattr__(self, "document_id", _bounded_text(self.document_id, "document_id", 500))
        object.__setattr__(self, "modality", _bounded_text(self.modality, "modality", 64).lower())
        object.__setattr__(self, "freshness", _finite(self.freshness, "freshness", minimum=0.0, maximum=1.0))


class FusionPolicy(Protocol):
    def score(self, features: CandidateFeatures) -> float: ...


@dataclass(frozen=True)
class LinearFusionPolicy:
    weights: Mapping[str, float]
    bias: float = 0.0
    freshness_weight: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.weights, Mapping) or not self.weights or len(self.weights) > 64:
            raise ValueError("weights must be a non-empty bounded mapping")
        normalized = {
            _bounded_text(name, "weight name", 100): _finite(value, "weight", minimum=-100.0, maximum=100.0)
            for name, value in self.weights.items()
        }
        object.__setattr__(self, "weights", normalized)
        object.__setattr__(self, "bias", _finite(self.bias, "bias", minimum=-100.0, maximum=100.0))
        object.__setattr__(self, "freshness_weight", _finite(self.freshness_weight, "freshness_weight", minimum=-100.0, maximum=100.0))

    def score(self, features: CandidateFeatures) -> float:
        raw = self.bias + self.freshness_weight * features.freshness
        for name, weight in self.weights.items():
            raw += weight * features.scores.get(name, 0.0)
        if raw >= 0.0:
            value = 1.0 / (1.0 + math.exp(-min(raw, 700.0)))
        else:
            exp_value = math.exp(max(raw, -700.0))
            value = exp_value / (1.0 + exp_value)
        return max(0.0, min(value, 1.0))


def fuse_features(
    rows: Sequence[CandidateFeatures],
    policy: FusionPolicy,
    *,
    max_per_source: int = 0,
    max_per_document: int = 0,
) -> tuple[tuple[str, float], ...]:
    if len(rows) > _MAX_CANDIDATES:
        raise ValueError("feature rows exceed the candidate limit")
    source_cap = _integer(max_per_source, "max_per_source", 0, _MAX_CANDIDATES)
    document_cap = _integer(max_per_document, "max_per_document", 0, _MAX_CANDIDATES)
    scored: list[tuple[CandidateFeatures, float]] = []
    for row in rows:
        score = _finite(policy.score(row), "fusion score", minimum=0.0, maximum=1.0)
        scored.append((row, score))
    scored.sort(key=lambda item: (-item[1], item[0].candidate_id))
    source_counts: dict[str, int] = {}
    document_counts: dict[str, int] = {}
    output: list[tuple[str, float]] = []
    for row, score in scored:
        if source_cap and row.source_id and source_counts.get(row.source_id, 0) >= source_cap:
            continue
        if document_cap and row.document_id and document_counts.get(row.document_id, 0) >= document_cap:
            continue
        output.append((row.candidate_id, score))
        if row.source_id:
            source_counts[row.source_id] = source_counts.get(row.source_id, 0) + 1
        if row.document_id:
            document_counts[row.document_id] = document_counts.get(row.document_id, 0) + 1
    return tuple(output)


@dataclass(frozen=True)
class CascadeStage:
    name: str
    reranker: Reranker
    keep: int
    min_top_score: float = 0.0
    estimated_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _bounded_text(self.name, "stage name", 100))
        object.__setattr__(self, "keep", _integer(self.keep, "keep", 1, _MAX_CANDIDATES))
        object.__setattr__(self, "min_top_score", _finite(self.min_top_score, "min_top_score", minimum=0.0, maximum=1.0))
        object.__setattr__(self, "estimated_cost", _finite(self.estimated_cost, "estimated_cost", minimum=0.0, maximum=1_000_000.0))


@dataclass(frozen=True)
class CascadeTrace:
    stage: str
    input_count: int
    output_count: int
    top_score: float
    escalated: bool


@dataclass(frozen=True)
class CascadeResult:
    candidates: tuple[RetrievalCandidate, ...]
    scores: Mapping[str, float]
    traces: tuple[CascadeTrace, ...]


def run_reranker_cascade(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    stages: Sequence[CascadeStage],
    *,
    budget: RuntimeBudget | None = None,
) -> CascadeResult:
    selected_query = _bounded_text(query, "query", _MAX_QUERY_CHARS)
    current = list(candidates[:_MAX_CANDIDATES])
    scores: dict[str, float] = {item.candidate_id: item.dense_score for item in current}
    traces: list[CascadeTrace] = []
    for stage_index, stage in enumerate(stages[:16]):
        if not current:
            break
        if budget is not None:
            try:
                budget.reserve(calls=1, cost=stage.estimated_cost)
            except RuntimeError:
                break
        try:
            raw_scores = stage.reranker.score(selected_query, current)
        except Exception:
            continue
        normalized: list[tuple[RetrievalCandidate, float]] = []
        for candidate in current:
            raw = raw_scores.get(candidate.candidate_id, 0.0)
            try:
                score = _finite(raw, "reranker score")
            except ValueError:
                score = 0.0
            score = max(0.0, min(score, 1.0))
            normalized.append((candidate, score))
            scores[candidate.candidate_id] = score
        normalized.sort(key=lambda item: (-item[1], item[0].candidate_id))
        current = [candidate for candidate, _ in normalized[: stage.keep]]
        top_score = normalized[0][1] if normalized else 0.0
        escalated = stage_index + 1 < len(stages) and top_score < stage.min_top_score
        traces.append(CascadeTrace(stage.name, len(normalized), len(current), top_score, escalated))
        if not escalated:
            break
    return CascadeResult(tuple(current), scores, tuple(traces))


__all__ = [
    "BudgetLimits",
    "BudgetSnapshot",
    "CandidateFeatures",
    "CascadeResult",
    "CascadeStage",
    "CascadeTrace",
    "FilterExpression",
    "FusionPolicy",
    "LinearFusionPolicy",
    "QueryTransformPlan",
    "QueryTransformProvider",
    "QueryVariant",
    "RuntimeBudget",
    "build_query_transform_plan",
    "filter_candidates",
    "fuse_features",
    "run_reranker_cascade",
]
