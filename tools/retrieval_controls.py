"""Structured candidate filtering and budget-aware reranker cascades."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from tools.hybrid_retrieval import RetrievalCandidate

_MAX_CANDIDATES = 500
_MAX_STAGES = 12


def _exact_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _finite(value: Any, label: str, minimum: float = 0.0, maximum: float = 1_000_000.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _text_set(values: Sequence[str], label: str) -> frozenset[str]:
    if isinstance(values, (str, bytes, bytearray)) or len(values) > 64:
        raise ValueError(f"{label} must be a bounded sequence.")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{label} values must be strings.")
        text = value.strip().lower()
        if not text or len(text) > 500:
            raise ValueError(f"{label} contains an invalid value.")
        result.add(text)
    return frozenset(result)


def _timestamp(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return float(text)
    except ValueError:
        pass
    try:
        parsed_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed_dt.tzinfo is None:
        parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
    return parsed_dt.timestamp()


@dataclass(frozen=True)
class StructuredFilter:
    """Metadata constraints applied before expensive retrieval stages."""

    mime_types: Sequence[str] = ()
    sections: Sequence[str] = ()
    provenance: Sequence[str] = ()
    min_page: int | None = None
    max_page: int | None = None
    modified_after: float | str | None = None
    modified_before: float | str | None = None
    metadata_equals: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mime_types", _text_set(tuple(self.mime_types), "mime_types"))
        object.__setattr__(self, "sections", _text_set(tuple(self.sections), "sections"))
        object.__setattr__(self, "provenance", _text_set(tuple(self.provenance), "provenance"))
        if self.min_page is not None:
            object.__setattr__(self, "min_page", _exact_int(self.min_page, "min_page", 0, 10_000_000))
        if self.max_page is not None:
            object.__setattr__(self, "max_page", _exact_int(self.max_page, "max_page", 0, 10_000_000))
        if self.min_page is not None and self.max_page is not None and self.min_page > self.max_page:
            raise ValueError("min_page cannot exceed max_page.")
        after = _timestamp(self.modified_after)
        before = _timestamp(self.modified_before)
        if self.modified_after is not None and after is None:
            raise ValueError("modified_after must be a timestamp or ISO-8601 datetime.")
        if self.modified_before is not None and before is None:
            raise ValueError("modified_before must be a timestamp or ISO-8601 datetime.")
        if after is not None and before is not None and after > before:
            raise ValueError("modified_after cannot exceed modified_before.")
        object.__setattr__(self, "modified_after", after)
        object.__setattr__(self, "modified_before", before)
        if not isinstance(self.metadata_equals, Mapping) or len(self.metadata_equals) > 32:
            raise ValueError("metadata_equals must be a bounded mapping.")
        normalized: dict[str, Any] = {}
        for key, value in self.metadata_equals.items():
            if not isinstance(key, str) or not key.strip() or len(key) > 200:
                raise ValueError("metadata_equals contains an invalid key.")
            normalized[key.strip()] = value
        object.__setattr__(self, "metadata_equals", normalized)


def _metadata_text(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def candidate_matches(candidate: RetrievalCandidate, spec: StructuredFilter) -> bool:
    if not isinstance(candidate, RetrievalCandidate) or not isinstance(spec, StructuredFilter):
        raise ValueError("candidate/spec type mismatch.")
    metadata = candidate.metadata
    if spec.mime_types and _metadata_text(metadata, "mime_type", "mime", "content_type") not in spec.mime_types:
        return False
    if spec.sections and _metadata_text(metadata, "section", "section_name") not in spec.sections:
        return False
    if spec.provenance and _metadata_text(metadata, "provenance", "source_type", "origin") not in spec.provenance:
        return False
    raw_page = metadata.get("page_number", metadata.get("page"))
    page: int | None = None
    if raw_page is not None and not isinstance(raw_page, bool):
        try:
            page = int(operator.index(raw_page))
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = float(raw_page)
                page = int(parsed) if parsed.is_integer() else None
            except (TypeError, ValueError, OverflowError):
                page = None
    if spec.min_page is not None and (page is None or page < spec.min_page):
        return False
    if spec.max_page is not None and (page is None or page > spec.max_page):
        return False
    if spec.modified_after is not None or spec.modified_before is not None:
        modified = _timestamp(metadata.get("modified_at", metadata.get("timestamp", metadata.get("date"))))
        if modified is None:
            return False
        if spec.modified_after is not None and modified < spec.modified_after:
            return False
        if spec.modified_before is not None and modified > spec.modified_before:
            return False
    return all(metadata.get(key) == expected for key, expected in spec.metadata_equals.items())


def filter_candidates(candidates: Sequence[RetrievalCandidate], spec: StructuredFilter | None = None) -> tuple[RetrievalCandidate, ...]:
    if isinstance(candidates, (str, bytes, bytearray)) or len(candidates) > _MAX_CANDIDATES:
        raise ValueError("candidates must be a bounded sequence.")
    selected_spec = spec or StructuredFilter()
    result: list[RetrievalCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, RetrievalCandidate):
            raise ValueError("candidates contains an invalid value.")
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        if candidate_matches(candidate, selected_spec):
            result.append(candidate)
    return tuple(result)


RerankerCallable = Callable[[str, Sequence[RetrievalCandidate]], Mapping[str, float]]


@dataclass(frozen=True)
class RerankerStage:
    name: str
    scorer: RerankerCallable
    candidate_limit: int = 100
    cost_units: float = 1.0
    required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
            raise ValueError("stage name is invalid.")
        object.__setattr__(self, "name", self.name.strip())
        if not callable(self.scorer):
            raise ValueError("stage scorer must be callable.")
        object.__setattr__(self, "candidate_limit", _exact_int(self.candidate_limit, "candidate_limit", 1, _MAX_CANDIDATES))
        object.__setattr__(self, "cost_units", _finite(self.cost_units, "cost_units"))
        if not isinstance(self.required, bool):
            raise ValueError("required must be boolean.")


@dataclass(frozen=True)
class CascadeResult:
    candidates: tuple[RetrievalCandidate, ...]
    scores: Mapping[str, float]
    stages_run: tuple[str, ...]
    stages_skipped: tuple[str, ...]
    cost_units: float
    early_stopped: bool


def run_reranker_cascade(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    stages: Sequence[RerankerStage],
    *,
    top_k: int = 10,
    max_cost_units: float = 10.0,
    early_stop_margin: float | None = None,
) -> CascadeResult:
    """Run increasingly expensive rerankers until budget/confidence terminates."""

    if not isinstance(query, str) or not query.strip() or len(query) > 20_000:
        raise ValueError("query must be a bounded non-empty string.")
    current = list(filter_candidates(candidates))
    requested = _exact_int(top_k, "top_k", 1, _MAX_CANDIDATES)
    if isinstance(stages, (str, bytes, bytearray)) or len(stages) > _MAX_STAGES:
        raise ValueError("stages must be a bounded sequence.")
    stage_values = tuple(stages)
    if any(not isinstance(stage, RerankerStage) for stage in stage_values):
        raise ValueError("stages contains an invalid value.")
    budget = _finite(max_cost_units, "max_cost_units")
    margin = None if early_stop_margin is None else _finite(early_stop_margin, "early_stop_margin", 0.0, 1.0)
    scores = {item.candidate_id: item.dense_score for item in current}
    spent = 0.0
    ran: list[str] = []
    skipped: list[str] = []
    early_stopped = False
    for stage in stage_values:
        if not current:
            break
        if spent + stage.cost_units > budget:
            if stage.required:
                raise RuntimeError(f"required reranker stage {stage.name!r} exceeds the budget.")
            skipped.append(stage.name)
            continue
        pool = current[: stage.candidate_limit]
        try:
            raw = stage.scorer(query, pool)
        except Exception as exc:
            if stage.required:
                raise RuntimeError(f"required reranker stage {stage.name!r} failed.") from exc
            skipped.append(stage.name)
            continue
        if not isinstance(raw, Mapping):
            if stage.required:
                raise RuntimeError(f"required reranker stage {stage.name!r} returned invalid scores.")
            skipped.append(stage.name)
            continue
        ids = {item.candidate_id for item in pool}
        sanitized: dict[str, float] = {}
        for candidate_id, value in raw.items():
            if candidate_id not in ids or isinstance(value, bool):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(parsed):
                sanitized[candidate_id] = max(0.0, min(parsed, 1.0))
        if stage.required and len(sanitized) < len(pool):
            raise RuntimeError(f"required reranker stage {stage.name!r} omitted candidate scores.")
        scores.update(sanitized)
        current.sort(key=lambda item: (scores.get(item.candidate_id, 0.0), item.dense_score, item.candidate_id), reverse=True)
        spent += stage.cost_units
        ran.append(stage.name)
        if margin is not None and len(current) > 1:
            first = scores.get(current[0].candidate_id, 0.0)
            second = scores.get(current[1].candidate_id, 0.0)
            if first - second >= margin:
                early_stopped = True
                break
    selected = tuple(current[: min(requested, len(current))])
    return CascadeResult(
        candidates=selected,
        scores={item.candidate_id: scores.get(item.candidate_id, 0.0) for item in selected},
        stages_run=tuple(ran),
        stages_skipped=tuple(skipped),
        cost_units=spent,
        early_stopped=early_stopped,
    )


__all__ = [
    "CascadeResult",
    "RerankerStage",
    "StructuredFilter",
    "candidate_matches",
    "filter_candidates",
    "run_reranker_cascade",
]
