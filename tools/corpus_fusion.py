"""Governed cross-corpus retrieval fusion and reranking cascade planning.

The important invariant is that corpora/retrievers remain independently observable.
Candidates are filtered and capped *before* reciprocal-rank fusion, fused deterministically,
then capped per document/source so one large corpus cannot silently dominate the answer.
Reranking stages are planned under explicit candidate, latency and cost budgets.

This module owns ranking mathematics and policy; it does not execute a retriever/model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Mapping, Sequence

_MAX_CANDIDATES = 2_000_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
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


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    corpus_id: str
    retriever_id: str
    document_id: str
    chunk_id: str
    rank: int
    raw_score: float | None = None
    source_id: str | None = None
    mime_type: str | None = None
    language: str | None = None
    published_date: date | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("candidate_id", "corpus_id", "retriever_id", "document_id", "chunk_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 1 <= self.rank <= _MAX_CANDIDATES:
            raise ValueError("rank must be a positive bounded integer")
        if self.raw_score is not None:
            object.__setattr__(self, "raw_score", _finite(self.raw_score, "raw_score"))
        for name in ("source_id", "mime_type", "language"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 2_000:
            raise ValueError("metadata must be a bounded mapping")
        cleaned = {
            _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
            for key, value in self.metadata.items()
        }
        object.__setattr__(self, "metadata", cleaned)


@dataclass(frozen=True)
class CandidateFilter:
    allowed_corpora: frozenset[str] | None = None
    allowed_mime_types: frozenset[str] | None = None
    allowed_languages: frozenset[str] | None = None
    published_on_or_after: date | None = None
    published_before: date | None = None
    required_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("allowed_corpora", "allowed_mime_types", "allowed_languages"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, frozenset(_identifier(value, name) for value in values))
        if self.published_on_or_after and self.published_before and self.published_before <= self.published_on_or_after:
            raise ValueError("published date window is empty or reversed")
        if not isinstance(self.required_metadata, Mapping) or len(self.required_metadata) > 1_000:
            raise ValueError("required_metadata must be bounded")
        object.__setattr__(
            self,
            "required_metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in self.required_metadata.items()
            },
        )

    def accepts(self, candidate: RetrievalCandidate) -> bool:
        if self.allowed_corpora is not None and candidate.corpus_id not in self.allowed_corpora:
            return False
        if self.allowed_mime_types is not None and candidate.mime_type not in self.allowed_mime_types:
            return False
        if self.allowed_languages is not None and candidate.language not in self.allowed_languages:
            return False
        if self.published_on_or_after is not None:
            if candidate.published_date is None or candidate.published_date < self.published_on_or_after:
                return False
        if self.published_before is not None:
            if candidate.published_date is None or candidate.published_date >= self.published_before:
                return False
        return all(candidate.metadata.get(key) == value for key, value in self.required_metadata.items())


@dataclass(frozen=True)
class FusionPolicy:
    rrf_k: int = 60
    max_per_input_list: int = 1_000
    max_fused_candidates: int = 1_000
    max_per_document: int = 3
    max_per_source: int | None = None
    corpus_weights: Mapping[str, float] = field(default_factory=dict)
    retriever_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, minimum in (
            ("rrf_k", 1),
            ("max_per_input_list", 1),
            ("max_fused_candidates", 1),
            ("max_per_document", 1),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= _MAX_CANDIDATES:
                raise ValueError(f"{name} is invalid")
        if self.max_per_source is not None:
            if isinstance(self.max_per_source, bool) or not isinstance(self.max_per_source, int) or not 1 <= self.max_per_source <= _MAX_CANDIDATES:
                raise ValueError("max_per_source is invalid")
        for name in ("corpus_weights", "retriever_weights"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping) or len(mapping) > 100_000:
                raise ValueError(f"{name} must be a bounded mapping")
            cleaned: dict[str, float] = {}
            for key, raw_value in mapping.items():
                value = _finite(raw_value, f"{name} weight")
                if value < 0.0:
                    raise ValueError(f"{name} weights must be non-negative")
                cleaned[_identifier(key, f"{name} key")] = value
            object.__setattr__(self, name, cleaned)


@dataclass(frozen=True)
class FusionContribution:
    corpus_id: str
    retriever_id: str
    rank: int
    weight: float
    contribution: float


@dataclass(frozen=True)
class FusedCandidate:
    candidate: RetrievalCandidate
    fused_score: float
    best_rank: int
    contributions: tuple[FusionContribution, ...]


def _weight(candidate: RetrievalCandidate, policy: FusionPolicy) -> float:
    return policy.corpus_weights.get(candidate.corpus_id, 1.0) * policy.retriever_weights.get(candidate.retriever_id, 1.0)


def reciprocal_rank_fuse(
    ranked_lists: Mapping[str, Sequence[RetrievalCandidate]],
    *,
    policy: FusionPolicy = FusionPolicy(),
    candidate_filter: CandidateFilter | None = None,
) -> tuple[FusedCandidate, ...]:
    """Fuse independent ranked lists with deterministic weighted reciprocal-rank fusion."""

    if not ranked_lists or len(ranked_lists) > 100_000:
        raise ValueError("ranked_lists must be a non-empty bounded mapping")
    states: dict[tuple[str, str], dict[str, Any]] = {}
    seen_candidate_ids: dict[str, tuple[str, str]] = {}
    for list_id in sorted(ranked_lists):
        _identifier(list_id, "ranked list id")
        entries = ranked_lists[list_id]
        if len(entries) > _MAX_CANDIDATES:
            raise ValueError("input ranked list is too large")
        if any(not isinstance(candidate, RetrievalCandidate) for candidate in entries):
            raise ValueError("ranked lists must contain RetrievalCandidate values")
        accepted = [candidate for candidate in entries if candidate_filter is None or candidate_filter.accepts(candidate)]
        accepted.sort(key=lambda candidate: (candidate.rank, candidate.candidate_id))
        for local_rank, candidate in enumerate(accepted[: policy.max_per_input_list], start=1):
            identity = (candidate.document_id, candidate.chunk_id)
            previous_identity = seen_candidate_ids.get(candidate.candidate_id)
            if previous_identity is not None and previous_identity != identity:
                raise ValueError("candidate_id maps to inconsistent document/chunk identities")
            seen_candidate_ids[candidate.candidate_id] = identity
            weight = _weight(candidate, policy)
            contribution = weight / (policy.rrf_k + local_rank)
            state = states.setdefault(
                identity,
                {"candidate": candidate, "score": 0.0, "best_rank": local_rank, "contributions": []},
            )
            previous_best_rank = state["best_rank"]
            current = state["candidate"]
            if (local_rank, candidate.candidate_id) < (previous_best_rank, current.candidate_id):
                state["candidate"] = candidate
            state["best_rank"] = min(previous_best_rank, local_rank)
            state["score"] += contribution
            state["contributions"].append(
                FusionContribution(candidate.corpus_id, candidate.retriever_id, local_rank, weight, contribution)
            )
    fused = [
        FusedCandidate(
            candidate=state["candidate"],
            fused_score=state["score"],
            best_rank=state["best_rank"],
            contributions=tuple(
                sorted(
                    state["contributions"],
                    key=lambda item: (item.corpus_id, item.retriever_id, item.rank),
                )
            ),
        )
        for state in states.values()
    ]
    fused.sort(
        key=lambda item: (
            -item.fused_score,
            item.best_rank,
            item.candidate.document_id,
            item.candidate.chunk_id,
            item.candidate.candidate_id,
        )
    )
    document_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    selected: list[FusedCandidate] = []
    for item in fused:
        document_id = item.candidate.document_id
        if document_counts.get(document_id, 0) >= policy.max_per_document:
            continue
        source_id = item.candidate.source_id
        if policy.max_per_source is not None and source_id is not None:
            if source_counts.get(source_id, 0) >= policy.max_per_source:
                continue
        selected.append(item)
        document_counts[document_id] = document_counts.get(document_id, 0) + 1
        if source_id is not None:
            source_counts[source_id] = source_counts.get(source_id, 0) + 1
        if len(selected) >= policy.max_fused_candidates:
            break
    return tuple(selected)


class CascadeStageKind(str, Enum):
    CHEAP_RERANK = "cheap_rerank"
    CROSS_ENCODER = "cross_encoder"
    LISTWISE = "listwise"
    SEMANTIC_SUPPORT = "semantic_support"


@dataclass(frozen=True)
class CascadeStage:
    name: str
    kind: CascadeStageKind
    input_cap: int
    output_cap: int
    estimated_ms_per_candidate: float
    estimated_cost_per_candidate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "cascade stage name", 500))
        if not isinstance(self.kind, CascadeStageKind):
            object.__setattr__(self, "kind", CascadeStageKind(self.kind))
        for name in ("input_cap", "output_cap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_CANDIDATES:
                raise ValueError(f"{name} is invalid")
        if self.output_cap > self.input_cap:
            raise ValueError("cascade output_cap may not exceed input_cap")
        for name in ("estimated_ms_per_candidate", "estimated_cost_per_candidate"):
            value = _finite(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PlannedCascadeStage:
    stage: CascadeStage
    planned_input_count: int
    planned_output_count: int
    estimated_latency_ms: float
    estimated_cost: float


@dataclass(frozen=True)
class CascadePlan:
    stages: tuple[PlannedCascadeStage, ...]
    final_candidate_count: int
    estimated_latency_ms: float
    estimated_cost: float
    truncated_by_budget: bool


def plan_rerank_cascade(
    initial_candidate_count: int,
    stages: Sequence[CascadeStage],
    *,
    max_latency_ms: float,
    max_cost: float,
) -> CascadePlan:
    """Plan the longest prefix of a rerank cascade that fits explicit budgets."""

    if isinstance(initial_candidate_count, bool) or not isinstance(initial_candidate_count, int) or not 0 <= initial_candidate_count <= _MAX_CANDIDATES:
        raise ValueError("initial_candidate_count is invalid")
    latency_budget = _finite(max_latency_ms, "max_latency_ms")
    cost_budget = _finite(max_cost, "max_cost")
    if latency_budget < 0.0 or cost_budget < 0.0:
        raise ValueError("cascade budgets must be non-negative")
    if len(stages) > 100:
        raise ValueError("too many cascade stages")
    current = initial_candidate_count
    total_latency = 0.0
    total_cost = 0.0
    planned: list[PlannedCascadeStage] = []
    truncated = False
    for stage in stages:
        if not isinstance(stage, CascadeStage):
            raise ValueError("stages must contain CascadeStage values")
        count = min(current, stage.input_cap)
        stage_latency = count * stage.estimated_ms_per_candidate
        stage_cost = count * stage.estimated_cost_per_candidate
        if total_latency + stage_latency > latency_budget or total_cost + stage_cost > cost_budget:
            truncated = True
            break
        output = min(count, stage.output_cap)
        planned.append(PlannedCascadeStage(stage, count, output, stage_latency, stage_cost))
        total_latency += stage_latency
        total_cost += stage_cost
        current = output
    return CascadePlan(tuple(planned), current, total_latency, total_cost, truncated)


__all__ = [
    "CandidateFilter",
    "CascadePlan",
    "CascadeStage",
    "CascadeStageKind",
    "FusedCandidate",
    "FusionContribution",
    "FusionPolicy",
    "PlannedCascadeStage",
    "RetrievalCandidate",
    "plan_rerank_cascade",
    "reciprocal_rank_fuse",
]
