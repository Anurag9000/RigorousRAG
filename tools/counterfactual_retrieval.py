"""Deterministic audits of already-executed retrieval alternatives.

The audit compares two retrieval snapshots by stable evidence identity and reports source,
rank, route, citation, and support changes. It is intentionally descriptive: a difference
between snapshots is not proof that a retriever *caused* an answer change unless an
experiment separately supplies that causal identification.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

_MAX_ITEMS = 100_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum or "\x00" in cleaned:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class RetrievalAuditItem:
    evidence_id: str
    source_id: str
    rank: int
    score: float
    route: str
    cited: bool = False
    support_state: str = "unknown"
    metadata_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id", 500))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1 or self.rank > _MAX_ITEMS:
            raise ValueError("rank is invalid")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "route", _text(self.route, "route", 128))
        if not isinstance(self.cited, bool):
            raise ValueError("cited must be boolean")
        state = _text(self.support_state, "support_state", 64).lower()
        if state not in {"supported", "partially_supported", "unsupported", "contradicted", "unknown"}:
            raise ValueError("support_state is invalid")
        object.__setattr__(self, "support_state", state)
        digest = self.metadata_fingerprint.strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("metadata_fingerprint must be empty or SHA-256")
        object.__setattr__(self, "metadata_fingerprint", digest)


@dataclass(frozen=True)
class RetrievalAuditSnapshot:
    snapshot_id: str
    items: tuple[RetrievalAuditItem, ...]
    retriever_id: str
    reranker_id: str = ""
    fusion_id: str = ""
    policy_fingerprint: str = ""
    query_sha256: str = ""
    result_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id", 500))
        if len(self.items) > _MAX_ITEMS or any(not isinstance(item, RetrievalAuditItem) for item in self.items):
            raise ValueError("items are invalid")
        evidence_ids = [item.evidence_id for item in self.items]
        ranks = [item.rank for item in self.items]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("snapshot contains duplicate evidence_id values")
        if len(set(ranks)) != len(ranks):
            raise ValueError("snapshot contains duplicate ranks")
        if ranks and sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("snapshot ranks must be contiguous starting at one")
        ordered = tuple(sorted(self.items, key=lambda item: item.rank))
        object.__setattr__(self, "items", ordered)
        object.__setattr__(self, "retriever_id", _text(self.retriever_id, "retriever_id", 500))
        object.__setattr__(self, "reranker_id", _text(self.reranker_id, "reranker_id", 500, allow_empty=True))
        object.__setattr__(self, "fusion_id", _text(self.fusion_id, "fusion_id", 500, allow_empty=True))
        for field in ("policy_fingerprint", "query_sha256"):
            digest = str(getattr(self, field)).strip().lower()
            if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
                raise ValueError(f"{field} must be empty or SHA-256")
            object.__setattr__(self, field, digest)
        object.__setattr__(self, "result_id", _text(self.result_id, "result_id", 500, allow_empty=True))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class RankChange:
    evidence_id: str
    source_id: str
    baseline_rank: int
    candidate_rank: int
    rank_delta: int
    baseline_score: float
    candidate_score: float
    score_delta: float
    route_changed: bool
    citation_changed: bool
    support_changed: bool


@dataclass(frozen=True)
class CounterfactualRetrievalAudit:
    baseline_fingerprint: str
    candidate_fingerprint: str
    same_query: bool
    added_evidence_ids: tuple[str, ...]
    removed_evidence_ids: tuple[str, ...]
    added_source_ids: tuple[str, ...]
    removed_source_ids: tuple[str, ...]
    rank_changes: tuple[RankChange, ...]
    top_k_overlap: Mapping[int, float]
    evidence_jaccard: float
    citation_added: tuple[str, ...]
    citation_removed: tuple[str, ...]
    support_state_changes: tuple[str, ...]
    route_changes: tuple[str, ...]
    component_changes: Mapping[str, tuple[str, str]]
    result_changed: bool | None
    interpretation: tuple[str, ...]
    fingerprint: str


def _top_ids(snapshot: RetrievalAuditSnapshot, k: int) -> set[str]:
    return {item.evidence_id for item in snapshot.items[: min(k, len(snapshot.items))]}


def audit_retrieval_change(
    baseline: RetrievalAuditSnapshot,
    candidate: RetrievalAuditSnapshot,
    *,
    top_k_values: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> CounterfactualRetrievalAudit:
    if not isinstance(baseline, RetrievalAuditSnapshot) or not isinstance(candidate, RetrievalAuditSnapshot):
        raise TypeError("baseline and candidate must be RetrievalAuditSnapshot values")
    selected_k: list[int] = []
    for value in top_k_values:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_ITEMS:
            raise ValueError("top_k_values contain an invalid value")
        if value not in selected_k:
            selected_k.append(value)
    if len(selected_k) > 100:
        raise ValueError("too many top_k_values")

    before = {item.evidence_id: item for item in baseline.items}
    after = {item.evidence_id: item for item in candidate.items}
    before_ids = set(before)
    after_ids = set(after)
    added = tuple(sorted(after_ids - before_ids))
    removed = tuple(sorted(before_ids - after_ids))
    shared = sorted(before_ids & after_ids)

    rank_changes: list[RankChange] = []
    route_changes: list[str] = []
    support_changes: list[str] = []
    for evidence_id in shared:
        left = before[evidence_id]
        right = after[evidence_id]
        route_changed = left.route != right.route
        citation_changed = left.cited != right.cited
        support_changed = left.support_state != right.support_state
        if route_changed:
            route_changes.append(evidence_id)
        if support_changed:
            support_changes.append(evidence_id)
        if (
            left.rank != right.rank
            or left.score != right.score
            or route_changed
            or citation_changed
            or support_changed
        ):
            rank_changes.append(
                RankChange(
                    evidence_id=evidence_id,
                    source_id=right.source_id,
                    baseline_rank=left.rank,
                    candidate_rank=right.rank,
                    rank_delta=right.rank - left.rank,
                    baseline_score=left.score,
                    candidate_score=right.score,
                    score_delta=right.score - left.score,
                    route_changed=route_changed,
                    citation_changed=citation_changed,
                    support_changed=support_changed,
                )
            )
    rank_changes.sort(key=lambda item: (abs(item.rank_delta), abs(item.score_delta), item.evidence_id), reverse=True)

    before_sources = {item.source_id for item in baseline.items}
    after_sources = {item.source_id for item in candidate.items}
    union = before_ids | after_ids
    evidence_jaccard = 1.0 if not union else len(before_ids & after_ids) / len(union)
    overlaps: dict[int, float] = {}
    for k in selected_k:
        left_ids = _top_ids(baseline, k)
        right_ids = _top_ids(candidate, k)
        denominator = max(len(left_ids), len(right_ids), 1)
        overlaps[k] = len(left_ids & right_ids) / denominator

    before_cited = {item.evidence_id for item in baseline.items if item.cited}
    after_cited = {item.evidence_id for item in candidate.items if item.cited}
    components: dict[str, tuple[str, str]] = {}
    for name in ("retriever_id", "reranker_id", "fusion_id", "policy_fingerprint"):
        left = str(getattr(baseline, name))
        right = str(getattr(candidate, name))
        if left != right:
            components[name] = (left, right)
    same_query = bool(baseline.query_sha256 and baseline.query_sha256 == candidate.query_sha256)
    if baseline.query_sha256 and candidate.query_sha256 and not same_query:
        interpretation = ["query_identity_changed; retrieval differences are not a controlled same-query comparison"]
    else:
        interpretation = ["observed retrieval differences are descriptive and do not by themselves establish causal attribution"]
    if components:
        interpretation.append("one or more retrieval components changed between snapshots")
    if added or removed:
        interpretation.append("the retrieved evidence set changed")
    if before_cited != after_cited:
        interpretation.append("the citation-bearing evidence set changed")
    if support_changes:
        interpretation.append("support labels changed for shared evidence and should be audited independently")

    result_changed: bool | None
    if baseline.result_id and candidate.result_id:
        result_changed = baseline.result_id != candidate.result_id
    else:
        result_changed = None

    payload = {
        "baseline_fingerprint": baseline.fingerprint,
        "candidate_fingerprint": candidate.fingerprint,
        "same_query": same_query,
        "added": added,
        "removed": removed,
        "rank_changes": [asdict(item) for item in rank_changes],
        "top_k_overlap": overlaps,
        "citation_added": sorted(after_cited - before_cited),
        "citation_removed": sorted(before_cited - after_cited),
        "support_changes": support_changes,
        "route_changes": route_changes,
        "components": components,
        "result_changed": result_changed,
    }
    fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
    return CounterfactualRetrievalAudit(
        baseline_fingerprint=baseline.fingerprint,
        candidate_fingerprint=candidate.fingerprint,
        same_query=same_query,
        added_evidence_ids=added,
        removed_evidence_ids=removed,
        added_source_ids=tuple(sorted(after_sources - before_sources)),
        removed_source_ids=tuple(sorted(before_sources - after_sources)),
        rank_changes=tuple(rank_changes),
        top_k_overlap=overlaps,
        evidence_jaccard=evidence_jaccard,
        citation_added=tuple(sorted(after_cited - before_cited)),
        citation_removed=tuple(sorted(before_cited - after_cited)),
        support_state_changes=tuple(support_changes),
        route_changes=tuple(route_changes),
        component_changes=components,
        result_changed=result_changed,
        interpretation=tuple(interpretation),
        fingerprint=fingerprint,
    )


__all__ = [
    "CounterfactualRetrievalAudit",
    "RankChange",
    "RetrievalAuditItem",
    "RetrievalAuditSnapshot",
    "audit_retrieval_change",
]
