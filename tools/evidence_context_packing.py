"""Deterministic, provenance-preserving context packing under token budgets.

Retrieval/reranking produces candidates; this module decides which immutable evidence
references are allowed into the generator context.  Selection operates on digests,
token counts and normalized utilities rather than raw text. Pairwise redundancy values
are caller-supplied and content-addressed, enabling MMR-style selection without making
this control-plane layer inspect document content.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 1000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0,1]")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0,1]")
    return selected


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


@dataclass(frozen=True)
class ContextEvidenceCandidate:
    evidence_id: str
    evidence_sha256: str
    document_id: str
    source_id: str
    generation_id: str
    token_count: int
    relevance: float
    support: float = 0.0
    contradiction: float = 0.0
    authority: float = 1.0
    mandatory: bool = False

    def __post_init__(self) -> None:
        for name in ("evidence_id", "document_id", "source_id", "generation_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 1:
            raise ValueError("token_count must be a positive integer")
        for name in ("relevance", "support", "contradiction", "authority"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        if not isinstance(self.mandatory, bool):
            raise ValueError("mandatory must be boolean")

    @property
    def candidate_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-context-evidence-candidate/v1", **asdict(self)})


@dataclass(frozen=True)
class EvidenceSimilarity:
    left_evidence_sha256: str
    right_evidence_sha256: str
    similarity: float

    def __post_init__(self) -> None:
        left = _sha(self.left_evidence_sha256, "left_evidence_sha256")
        right = _sha(self.right_evidence_sha256, "right_evidence_sha256")
        if left == right:
            raise ValueError("similarity pair must contain distinct evidence")
        if right < left:
            left, right = right, left
        object.__setattr__(self, "left_evidence_sha256", left)
        object.__setattr__(self, "right_evidence_sha256", right)
        object.__setattr__(self, "similarity", _probability(self.similarity, "similarity"))

    @property
    def pair_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-evidence-similarity/v1", **asdict(self)})


@dataclass(frozen=True)
class ContextPackingPolicy:
    max_context_tokens: int
    max_items: int = 32
    max_per_document: int = 5
    max_per_source: int = 10
    relevance_weight: float = 1.0
    support_weight: float = 0.5
    contradiction_weight: float = 0.5
    authority_weight: float = 0.5
    redundancy_penalty: float = 0.35
    token_cost_penalty: float = 0.0
    min_utility: float = 0.0
    min_counterevidence_items: int = 0
    counterevidence_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("max_context_tokens", "max_items", "max_per_document", "max_per_source"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_counterevidence_items",):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("relevance_weight", "support_weight", "contradiction_weight", "authority_weight", "redundancy_penalty", "token_cost_penalty", "min_utility"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        object.__setattr__(self, "counterevidence_threshold", _probability(self.counterevidence_threshold, "counterevidence_threshold"))

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-context-packing-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class PackedEvidence:
    order: int
    evidence_id: str
    evidence_sha256: str
    document_id: str
    source_id: str
    token_count: int
    utility: float
    max_redundancy: float
    selection_reason: str

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 1:
            raise ValueError("order must be positive")
        for name in ("evidence_id", "document_id", "source_id", "selection_reason"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count < 1:
            raise ValueError("token_count must be positive")
        object.__setattr__(self, "utility", float(self.utility))
        object.__setattr__(self, "max_redundancy", _probability(self.max_redundancy, "max_redundancy"))


@dataclass(frozen=True)
class ContextPackingReceipt:
    candidate_pool_sha256: str
    similarity_set_sha256: str
    policy_sha256: str
    selected: tuple[PackedEvidence, ...]
    total_tokens: int
    mandatory_count: int
    counterevidence_count: int
    dropped_counts: tuple[tuple[str, int], ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("candidate_pool_sha256", "similarity_set_sha256", "policy_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        selected = tuple(self.selected)
        if [row.order for row in selected] != list(range(1, len(selected) + 1)):
            raise ValueError("packed evidence order must be contiguous")
        if len({row.evidence_sha256 for row in selected}) != len(selected):
            raise ValueError("packed context contains duplicate evidence")
        object.__setattr__(self, "selected", selected)
        for name in ("total_tokens", "mandatory_count", "counterevidence_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.total_tokens != sum(row.token_count for row in selected):
            raise ValueError("total_tokens does not match selected evidence")
        dropped = tuple(sorted((str(reason), int(count)) for reason, count in self.dropped_counts))
        if any(not reason or count < 0 for reason, count in dropped):
            raise ValueError("dropped_counts is invalid")
        object.__setattr__(self, "dropped_counts", dropped)
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("receipt_sha256 does not match context packing receipt")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-context-packing-receipt/v1",
            "candidate_pool_sha256": self.candidate_pool_sha256,
            "similarity_set_sha256": self.similarity_set_sha256,
            "policy_sha256": self.policy_sha256,
            "selected": [asdict(row) for row in self.selected],
            "total_tokens": self.total_tokens,
            "mandatory_count": self.mandatory_count,
            "counterevidence_count": self.counterevidence_count,
            "dropped_counts": self.dropped_counts,
        }


def _candidate_pool_digest(candidates: Sequence[ContextEvidenceCandidate]) -> str:
    return _digest({"schema": "rigorousrag-context-candidate-pool/v1", "candidates": sorted(row.candidate_sha256 for row in candidates)})


def _similarity_set_digest(rows: Sequence[EvidenceSimilarity]) -> str:
    return _digest({"schema": "rigorousrag-evidence-similarity-set/v1", "pairs": sorted(row.pair_sha256 for row in rows)})


def _base_utility(candidate: ContextEvidenceCandidate, policy: ContextPackingPolicy) -> float:
    return (
        policy.relevance_weight * candidate.relevance
        + policy.support_weight * candidate.support
        + policy.contradiction_weight * candidate.contradiction
        + policy.authority_weight * candidate.authority
        - policy.token_cost_penalty * candidate.token_count / policy.max_context_tokens
    )


def pack_evidence_context(
    candidates: Iterable[ContextEvidenceCandidate],
    *,
    policy: ContextPackingPolicy,
    similarities: Iterable[EvidenceSimilarity] = (),
) -> ContextPackingReceipt:
    values = tuple(candidates)
    if not values or len(values) > 1_000_000:
        raise ValueError("context candidate pool must be non-empty and bounded")
    if any(not isinstance(row, ContextEvidenceCandidate) for row in values):
        raise ValueError("candidates contains invalid values")
    if len({row.evidence_sha256 for row in values}) != len(values) or len({row.evidence_id for row in values}) != len(values):
        raise ValueError("context candidates must be unique by evidence id and digest")
    if not isinstance(policy, ContextPackingPolicy):
        raise ValueError("policy must be ContextPackingPolicy")
    sim_rows = tuple(similarities)
    if any(not isinstance(row, EvidenceSimilarity) for row in sim_rows):
        raise ValueError("similarities contains invalid values")
    known = {row.evidence_sha256 for row in values}
    similarity_map: dict[tuple[str, str], float] = {}
    for row in sim_rows:
        if row.left_evidence_sha256 not in known or row.right_evidence_sha256 not in known:
            raise ValueError("similarity references evidence outside the candidate pool")
        key = (row.left_evidence_sha256, row.right_evidence_sha256)
        if key in similarity_map and abs(similarity_map[key] - row.similarity) > 1e-12:
            raise ValueError("conflicting duplicate evidence similarity")
        similarity_map[key] = row.similarity

    mandatory = sorted((row for row in values if row.mandatory), key=lambda row: (row.evidence_sha256, row.evidence_id))
    if len(mandatory) > policy.max_items or sum(row.token_count for row in mandatory) > policy.max_context_tokens:
        raise ValueError("mandatory evidence exceeds context packing budget")
    doc_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    selected: list[ContextEvidenceCandidate] = []
    packed: list[PackedEvidence] = []
    total_tokens = 0
    dropped: dict[str, int] = {}

    def redundancy(candidate: ContextEvidenceCandidate) -> float:
        if not selected:
            return 0.0
        maximum = 0.0
        for prior in selected:
            left, right = sorted((candidate.evidence_sha256, prior.evidence_sha256))
            maximum = max(maximum, similarity_map.get((left, right), 0.0))
        return maximum

    def eligible(candidate: ContextEvidenceCandidate) -> str | None:
        if len(selected) >= policy.max_items:
            return "item_cap"
        if total_tokens + candidate.token_count > policy.max_context_tokens:
            return "token_budget"
        if doc_counts.get(candidate.document_id, 0) >= policy.max_per_document:
            return "document_cap"
        if source_counts.get(candidate.source_id, 0) >= policy.max_per_source:
            return "source_cap"
        return None

    def add(candidate: ContextEvidenceCandidate, reason: str, *, ignore_caps: bool = False) -> None:
        nonlocal total_tokens
        if not ignore_caps:
            why = eligible(candidate)
            if why is not None:
                raise RuntimeError(f"candidate unexpectedly failed selection eligibility: {why}")
        red = redundancy(candidate)
        utility = _base_utility(candidate, policy) - policy.redundancy_penalty * red
        selected.append(candidate)
        packed.append(PackedEvidence(len(packed) + 1, candidate.evidence_id, candidate.evidence_sha256, candidate.document_id, candidate.source_id, candidate.token_count, utility, red, reason))
        total_tokens += candidate.token_count
        doc_counts[candidate.document_id] = doc_counts.get(candidate.document_id, 0) + 1
        source_counts[candidate.source_id] = source_counts.get(candidate.source_id, 0) + 1

    # Mandatory evidence is authoritative policy input: if it violates per-doc/source caps
    # we fail rather than silently dropping it. Token/item budgets were checked above.
    for row in mandatory:
        if doc_counts.get(row.document_id, 0) >= policy.max_per_document or source_counts.get(row.source_id, 0) >= policy.max_per_source:
            raise ValueError("mandatory evidence violates document/source caps")
        add(row, "mandatory", ignore_caps=True)

    remaining = [row for row in values if not row.mandatory]

    # Preserve explicit counterevidence before the general MMR pass so high-support
    # evidence cannot crowd all contradictory evidence out of the final prompt.
    counter_pool = [row for row in remaining if row.contradiction >= policy.counterevidence_threshold]
    counter_pool.sort(key=lambda row: (-row.contradiction, -_base_utility(row, policy), row.token_count, row.evidence_sha256))
    current_counter = sum(row.contradiction >= policy.counterevidence_threshold for row in selected)
    for row in counter_pool:
        if current_counter >= policy.min_counterevidence_items:
            break
        reason = eligible(row)
        if reason is not None:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        add(row, "counterevidence_quota")
        current_counter += 1
        remaining.remove(row)

    while remaining and len(selected) < policy.max_items:
        ranked = []
        for row in remaining:
            reason = eligible(row)
            if reason is not None:
                ranked.append((None, row, reason))
                continue
            red = redundancy(row)
            utility = _base_utility(row, policy) - policy.redundancy_penalty * red
            ranked.append((utility, row, None))
        feasible = [(utility, row) for utility, row, reason in ranked if reason is None and utility is not None and utility >= policy.min_utility]
        if not feasible:
            for utility, row, reason in ranked:
                dropped[reason or "below_min_utility"] = dropped.get(reason or "below_min_utility", 0) + 1
            break
        feasible.sort(key=lambda pair: (-pair[0], pair[1].token_count, pair[1].source_id, pair[1].document_id, pair[1].evidence_sha256))
        _, chosen = feasible[0]
        add(chosen, "mmr_utility")
        remaining.remove(chosen)

    # Account for candidates never selected because the item cap was reached.
    for row in remaining:
        if row in selected:
            continue
        reason = eligible(row) or "not_selected"
        dropped[reason] = dropped.get(reason, 0) + 1

    payload = {
        "schema": "rigorousrag-context-packing-receipt/v1",
        "candidate_pool_sha256": _candidate_pool_digest(values),
        "similarity_set_sha256": _similarity_set_digest(sim_rows),
        "policy_sha256": policy.policy_sha256,
        "selected": [asdict(row) for row in packed],
        "total_tokens": total_tokens,
        "mandatory_count": sum(row.selection_reason == "mandatory" for row in packed),
        "counterevidence_count": sum(values_by_sha.contradiction >= policy.counterevidence_threshold for values_by_sha in (next(candidate for candidate in values if candidate.evidence_sha256 == row.evidence_sha256) for row in packed)),
        "dropped_counts": tuple(sorted(dropped.items())),
    }
    return ContextPackingReceipt(**payload, receipt_sha256=_digest(payload))


__all__ = [
    "ContextEvidenceCandidate",
    "ContextPackingPolicy",
    "ContextPackingReceipt",
    "EvidenceSimilarity",
    "PackedEvidence",
    "pack_evidence_context",
]
