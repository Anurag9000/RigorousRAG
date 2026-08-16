"""Answer-version projections over immutable results and replacement lineage.

This module intentionally does not create a second answer-version database. Durable
``ResearchResultStore`` records are the immutable snapshots and ``ArtifactReplacementStore``
is the append-only old→new lineage authority. The projection exposes changes without
inventing claim-level support semantics that were not persisted at publication time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from tools.artifact_replacements import ArtifactReplacement, ArtifactReplacementStore
from tools.dependency_invalidation import DependencyRef
from tools.research_result_store import ResearchResultStore, StoredResearchResult

_RUNTIME_KEY = "_rigorousrag_runtime"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _runtime(result: StoredResearchResult) -> Mapping[str, Any]:
    raw = result.metadata.get(_RUNTIME_KEY)
    return raw if isinstance(raw, Mapping) else {}


def _policy_fingerprint(result: StoredResearchResult) -> str:
    runtime = _runtime(result)
    selected = runtime.get("selected_capabilities")
    if not isinstance(selected, Mapping):
        return ""
    policy = selected.get("policy")
    if not isinstance(policy, Mapping):
        return ""
    value = policy.get("fingerprint")
    return str(value) if isinstance(value, str) else ""


@dataclass(frozen=True)
class AnswerSnapshot:
    result_id: str
    query_sha256: str
    answer_sha256: str
    citation_ids: tuple[str, ...]
    model: str
    strategy: str
    runtime_config_sha256: str
    capability_registry_sha256: str
    policy_sha256: str
    created_at: float


@dataclass(frozen=True)
class AnswerTransition:
    old: AnswerSnapshot
    new: AnswerSnapshot
    reason: str
    triggering_event_sha256: str
    replacement_sha256: str
    replacement_created_at: float
    answer_changed: bool
    added_citation_ids: tuple[str, ...]
    removed_citation_ids: tuple[str, ...]
    model_changed: bool
    strategy_changed: bool
    runtime_config_changed: bool
    capability_registry_changed: bool
    policy_changed: bool


def snapshot(result: StoredResearchResult) -> AnswerSnapshot:
    if not isinstance(result, StoredResearchResult):
        raise TypeError("result must be StoredResearchResult")
    runtime = _runtime(result)
    return AnswerSnapshot(
        result_id=result.result_id,
        query_sha256=result.query_sha256,
        answer_sha256=_sha_text(result.answer),
        citation_ids=result.citation_ids,
        model=result.model,
        strategy=result.strategy,
        runtime_config_sha256=str(runtime.get("runtime_config_sha256") or ""),
        capability_registry_sha256=str(runtime.get("capability_registry_sha256") or ""),
        policy_sha256=_policy_fingerprint(result),
        created_at=result.created_at,
    )


def transition(
    old: StoredResearchResult,
    new: StoredResearchResult,
    replacement: ArtifactReplacement,
) -> AnswerTransition:
    if old.query_sha256 != new.query_sha256:
        raise RuntimeError("replacement result changed the original query identity")
    left = snapshot(old)
    right = snapshot(new)
    old_citations = set(left.citation_ids)
    new_citations = set(right.citation_ids)
    return AnswerTransition(
        old=left,
        new=right,
        reason=replacement.reason,
        triggering_event_sha256=replacement.triggering_event_sha256,
        replacement_sha256=replacement.replacement_sha256,
        replacement_created_at=replacement.created_at,
        answer_changed=left.answer_sha256 != right.answer_sha256,
        added_citation_ids=tuple(sorted(new_citations - old_citations)),
        removed_citation_ids=tuple(sorted(old_citations - new_citations)),
        model_changed=left.model != right.model,
        strategy_changed=left.strategy != right.strategy,
        runtime_config_changed=left.runtime_config_sha256 != right.runtime_config_sha256,
        capability_registry_changed=left.capability_registry_sha256 != right.capability_registry_sha256,
        policy_changed=left.policy_sha256 != right.policy_sha256,
    )


def answer_history(
    owner_id: str,
    result_id: str,
    *,
    results: ResearchResultStore,
    replacements: ArtifactReplacementStore,
    max_depth: int = 64,
) -> tuple[AnswerSnapshot, tuple[AnswerTransition, ...]]:
    if not isinstance(results, ResearchResultStore):
        raise TypeError("results must be ResearchResultStore")
    if not isinstance(replacements, ArtifactReplacementStore):
        raise TypeError("replacements must be ArtifactReplacementStore")
    start_ref = DependencyRef("result", result_id)
    first = results.get(owner_id, start_ref.resource_id)
    chain = replacements.chain(owner_id, start_ref, max_depth=max_depth)
    transitions: list[AnswerTransition] = []
    current = first
    for item in chain:
        if item.old.kind != "result" or item.new.kind != "result":
            raise RuntimeError("result replacement chain contains another artifact kind")
        if item.old.resource_id != current.result_id:
            raise RuntimeError("result replacement chain is discontinuous")
        next_result = results.get(owner_id, item.new.resource_id)
        transitions.append(transition(current, next_result, item))
        current = next_result
    return snapshot(first), tuple(transitions)


__all__ = [
    "AnswerSnapshot",
    "AnswerTransition",
    "answer_history",
    "snapshot",
    "transition",
]
