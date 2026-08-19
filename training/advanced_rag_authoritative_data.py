"""Richer backward-compatible data records for authoritative advanced RAG training.

The original record classes remain stable research primitives. This module subclasses them
so configuration-driven runs can preserve two pieces of supervision that otherwise collapse:

* supporting and contradicting evidence identities may coexist for one claim; and
* a logged dynamic-RAG state may declare the exact closed set of actions that were legal.

Subclasses remain ``isinstance`` compatible with the existing collators and models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_data import (
    AdvancedDatasetBinding,
    DynamicRagEpisodeStep,
    GroundedClaimAnnotation,
    GroundedEvidenceRecord,
    GroundedGenerationExample,
    TextSpan,
    sha256_file,
)
from training.dynamic_retrieval_policy import DEFAULT_FEATURE_NAMES, DynamicRetrievalAction
from training.grounded_generation import ReflectionAction

_MAX_EVIDENCE = 4096
_MAX_RECORDS = 100_000_000
_MAX_BYTES_PER_LINE = 64 * 1024 * 1024


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _span(value: Any) -> TextSpan:
    if not isinstance(value, Mapping) or set(value) != {"start", "end"}:
        raise ValueError("span must be a closed {start,end} object")
    return TextSpan(start=value["start"], end=value["end"])


def _ids(raw: Any, label: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    values = tuple(_identifier(value, label) for value in raw)
    if len(values) > _MAX_EVIDENCE or len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique and bounded")
    return values


@dataclass(frozen=True)
class StancedGroundedClaimAnnotation(GroundedClaimAnnotation):
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.span, TextSpan):
            raise ValueError("claim span must be TextSpan")
        supporting = tuple(_identifier(value, "supporting evidence id") for value in self.supporting_evidence_ids)
        contradicting = tuple(_identifier(value, "contradicting evidence id") for value in self.contradicting_evidence_ids)
        if len(supporting) > _MAX_EVIDENCE or len(set(supporting)) != len(supporting):
            raise ValueError("supporting evidence ids must be unique and bounded")
        if len(contradicting) > _MAX_EVIDENCE or len(set(contradicting)) != len(contradicting):
            raise ValueError("contradicting evidence ids must be unique and bounded")
        overlap = set(supporting) & set(contradicting)
        if overlap:
            raise ValueError(f"one evidence item may not be both supporting and contradicting: {sorted(overlap)[:20]}")
        legacy = tuple(_identifier(value, "claim evidence id") for value in self.evidence_ids)
        if supporting or contradicting:
            union = tuple(dict.fromkeys((*supporting, *contradicting)))
            if legacy and set(legacy) != set(union):
                raise ValueError("legacy evidence_ids must equal the union of stanced evidence ids")
            legacy = union
        if len(legacy) > _MAX_EVIDENCE or len(set(legacy)) != len(legacy):
            raise ValueError("claim evidence ids must be unique and bounded")
        object.__setattr__(self, "supporting_evidence_ids", supporting)
        object.__setattr__(self, "contradicting_evidence_ids", contradicting)
        object.__setattr__(self, "evidence_ids", legacy)
        object.__setattr__(self, "supported", bool(self.supported or supporting))
        object.__setattr__(self, "contradicted", bool(self.contradicted or contradicting))


@dataclass(frozen=True)
class LegalDynamicRagEpisodeStep(DynamicRagEpisodeStep):
    valid_actions: tuple[DynamicRetrievalAction, ...] = tuple(DynamicRetrievalAction)

    def __post_init__(self) -> None:
        super().__post_init__()
        actions = tuple(action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action) for action in self.valid_actions)
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("valid_actions must be a non-empty unique action sequence")
        if self.action not in actions:
            raise ValueError("logged action must be present in valid_actions")
        object.__setattr__(self, "valid_actions", actions)


def parse_authoritative_grounded_example(value: Any) -> GroundedGenerationExample:
    if not isinstance(value, Mapping):
        raise ValueError("grounded training record must be an object")
    allowed = {
        "example_id", "prompt", "answer", "evidence", "claims", "abstain", "reflection_action",
        "unsupported_spans", "chosen_answer", "rejected_answer", "reference_chosen_log_prob",
        "reference_rejected_log_prob", "teacher_cache_key", "retriever_cache_key", "metadata",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"grounded training record contains unsupported fields: {sorted(unknown)}")
    evidence_raw = value.get("evidence")
    claims_raw = value.get("claims") or []
    unsupported_raw = value.get("unsupported_spans") or []
    if not isinstance(evidence_raw, list) or not isinstance(claims_raw, list) or not isinstance(unsupported_raw, list):
        raise ValueError("evidence/claims/unsupported_spans must be arrays")
    evidence = []
    for item in evidence_raw:
        if not isinstance(item, Mapping) or set(item) - {"evidence_id", "text", "source_id"}:
            raise ValueError("evidence entry has unsupported fields")
        evidence.append(GroundedEvidenceRecord(item.get("evidence_id"), item.get("text"), item.get("source_id")))
    claims = []
    for item in claims_raw:
        if not isinstance(item, Mapping):
            raise ValueError("claim annotation must be an object")
        allowed_claim = {"span", "evidence_ids", "supporting_evidence_ids", "contradicting_evidence_ids", "supported", "contradicted"}
        unknown_claim = set(item) - allowed_claim
        if unknown_claim:
            raise ValueError(f"claim annotation contains unsupported fields: {sorted(unknown_claim)}")
        legacy = _ids(item.get("evidence_ids"), "evidence_ids")
        supporting = _ids(item.get("supporting_evidence_ids"), "supporting_evidence_ids")
        contradicting = _ids(item.get("contradicting_evidence_ids"), "contradicting_evidence_ids")
        supported = bool(item.get("supported", False))
        contradicted = bool(item.get("contradicted", False))
        # Backward compatibility: legacy evidence_ids inherit the declared binary stance.
        if legacy and not supporting and not contradicting:
            if supported and contradicted:
                raise ValueError("legacy evidence_ids cannot be split across both stances; provide explicit stanced ids")
            if contradicted:
                contradicting = legacy
            elif supported:
                supporting = legacy
        claims.append(
            StancedGroundedClaimAnnotation(
                span=_span(item.get("span")),
                evidence_ids=legacy,
                supported=supported,
                contradicted=contradicted,
                supporting_evidence_ids=supporting,
                contradicting_evidence_ids=contradicting,
            )
        )
    return GroundedGenerationExample(
        example_id=value.get("example_id"), prompt=value.get("prompt"), answer=value.get("answer", ""),
        evidence=tuple(evidence), claims=tuple(claims), abstain=bool(value.get("abstain", False)),
        reflection_action=value.get("reflection_action", ReflectionAction.STOP.value),
        unsupported_spans=tuple(_span(item) for item in unsupported_raw), chosen_answer=value.get("chosen_answer"),
        rejected_answer=value.get("rejected_answer"), reference_chosen_log_prob=value.get("reference_chosen_log_prob"),
        reference_rejected_log_prob=value.get("reference_rejected_log_prob"), teacher_cache_key=value.get("teacher_cache_key"),
        retriever_cache_key=value.get("retriever_cache_key"), metadata=value.get("metadata") or {},
    )


def parse_authoritative_dynamic_step(value: Any) -> LegalDynamicRagEpisodeStep:
    if not isinstance(value, Mapping):
        raise ValueError("dynamic episode record must be an object")
    allowed = {
        "episode_id", "step_id", "context", "features", "action", "realized_retrieval_gain",
        "behavior_action_probability", "advantage", "need_spans", "hidden_state_cache_key",
        "terminal_utility", "metadata", "valid_actions",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"dynamic episode record contains unsupported fields: {sorted(unknown)}")
    need = value.get("need_spans") or []
    if not isinstance(need, list):
        raise ValueError("need_spans must be an array")
    raw_valid = value.get("valid_actions")
    valid = tuple(DynamicRetrievalAction(action) for action in raw_valid) if raw_valid is not None else tuple(DynamicRetrievalAction)
    return LegalDynamicRagEpisodeStep(
        episode_id=value.get("episode_id"), step_id=value.get("step_id"), context=value.get("context"),
        features=value.get("features") or {}, action=value.get("action"), realized_retrieval_gain=value.get("realized_retrieval_gain", 0.0),
        behavior_action_probability=value.get("behavior_action_probability"), advantage=value.get("advantage"),
        need_spans=tuple(_span(item) for item in need), hidden_state_cache_key=value.get("hidden_state_cache_key"),
        terminal_utility=value.get("terminal_utility"), metadata=value.get("metadata") or {}, valid_actions=valid,
    )


class ManifestBoundAuthoritativeJsonlDataset:
    """Strict manifest-bound parser for authoritative grounded/dynamic records."""
    def __init__(self, path: str | Path, *, expected_sha256: str, dataset_manifest_sha256: str, split_name: str, record_kind: str, expected_record_count: int | None = None) -> None:
        selected = safe_advanced_path(path, label="advanced training dataset", must_exist=True, require_file=True)
        actual = sha256_file(selected)
        expected = str(expected_sha256).strip().lower()
        if actual != expected:
            raise ValueError("local advanced-training data digest does not match expected artifact")
        if record_kind not in {"grounded_generation", "dynamic_rag_episode"}:
            raise ValueError("record_kind must be grounded_generation or dynamic_rag_episode")
        parser = parse_authoritative_grounded_example if record_kind == "grounded_generation" else parse_authoritative_dynamic_step
        records = []
        with selected.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(line.encode("utf-8")) > _MAX_BYTES_PER_LINE:
                    raise ValueError(f"advanced training JSON line {line_number} exceeds byte safety bound")
                if len(records) >= _MAX_RECORDS:
                    raise ValueError("advanced training dataset exceeds record safety bound")
                try:
                    payload = json.loads(line, parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
                    records.append(parser(payload))
                except Exception as exc:
                    raise ValueError(f"invalid {record_kind} JSON at line {line_number}") from exc
        if expected_record_count is not None and len(records) != expected_record_count:
            raise ValueError("advanced training record count differs from manifest")
        ids = [record.example_id if record_kind == "grounded_generation" else f"{record.episode_id}:{record.step_id}" for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError("advanced training record identities must be unique")
        self._records = tuple(records)
        self.binding = AdvancedDatasetBinding(str(selected), actual, dataset_manifest_sha256, split_name, len(records), record_kind)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Any:
        return self._records[index]


__all__ = [
    "LegalDynamicRagEpisodeStep", "ManifestBoundAuthoritativeJsonlDataset",
    "StancedGroundedClaimAnnotation", "parse_authoritative_dynamic_step",
    "parse_authoritative_grounded_example",
]
