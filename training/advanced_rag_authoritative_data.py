"""Richer backward-compatible data records for authoritative advanced RAG training.

The authoritative JSONL dataset validates exact source bytes and every record once, then keeps
only a compact random-access index. Records are reparsed lazily and their exact line SHA-256 is
rechecked on every access, preserving content authority without retaining the full corpus as
Python objects in memory.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import threading
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_data import AdvancedDatasetBinding, DynamicRagEpisodeStep, GroundedClaimAnnotation, GroundedEvidenceRecord, GroundedGenerationExample, TextSpan
from training.dynamic_retrieval_policy import DynamicRetrievalAction
from training.grounded_generation import ReflectionAction

_MAX_EVIDENCE = 4096
_MAX_RECORDS = 100_000_000
_MAX_BYTES_PER_LINE = 64 * 1024 * 1024
_MAX_METADATA = 2000
_HEX = frozenset("0123456789abcdef")


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        selected = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


import math


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


def _metadata(raw: Any) -> Mapping[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping) or len(raw) > _MAX_METADATA:
        raise ValueError("metadata must be a bounded object")
    return {_identifier(str(key), "metadata key", 300): _identifier(str(value), "metadata value", 10000) for key, value in raw.items()}


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
    value_target: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        actions = tuple(action if isinstance(action, DynamicRetrievalAction) else DynamicRetrievalAction(action) for action in self.valid_actions)
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("valid_actions must be a non-empty unique action sequence")
        if self.action not in actions:
            raise ValueError("logged action must be present in valid_actions")
        object.__setattr__(self, "valid_actions", actions)
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.value_target is not None:
            object.__setattr__(self, "value_target", _finite(self.value_target, "value_target"))


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
        if legacy and not supporting and not contradicting:
            if supported and contradicted:
                raise ValueError("legacy evidence_ids cannot be split across both stances; provide explicit stanced ids")
            if contradicted:
                contradicting = legacy
            elif supported:
                supporting = legacy
        claims.append(StancedGroundedClaimAnnotation(
            span=_span(item.get("span")), evidence_ids=legacy, supported=supported, contradicted=contradicted,
            supporting_evidence_ids=supporting, contradicting_evidence_ids=contradicting,
        ))
    return GroundedGenerationExample(
        example_id=value.get("example_id"), prompt=value.get("prompt"), answer=value.get("answer", ""),
        evidence=tuple(evidence), claims=tuple(claims), abstain=bool(value.get("abstain", False)),
        reflection_action=value.get("reflection_action", ReflectionAction.STOP.value),
        unsupported_spans=tuple(_span(item) for item in unsupported_raw), chosen_answer=value.get("chosen_answer"),
        rejected_answer=value.get("rejected_answer"), reference_chosen_log_prob=value.get("reference_chosen_log_prob"),
        reference_rejected_log_prob=value.get("reference_rejected_log_prob"), teacher_cache_key=value.get("teacher_cache_key"),
        retriever_cache_key=value.get("retriever_cache_key"), metadata=_metadata(value.get("metadata")),
    )


def parse_authoritative_dynamic_step(value: Any) -> LegalDynamicRagEpisodeStep:
    if not isinstance(value, Mapping):
        raise ValueError("dynamic episode record must be an object")
    allowed = {
        "episode_id", "step_id", "context", "features", "action", "realized_retrieval_gain",
        "behavior_action_probability", "advantage", "need_spans", "hidden_state_cache_key",
        "terminal_utility", "metadata", "valid_actions", "value_target",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"dynamic episode record contains unsupported fields: {sorted(unknown)}")
    need = value.get("need_spans") or []
    if not isinstance(need, list):
        raise ValueError("need_spans must be an array")
    raw_valid = value.get("valid_actions")
    if raw_valid is not None and not isinstance(raw_valid, list):
        raise ValueError("valid_actions must be an array when supplied")
    metadata = _metadata(value.get("metadata"))
    valid = tuple(DynamicRetrievalAction(action) for action in raw_valid) if raw_valid is not None else tuple(DynamicRetrievalAction)
    return LegalDynamicRagEpisodeStep(
        episode_id=value.get("episode_id"), step_id=value.get("step_id"), context=value.get("context"),
        features=value.get("features") or {}, action=value.get("action"), realized_retrieval_gain=value.get("realized_retrieval_gain", 0.0),
        behavior_action_probability=value.get("behavior_action_probability"), advantage=value.get("advantage"),
        need_spans=tuple(_span(item) for item in need), hidden_state_cache_key=value.get("hidden_state_cache_key"),
        terminal_utility=value.get("terminal_utility"), metadata=metadata, valid_actions=valid,
        value_target=value.get("value_target"),
    )


def _strict_json_line(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _record_identity(record: Any, record_kind: str) -> str:
    if record_kind == "grounded_generation":
        return "grounded:" + record.example_id
    # JSON tuple encoding is injective even when identifiers themselves contain ':' or other
    # punctuation; simple string concatenation is not.
    return "dynamic:" + json.dumps([record.episode_id, record.step_id], ensure_ascii=False, separators=(",", ":"))


class ManifestBoundAuthoritativeJsonlDataset:
    """Lazy random-access JSONL dataset bound to exact validated source bytes.

    Construction performs one full streaming pass that simultaneously verifies the configured
    source SHA-256, strict-parses every non-empty line, validates record semantics and detects
    duplicate logical record identities through a temporary disk-backed SQLite uniqueness
    index. Only byte offsets, byte lengths and 32-byte line digests remain resident afterward.

    ``__getitem__`` reads exactly the indexed byte range, rechecks its SHA-256 against the
    construction-time digest, and reparses it. Thus external file replacement cannot silently
    change a consumed record after validation, while deterministic random-access samplers remain
    available without keeping the full corpus as Python objects.
    """

    def __init__(self, path: str | Path, *, expected_sha256: str, dataset_manifest_sha256: str, split_name: str, record_kind: str, expected_record_count: int | None = None) -> None:
        selected = safe_advanced_path(path, label="advanced training dataset", must_exist=True, require_file=True)
        expected = _sha256(expected_sha256, "expected_sha256")
        manifest_sha = _sha256(dataset_manifest_sha256, "dataset_manifest_sha256")
        if record_kind not in {"grounded_generation", "dynamic_rag_episode"}:
            raise ValueError("record_kind must be grounded_generation or dynamic_rag_episode")
        parser = parse_authoritative_grounded_example if record_kind == "grounded_generation" else parse_authoritative_dynamic_step

        offsets = array("Q")
        lengths = array("Q")
        line_digests = bytearray()
        whole = hashlib.sha256()
        descriptor, uniqueness_name = tempfile.mkstemp(prefix="rigorousrag-authoritative-ids-", suffix=".sqlite3")
        os.close(descriptor)
        connection: sqlite3.Connection | None = None
        count = 0
        try:
            connection = sqlite3.connect(uniqueness_name)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("CREATE TABLE identities(value TEXT PRIMARY KEY) WITHOUT ROWID")
            connection.execute("BEGIN")
            with selected.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    line_number += 1
                    whole.update(raw)
                    if not raw.strip():
                        continue
                    if len(raw) > _MAX_BYTES_PER_LINE:
                        raise ValueError(f"advanced training JSON line {line_number} exceeds byte safety bound")
                    if count >= _MAX_RECORDS:
                        raise ValueError("advanced training dataset exceeds record safety bound")
                    payload = _strict_json_line(raw, label=f"advanced training JSON line {line_number}")
                    try:
                        record = parser(payload)
                    except Exception as exc:
                        raise ValueError(f"invalid {record_kind} JSON at line {line_number}") from exc
                    identity = _record_identity(record, record_kind)
                    try:
                        connection.execute("INSERT INTO identities(value) VALUES (?)", (identity,))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"advanced training record identity is duplicated at line {line_number}") from exc
                    offsets.append(offset)
                    lengths.append(len(raw))
                    line_digests.extend(hashlib.sha256(raw).digest())
                    count += 1
            connection.commit()
        finally:
            if connection is not None:
                connection.close()
            try:
                os.unlink(uniqueness_name)
            except FileNotFoundError:
                pass

        actual = whole.hexdigest()
        if actual != expected:
            raise ValueError("local advanced-training data digest does not match expected artifact")
        if expected_record_count is not None:
            if isinstance(expected_record_count, bool) or not isinstance(expected_record_count, int) or expected_record_count < 0:
                raise ValueError("expected_record_count must be non-negative or None")
            if count != expected_record_count:
                raise ValueError("advanced training record count differs from manifest")

        self._path = str(selected)
        self._offsets = offsets
        self._lengths = lengths
        self._line_digests = bytes(line_digests)
        self._parser = parser
        self._record_kind = record_kind
        self._local = threading.local()
        self.binding = AdvancedDatasetBinding(str(selected), actual, manifest_sha, split_name, count, record_kind)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state.pop("_local", None)
        # Module-level parser functions are recoverable from record_kind and need not be pickled.
        state.pop("_parser", None)
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._parser = parse_authoritative_grounded_example if self._record_kind == "grounded_generation" else parse_authoritative_dynamic_step
        self._local = threading.local()

    def _handle(self) -> Any:
        handle = getattr(self._local, "handle", None)
        if handle is None or handle.closed:
            handle = open(self._path, "rb")
            self._local.handle = handle
        return handle

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("advanced training dataset index must be an integer")
        length = len(self)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("advanced training dataset index out of range")
        offset = int(self._offsets[index])
        byte_length = int(self._lengths[index])
        handle = self._handle()
        handle.seek(offset)
        raw = handle.read(byte_length)
        if len(raw) != byte_length:
            raise ValueError("advanced training dataset changed after validation: indexed row is truncated")
        expected_digest = self._line_digests[index * 32 : (index + 1) * 32]
        if hashlib.sha256(raw).digest() != expected_digest:
            raise ValueError("advanced training dataset changed after validation: indexed row digest mismatch")
        payload = _strict_json_line(raw, label=f"advanced training record {index}")
        try:
            return self._parser(payload)
        except Exception as exc:
            raise ValueError(f"advanced training dataset row {index} no longer parses authoritatively") from exc


__all__ = ["LegalDynamicRagEpisodeStep", "ManifestBoundAuthoritativeJsonlDataset", "StancedGroundedClaimAnnotation", "parse_authoritative_dynamic_step", "parse_authoritative_grounded_example"]
