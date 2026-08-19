"""Governed local-only conversion into authoritative grounded-generator training records.

This importer is intentionally declarative and source-only. It converts already-local JSON or
JSONL benchmark/annotation bytes into the exact record schema consumed by
``ManifestBoundAuthoritativeJsonlDataset(record_kind='grounded_generation')``. It never
invents semantic supervision: abstention/reflection must come from configured source paths or
explicit operator-declared constants; claims/evidence are mapped only from configured fields;
ambiguous text-to-character-span alignment fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from evaluation.dataset_governance import DatasetCard, DatasetManifest, DatasetModality, DatasetTask, LicenseStatus, SplitManifest
from training.advanced_rag_authoritative_data import parse_authoritative_grounded_example
from training.grounded_generation import ReflectionAction

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_JSON_BYTES = 512 * 1024 * 1024
_MAX_LINE_BYTES = 64 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_COLLECTION = 100_000
_MAX_TEXT = 8_000_000
_WINDOWS_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        value = str(value)
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _text(value: Any, label: str, *, allow_empty: bool = False, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        value = str(value)
    if "\x00" in value or len(value) > maximum:
        raise ValueError(f"{label} exceeds safety bound or contains NUL")
    selected = value.strip()
    if not allow_empty and not selected:
        raise ValueError(f"{label} may not be empty")
    return selected


def _safe_path(value: str | os.PathLike[str], *, label: str, must_exist: bool, directory: bool | None = None) -> Path:
    path = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & _WINDOWS_REPARSE_POINT):
            raise ValueError(f"{label} may not traverse a symlink or reparse point")
    if must_exist and not path.exists():
        raise ValueError(f"{label} does not exist")
    if path.exists():
        if directory is True and not path.is_dir():
            raise ValueError(f"{label} must be a directory")
        if directory is False and not path.is_file():
            raise ValueError(f"{label} must be a regular file")
    return path


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _walk(root: Any, expression: str) -> list[Any]:
    selected = _identifier(expression, "field expression", 1_000)
    nodes = [root]
    for raw in selected.split("."):
        expand = raw.endswith("[]")
        key = _identifier(raw[:-2] if expand else raw, "field path segment", 300)
        following: list[Any] = []
        for node in nodes:
            if not isinstance(node, Mapping) or key not in node:
                continue
            value = node[key]
            if expand:
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                    following.extend(value)
                elif value is not None:
                    raise ValueError(f"field expression {expression!r} expected an array")
            else:
                following.append(value)
            if len(following) > _MAX_COLLECTION:
                raise ValueError("field expression expands beyond safety bound")
        nodes = following
        if not nodes:
            break
    return nodes


def _first(root: Mapping[str, Any], paths: Sequence[str]) -> Any | None:
    for path in paths:
        for value in _walk(root, path):
            if value is not None and (not isinstance(value, str) or value.strip()):
                return value
    return None


def _collect(root: Mapping[str, Any], paths: Sequence[str]) -> tuple[Any, ...]:
    values: list[Any] = []
    for path in paths:
        for value in _walk(root, path):
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, Mapping)):
                values.extend(value)
            elif value is not None:
                values.append(value)
            if len(values) > _MAX_COLLECTION:
                raise ValueError("field collection exceeds safety bound")
    return tuple(values)


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{label} must be an explicit boolean")


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        selected = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if selected < 0:
        raise ValueError(f"{label} must be non-negative")
    return selected


def _unique_text_span(answer: str, text: str, label: str) -> Mapping[str, int]:
    needle = _text(text, label)
    positions: list[int] = []
    cursor = 0
    while True:
        index = answer.find(needle, cursor)
        if index < 0:
            break
        positions.append(index)
        if len(positions) > 1:
            raise ValueError(f"{label} occurs more than once; provide explicit start/end offsets")
        cursor = index + max(1, len(needle))
    if len(positions) != 1:
        raise ValueError(f"{label} does not occur exactly once in answer")
    return {"start": positions[0], "end": positions[0] + len(needle)}


@dataclass(frozen=True)
class DeclarativeGroundedProfile:
    name: str
    id_paths: tuple[str, ...]
    prompt_paths: tuple[str, ...]
    answer_paths: tuple[str, ...]
    evidence_path: str
    evidence_id_paths: tuple[str, ...]
    evidence_text_paths: tuple[str, ...]
    evidence_source_paths: tuple[str, ...] = ()
    claims_path: str | None = None
    claim_start_paths: tuple[str, ...] = ()
    claim_end_paths: tuple[str, ...] = ()
    claim_text_paths: tuple[str, ...] = ()
    claim_evidence_id_paths: tuple[str, ...] = ()
    claim_supporting_id_paths: tuple[str, ...] = ()
    claim_contradicting_id_paths: tuple[str, ...] = ()
    claim_supported_paths: tuple[str, ...] = ()
    claim_contradicted_paths: tuple[str, ...] = ()
    unsupported_spans_path: str | None = None
    unsupported_start_paths: tuple[str, ...] = ()
    unsupported_end_paths: tuple[str, ...] = ()
    unsupported_text_paths: tuple[str, ...] = ()
    abstain_paths: tuple[str, ...] = ()
    constant_abstain: bool | None = None
    reflection_action_paths: tuple[str, ...] = ()
    constant_reflection_action: ReflectionAction | None = None
    chosen_answer_paths: tuple[str, ...] = ()
    rejected_answer_paths: tuple[str, ...] = ()
    metadata_paths: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    constant_metadata: Mapping[str, str] = field(default_factory=dict)
    generate_missing_ids: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "profile name", 300))
        tuple_fields = (
            "id_paths", "prompt_paths", "answer_paths", "evidence_id_paths", "evidence_text_paths", "evidence_source_paths",
            "claim_start_paths", "claim_end_paths", "claim_text_paths", "claim_evidence_id_paths", "claim_supporting_id_paths", "claim_contradicting_id_paths",
            "claim_supported_paths", "claim_contradicted_paths", "unsupported_start_paths", "unsupported_end_paths", "unsupported_text_paths", "abstain_paths",
            "reflection_action_paths", "chosen_answer_paths", "rejected_answer_paths",
        )
        for name in tuple_fields:
            values = tuple(_identifier(item, name, 1_000) for item in getattr(self, name))
            if len(values) > 100:
                raise ValueError(f"{name} exceeds safety bound")
            object.__setattr__(self, name, values)
        if not self.prompt_paths or not self.answer_paths or not self.evidence_id_paths or not self.evidence_text_paths:
            raise ValueError("profile requires prompt/answer/evidence id/evidence text paths")
        object.__setattr__(self, "evidence_path", _identifier(self.evidence_path, "evidence_path", 1_000))
        for name in ("claims_path", "unsupported_spans_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name, 1_000))
        if self.claims_path is not None:
            explicit_offsets = bool(self.claim_start_paths and self.claim_end_paths)
            text_offsets = bool(self.claim_text_paths)
            if explicit_offsets == text_offsets:
                raise ValueError("claims require exactly one of explicit start/end paths or claim_text_paths")
        if self.unsupported_spans_path is not None:
            explicit_offsets = bool(self.unsupported_start_paths and self.unsupported_end_paths)
            text_offsets = bool(self.unsupported_text_paths)
            if explicit_offsets == text_offsets:
                raise ValueError("unsupported spans require explicit start/end paths or unsupported_text_paths")
        if bool(self.abstain_paths) == (self.constant_abstain is not None):
            raise ValueError("configure exactly one of abstain_paths or constant_abstain")
        if bool(self.reflection_action_paths) == (self.constant_reflection_action is not None):
            raise ValueError("configure exactly one of reflection_action_paths or constant_reflection_action")
        if self.constant_reflection_action is not None and not isinstance(self.constant_reflection_action, ReflectionAction):
            object.__setattr__(self, "constant_reflection_action", ReflectionAction(self.constant_reflection_action))
        if bool(self.chosen_answer_paths) != bool(self.rejected_answer_paths):
            raise ValueError("chosen/rejected answer paths must be configured together")
        metadata = {_identifier(key, "metadata key", 300): tuple(_identifier(path, "metadata path", 1_000) for path in paths) for key, paths in self.metadata_paths.items()}
        object.__setattr__(self, "metadata_paths", metadata)
        object.__setattr__(self, "constant_metadata", {_identifier(key, "constant metadata key", 300): _identifier(str(value), "constant metadata value", 10_000) for key, value in self.constant_metadata.items()})

    @property
    def profile_sha256(self) -> str:
        payload = asdict(self)
        if self.constant_reflection_action is not None:
            payload["constant_reflection_action"] = self.constant_reflection_action.value
        return _digest({"schema": "rigorousrag-declarative-grounded-profile/v1", **payload})

    def _span(self, item: Mapping[str, Any], answer: str, *, prefix: str) -> Mapping[str, int]:
        start_paths = self.claim_start_paths if prefix == "claim" else self.unsupported_start_paths
        end_paths = self.claim_end_paths if prefix == "claim" else self.unsupported_end_paths
        text_paths = self.claim_text_paths if prefix == "claim" else self.unsupported_text_paths
        if start_paths and end_paths:
            start_raw, end_raw = _first(item, start_paths), _first(item, end_paths)
            if start_raw is None or end_raw is None:
                raise ValueError(f"{prefix} lacks explicit span offsets")
            start, end = _int(start_raw, f"{prefix} start"), _int(end_raw, f"{prefix} end")
            if end <= start or end > len(answer):
                raise ValueError(f"{prefix} span lies outside answer")
            return {"start": start, "end": end}
        text_raw = _first(item, text_paths)
        if text_raw is None:
            raise ValueError(f"{prefix} lacks text for deterministic span alignment")
        return _unique_text_span(answer, str(text_raw), f"{prefix} text")

    def adapt(self, row: Mapping[str, Any], *, dataset_id: str, split_name: str, ordinal: int) -> Mapping[str, Any]:
        prompt_raw, answer_raw = _first(row, self.prompt_paths), _first(row, self.answer_paths)
        if prompt_raw is None or answer_raw is None:
            raise ValueError("grounded row lacks prompt or answer")
        prompt, answer = _text(prompt_raw, "prompt"), _text(answer_raw, "answer", allow_empty=True)
        id_raw = _first(row, self.id_paths)
        if id_raw is None:
            if not self.generate_missing_ids:
                raise ValueError("grounded row lacks example id")
            example_id = "generated-" + hashlib.sha256(f"{dataset_id}\n{split_name}\n{ordinal}\n{prompt}\n{answer}".encode("utf-8")).hexdigest()[:32]
        else:
            example_id = _identifier(id_raw, "example id", 10_000)
        evidence_items = _walk(row, self.evidence_path)
        if len(evidence_items) == 1 and isinstance(evidence_items[0], Sequence) and not isinstance(evidence_items[0], (str, bytes, bytearray, Mapping)):
            evidence_items = list(evidence_items[0])
        if not evidence_items:
            raise ValueError("grounded row requires non-empty evidence")
        evidence = []
        for index, item in enumerate(evidence_items):
            if not isinstance(item, Mapping):
                raise ValueError("evidence collection must contain objects")
            evidence_id, text_value = _first(item, self.evidence_id_paths), _first(item, self.evidence_text_paths)
            if evidence_id is None or text_value is None:
                raise ValueError("evidence object lacks configured id/text")
            source_value = _first(item, self.evidence_source_paths) if self.evidence_source_paths else None
            evidence.append({"evidence_id": _identifier(evidence_id, "evidence id", 10_000), "text": _text(text_value, "evidence text"), "source_id": None if source_value is None else _identifier(source_value, "evidence source id", 10_000)})
        claims = []
        if self.claims_path is not None:
            claim_items = _walk(row, self.claims_path)
            if len(claim_items) == 1 and isinstance(claim_items[0], Sequence) and not isinstance(claim_items[0], (str, bytes, bytearray, Mapping)):
                claim_items = list(claim_items[0])
            for item in claim_items:
                if not isinstance(item, Mapping):
                    raise ValueError("claims collection must contain objects")
                legacy = tuple(_identifier(value, "claim evidence id", 10_000) for value in _collect(item, self.claim_evidence_id_paths))
                supporting = tuple(_identifier(value, "supporting evidence id", 10_000) for value in _collect(item, self.claim_supporting_id_paths))
                contradicting = tuple(_identifier(value, "contradicting evidence id", 10_000) for value in _collect(item, self.claim_contradicting_id_paths))
                supported_raw = _first(item, self.claim_supported_paths) if self.claim_supported_paths else None
                contradicted_raw = _first(item, self.claim_contradicted_paths) if self.claim_contradicted_paths else None
                supported = _bool(supported_raw, "claim supported") if supported_raw is not None else bool(supporting)
                contradicted = _bool(contradicted_raw, "claim contradicted") if contradicted_raw is not None else bool(contradicting)
                if (supporting or contradicting) and legacy and set(legacy) != set((*supporting, *contradicting)):
                    raise ValueError("legacy claim evidence ids differ from explicit stance union")
                claims.append({
                    "span": self._span(item, answer, prefix="claim"),
                    "evidence_ids": list(legacy or tuple(dict.fromkeys((*supporting, *contradicting)))),
                    "supporting_evidence_ids": list(supporting),
                    "contradicting_evidence_ids": list(contradicting),
                    "supported": supported,
                    "contradicted": contradicted,
                })
        unsupported = []
        if self.unsupported_spans_path is not None:
            items = _walk(row, self.unsupported_spans_path)
            if len(items) == 1 and isinstance(items[0], Sequence) and not isinstance(items[0], (str, bytes, bytearray, Mapping)):
                items = list(items[0])
            for item in items:
                if not isinstance(item, Mapping):
                    raise ValueError("unsupported span collection must contain objects")
                unsupported.append(self._span(item, answer, prefix="unsupported"))
        if self.abstain_paths:
            raw_abstain = _first(row, self.abstain_paths)
            if raw_abstain is None:
                raise ValueError("grounded row lacks configured abstention label")
            abstain = _bool(raw_abstain, "abstain")
        else:
            assert self.constant_abstain is not None
            abstain = self.constant_abstain
        if self.reflection_action_paths:
            raw_action = _first(row, self.reflection_action_paths)
            if raw_action is None:
                raise ValueError("grounded row lacks configured reflection action")
            reflection = ReflectionAction(_identifier(str(raw_action), "reflection action", 100).lower())
        else:
            assert self.constant_reflection_action is not None
            reflection = self.constant_reflection_action
        chosen = _first(row, self.chosen_answer_paths) if self.chosen_answer_paths else None
        rejected = _first(row, self.rejected_answer_paths) if self.rejected_answer_paths else None
        if (chosen is None) != (rejected is None):
            raise ValueError("grounded row has incomplete chosen/rejected pair")
        metadata: dict[str, str] = dict(self.constant_metadata)
        for key, paths in self.metadata_paths.items():
            value = _first(row, paths)
            if value is not None:
                metadata[key] = _identifier(str(value), f"metadata {key}", 10_000)
        metadata.update({"source_split": split_name, "grounded_import_profile": self.name, "grounded_import_profile_sha256": self.profile_sha256})
        payload = {
            "example_id": example_id, "prompt": prompt, "answer": answer, "evidence": evidence, "claims": claims,
            "abstain": abstain, "reflection_action": reflection.value, "unsupported_spans": unsupported,
            "chosen_answer": None if chosen is None else _text(chosen, "chosen answer"),
            "rejected_answer": None if rejected is None else _text(rejected, "rejected answer"),
            "reference_chosen_log_prob": None, "reference_rejected_log_prob": None,
            "teacher_cache_key": None, "retriever_cache_key": None, "metadata": metadata,
        }
        parse_authoritative_grounded_example(payload)
        return payload


@dataclass(frozen=True)
class GroundedSplitImportSpec:
    name: str
    source_path: str
    source_sha256: str
    profile: DeclarativeGroundedProfile
    input_format: str = "jsonl"
    expected_record_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "split name", 200))
        object.__setattr__(self, "source_path", str(_safe_path(self.source_path, label="grounded source", must_exist=True, directory=False)))
        object.__setattr__(self, "source_sha256", _sha(self.source_sha256, "source_sha256"))
        if not isinstance(self.profile, DeclarativeGroundedProfile):
            raise ValueError("profile must be DeclarativeGroundedProfile")
        selected = _identifier(self.input_format, "input_format", 20).lower()
        if selected not in {"jsonl", "json"}:
            raise ValueError("input_format must be jsonl or json")
        object.__setattr__(self, "input_format", selected)
        if self.expected_record_count is not None and (isinstance(self.expected_record_count, bool) or not isinstance(self.expected_record_count, int) or self.expected_record_count < 0):
            raise ValueError("expected_record_count must be non-negative")

    @property
    def transformation_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-grounded-split-import/v1", "name": self.name, "input_format": self.input_format, "profile_sha256": self.profile.profile_sha256})


@dataclass(frozen=True)
class GroundedDatasetGovernanceSpec:
    dataset_id: str
    exact_version: str
    source_locator: str
    license_identifier: str
    license_status: LicenseStatus
    license_evidence: str
    tasks: tuple[DatasetTask, ...]
    modalities: tuple[DatasetModality, ...]
    card: DatasetCard
    metadata: Mapping[str, str] = field(default_factory=dict)
    require_promotable: bool = False


@dataclass(frozen=True)
class GroundedSplitImportReceipt:
    name: str
    source_sha256: str
    output_path: str
    output_sha256: str
    record_count: int
    record_id_sha256: str
    evidence_id_sha256: str
    transformation_sha256: str


@dataclass(frozen=True)
class GovernedGroundedImportReceipt:
    dataset_manifest_sha256: str
    source_set_sha256: str
    transformation_sha256: str
    manifest_path: str
    splits: tuple[GroundedSplitImportReceipt, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        for name in ("dataset_manifest_sha256", "source_set_sha256", "transformation_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if not self.splits:
            raise ValueError("grounded import receipt requires splits")
        if _digest(self._unsigned()) != self.receipt_sha256:
            raise ValueError("governed grounded import receipt digest mismatch")

    def _unsigned(self) -> Mapping[str, Any]:
        return {"schema": "rigorousrag-governed-grounded-import-receipt/v1", "dataset_manifest_sha256": self.dataset_manifest_sha256, "source_set_sha256": self.source_set_sha256, "transformation_sha256": self.transformation_sha256, "manifest_path": self.manifest_path, "splits": [asdict(item) for item in self.splits]}


def _iter_rows(spec: GroundedSplitImportSpec) -> Iterator[Mapping[str, Any]]:
    source = Path(spec.source_path)
    if _stream_sha(source) != spec.source_sha256:
        raise ValueError(f"grounded split {spec.name} source digest differs from configured bytes")
    if spec.input_format == "jsonl":
        with source.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip(): continue
                if len(raw) > _MAX_LINE_BYTES: raise ValueError(f"grounded split {spec.name} line {line_number} exceeds safety bound")
                value = _strict_json_bytes(raw, f"grounded split {spec.name} line {line_number}")
                if not isinstance(value, Mapping): raise ValueError("grounded source row must be an object")
                yield value
        return
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError("grounded JSON source exceeds whole-file safety bound")
    value = _strict_json_bytes(source.read_bytes(), f"grounded split {spec.name}")
    if not isinstance(value, list): raise ValueError("grounded JSON source must be an array")
    for row in value:
        if not isinstance(row, Mapping): raise ValueError("grounded JSON row must be an object")
        yield row


def _id_digest(values: Sequence[str]) -> str:
    selected = sorted(set(values))
    return hashlib.sha256(("\n".join(selected) + ("\n" if selected else "")).encode("utf-8")).hexdigest()


def _import_split(spec: GroundedSplitImportSpec, *, dataset_id: str, output_dir: Path) -> GroundedSplitImportReceipt:
    destination = _safe_path(output_dir / f"{spec.name}.grounded.jsonl", label="grounded split output", must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    digest = hashlib.sha256(); record_ids: list[str] = []; evidence_ids: list[str] = []; count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for ordinal, row in enumerate(_iter_rows(spec), start=1):
                if count >= _MAX_RECORDS: raise ValueError("grounded split exceeds record safety bound")
                payload = spec.profile.adapt(row, dataset_id=dataset_id, split_name=spec.name, ordinal=ordinal)
                example_id = str(payload["example_id"])
                if example_id in record_ids: raise ValueError(f"duplicate grounded example id {example_id!r}")
                record_ids.append(example_id)
                evidence_ids.extend(str(item["evidence_id"]) for item in payload["evidence"])
                encoded = _canonical(payload) + b"\n"
                if len(encoded) > _MAX_LINE_BYTES: raise ValueError("canonical grounded row exceeds line safety bound")
                handle.write(encoded); digest.update(encoded); count += 1
            handle.flush(); os.fsync(handle.fileno())
        if spec.expected_record_count is not None and count != spec.expected_record_count:
            raise ValueError("grounded split record count differs from expected_record_count")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    output_sha = digest.hexdigest()
    if _stream_sha(destination) != output_sha: raise RuntimeError("grounded split changed during publication")
    return GroundedSplitImportReceipt(spec.name, spec.source_sha256, str(destination), output_sha, count, _id_digest(record_ids), _id_digest(evidence_ids), spec.transformation_sha256)


def import_governed_grounded_dataset(governance: GroundedDatasetGovernanceSpec, splits: Sequence[GroundedSplitImportSpec], *, output_dir: str | Path) -> tuple[DatasetManifest, GovernedGroundedImportReceipt]:
    if not isinstance(governance, GroundedDatasetGovernanceSpec): raise ValueError("governance must be GroundedDatasetGovernanceSpec")
    selected = tuple(splits)
    if not selected or len(selected) > 100 or any(not isinstance(item, GroundedSplitImportSpec) for item in selected): raise ValueError("splits must be a bounded non-empty GroundedSplitImportSpec sequence")
    if len({item.name for item in selected}) != len(selected): raise ValueError("grounded split names must be unique")
    root = _safe_path(output_dir, label="grounded import output", must_exist=False)
    if root.exists() and not root.is_dir(): raise ValueError("grounded import output must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    receipts = tuple(_import_split(item, dataset_id=governance.dataset_id, output_dir=root) for item in selected)
    source_set = _digest({"schema": "rigorousrag-grounded-source-set/v1", "dataset_id": governance.dataset_id, "exact_version": governance.exact_version, "splits": [{"name": item.name, "sha256": item.source_sha256} for item in receipts]})
    transformation = _digest({"schema": "rigorousrag-governed-grounded-transformation/v1", "loader": "training.governed_grounded_import", "version": "1", "splits": [{"name": item.name, "sha256": item.transformation_sha256} for item in receipts]})
    manifest = DatasetManifest(
        dataset_id=governance.dataset_id, exact_version=governance.exact_version, source_locator=governance.source_locator,
        artifact_sha256=source_set, license_identifier=governance.license_identifier, license_status=governance.license_status,
        license_evidence=governance.license_evidence, loader_name="training.governed_grounded_import", loader_version="1", transformation_sha256=transformation,
        splits=tuple(SplitManifest(name=item.name, content_sha256=item.output_sha256, record_count=item.record_count, record_id_sha256=item.record_id_sha256, query_id_sha256=item.record_id_sha256, document_id_sha256=item.evidence_id_sha256) for item in receipts),
        tasks=governance.tasks, modalities=governance.modalities, card=governance.card,
        metadata={**governance.metadata, "canonical_record_kind": "grounded_generation", "canonical_parser": "parse_authoritative_grounded_example"},
    )
    if governance.require_promotable: manifest.assert_promotable()
    manifest_path = root / "dataset_manifest.json"
    _atomic(manifest_path, _canonical({"schema": "rigorousrag-dataset-manifest/v1", "manifest": asdict(manifest), "manifest_sha256": manifest.manifest_digest}) + b"\n")
    unsigned = {"schema": "rigorousrag-governed-grounded-import-receipt/v1", "dataset_manifest_sha256": manifest.manifest_digest, "source_set_sha256": source_set, "transformation_sha256": transformation, "manifest_path": str(manifest_path), "splits": [asdict(item) for item in receipts]}
    receipt = GovernedGroundedImportReceipt(manifest.manifest_digest, source_set, transformation, str(manifest_path), receipts, _digest(unsigned))
    _atomic(root / "import_receipt.json", _canonical({**unsigned, "receipt_sha256": receipt.receipt_sha256}) + b"\n")
    return manifest, receipt


__all__ = ["DeclarativeGroundedProfile", "GovernedGroundedImportReceipt", "GroundedDatasetGovernanceSpec", "GroundedSplitImportReceipt", "GroundedSplitImportSpec", "import_governed_grounded_dataset"]
