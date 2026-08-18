"""Manifest-bound advanced RAG datasets and deterministic training collators.

This module closes the data-to-tensor seam for grounded-generator and generation-time
retrieval-policy training. It does not download data, load pretrained weights, or execute a
training loop. Operators provide local JSONL artifacts whose bytes and dataset-manifest
identities are pinned before parsing.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    class Dataset:  # type: ignore[no-redef]
        pass

from training.dynamic_retrieval_policy import DEFAULT_FEATURE_NAMES, DynamicPolicyArchitecture, DynamicRetrievalAction
from training.grounded_generation import ReflectionAction

_MAX_FILE_BYTES = 100 * 1024 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_TEXT = 2_000_000
_MAX_EVIDENCE = 4096
_MAX_CLAIMS = 4096
_MAX_SPANS = 100_000
_HEX = frozenset("0123456789abcdef")


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("advanced RAG collation requires optional PyTorch dependencies")


def _identifier(value: Any, label: str, maximum: int = 2000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if "\x00" in value or len(value) > _MAX_TEXT or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} is empty, too long, or contains NUL")
    return value


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
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


def sha256_file(path: str | Path) -> str:
    selected = Path(path).expanduser().resolve(strict=True)
    if not selected.is_file() or selected.is_symlink():
        raise ValueError("advanced training dataset must be a regular non-symlink file")
    if selected.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("advanced training dataset exceeds byte safety bound")
    digest = hashlib.sha256()
    with selected.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, order=True)
class TextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool) or not isinstance(self.start, int) or not isinstance(self.end, int) or self.start < 0 or self.end <= self.start or self.end > _MAX_TEXT:
            raise ValueError("text span must be a positive bounded half-open interval")


@dataclass(frozen=True)
class GroundedEvidenceRecord:
    evidence_id: str
    text: str
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "text", _text(self.text, "evidence text"))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))


@dataclass(frozen=True)
class GroundedClaimAnnotation:
    span: TextSpan
    evidence_ids: tuple[str, ...] = ()
    supported: bool = False
    contradicted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.span, TextSpan):
            raise ValueError("claim span must be TextSpan")
        ids = tuple(_identifier(value, "claim evidence id") for value in self.evidence_ids)
        if len(ids) > _MAX_EVIDENCE or len(set(ids)) != len(ids):
            raise ValueError("claim evidence ids must be unique and bounded")
        object.__setattr__(self, "evidence_ids", ids)
        if bool(self.supported) and bool(self.contradicted):
            raise ValueError("claim cannot be simultaneously supported and contradicted")


@dataclass(frozen=True)
class GroundedGenerationExample:
    example_id: str
    prompt: str
    answer: str
    evidence: tuple[GroundedEvidenceRecord, ...]
    claims: tuple[GroundedClaimAnnotation, ...] = ()
    abstain: bool = False
    reflection_action: ReflectionAction = ReflectionAction.STOP
    unsupported_spans: tuple[TextSpan, ...] = ()
    chosen_answer: str | None = None
    rejected_answer: str | None = None
    reference_chosen_log_prob: float | None = None
    reference_rejected_log_prob: float | None = None
    teacher_cache_key: str | None = None
    retriever_cache_key: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _identifier(self.example_id, "example_id"))
        object.__setattr__(self, "prompt", _text(self.prompt, "prompt"))
        object.__setattr__(self, "answer", _text(self.answer, "answer", allow_empty=bool(self.abstain)))
        evidence = tuple(self.evidence)
        if not evidence or len(evidence) > _MAX_EVIDENCE or any(not isinstance(item, GroundedEvidenceRecord) for item in evidence):
            raise ValueError("grounded example requires a bounded non-empty evidence set")
        ids = [item.evidence_id for item in evidence]
        if len(set(ids)) != len(ids):
            raise ValueError("grounded evidence ids must be unique")
        object.__setattr__(self, "evidence", evidence)
        claims = tuple(self.claims)
        if len(claims) > _MAX_CLAIMS or any(not isinstance(item, GroundedClaimAnnotation) for item in claims):
            raise ValueError("claims must be bounded GroundedClaimAnnotation values")
        known = set(ids)
        for claim in claims:
            if claim.span.end > len(self.answer):
                raise ValueError("claim span lies outside answer")
            unknown = set(claim.evidence_ids) - known
            if unknown:
                raise ValueError(f"claim references unknown evidence ids: {sorted(unknown)[:20]}")
        object.__setattr__(self, "claims", claims)
        spans = tuple(self.unsupported_spans)
        if len(spans) > _MAX_SPANS or any(not isinstance(span, TextSpan) for span in spans) or any(span.end > len(self.answer) for span in spans):
            raise ValueError("unsupported spans are invalid or lie outside answer")
        object.__setattr__(self, "unsupported_spans", spans)
        if not isinstance(self.reflection_action, ReflectionAction):
            object.__setattr__(self, "reflection_action", ReflectionAction(self.reflection_action))
        if (self.chosen_answer is None) != (self.rejected_answer is None):
            raise ValueError("chosen_answer and rejected_answer must be supplied together")
        for name in ("chosen_answer", "rejected_answer"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        for name in ("reference_chosen_log_prob", "reference_rejected_log_prob"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if (self.reference_chosen_log_prob is None) != (self.reference_rejected_log_prob is None):
            raise ValueError("reference preference log probabilities must be supplied together")
        for name in ("teacher_cache_key", "retriever_cache_key"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 2000:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {_identifier(str(k), "metadata key", 300): _identifier(str(v), "metadata value", 10000) for k, v in self.metadata.items()})


@dataclass(frozen=True)
class DynamicRagEpisodeStep:
    episode_id: str
    step_id: str
    context: str
    features: Mapping[str, float]
    action: DynamicRetrievalAction
    realized_retrieval_gain: float = 0.0
    behavior_action_probability: float | None = None
    advantage: float | None = None
    need_spans: tuple[TextSpan, ...] = ()
    hidden_state_cache_key: str | None = None
    terminal_utility: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "step_id", _identifier(self.step_id, "step_id"))
        object.__setattr__(self, "context", _text(self.context, "context"))
        if not isinstance(self.action, DynamicRetrievalAction):
            object.__setattr__(self, "action", DynamicRetrievalAction(self.action))
        if not isinstance(self.features, Mapping):
            raise ValueError("features must be a mapping")
        selected = {str(k): _finite(v, f"feature {k}") for k, v in self.features.items()}
        unknown = set(selected) - set(DEFAULT_FEATURE_NAMES)
        missing = set(DEFAULT_FEATURE_NAMES) - set(selected)
        if unknown or missing:
            raise ValueError(f"dynamic feature schema mismatch; unknown={sorted(unknown)}, missing={sorted(missing)}")
        object.__setattr__(self, "features", selected)
        object.__setattr__(self, "realized_retrieval_gain", _finite(self.realized_retrieval_gain, "realized_retrieval_gain"))
        if self.behavior_action_probability is not None:
            probability = _finite(self.behavior_action_probability, "behavior_action_probability")
            if not 0.0 < probability <= 1.0:
                raise ValueError("behavior_action_probability must lie in (0,1]")
            object.__setattr__(self, "behavior_action_probability", probability)
        if self.advantage is not None:
            object.__setattr__(self, "advantage", _finite(self.advantage, "advantage"))
        spans = tuple(self.need_spans)
        if len(spans) > _MAX_SPANS or any(not isinstance(span, TextSpan) for span in spans) or any(span.end > len(self.context) for span in spans):
            raise ValueError("information-need spans are invalid or lie outside context")
        object.__setattr__(self, "need_spans", spans)
        if self.hidden_state_cache_key is not None:
            object.__setattr__(self, "hidden_state_cache_key", _identifier(self.hidden_state_cache_key, "hidden_state_cache_key"))
        if self.terminal_utility is not None:
            object.__setattr__(self, "terminal_utility", _finite(self.terminal_utility, "terminal_utility"))


@dataclass(frozen=True)
class AdvancedDatasetBinding:
    path: str
    content_sha256: str
    dataset_manifest_sha256: str
    split_name: str
    record_count: int
    record_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(Path(self.path).expanduser().resolve()))
        object.__setattr__(self, "content_sha256", _sha256(self.content_sha256, "content_sha256"))
        object.__setattr__(self, "dataset_manifest_sha256", _sha256(self.dataset_manifest_sha256, "dataset_manifest_sha256"))
        object.__setattr__(self, "split_name", _identifier(self.split_name, "split_name", 300))
        object.__setattr__(self, "record_kind", _identifier(self.record_kind, "record_kind", 100))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or not 0 <= self.record_count <= _MAX_RECORDS:
            raise ValueError("record_count is invalid")


def _parse_span(value: Any) -> TextSpan:
    if not isinstance(value, Mapping):
        raise ValueError("span must be an object")
    return TextSpan(start=value.get("start"), end=value.get("end"))


def parse_grounded_example(value: Any) -> GroundedGenerationExample:
    if not isinstance(value, Mapping):
        raise ValueError("grounded training record must be an object")
    evidence_raw = value.get("evidence")
    claims_raw = value.get("claims") or []
    unsupported_raw = value.get("unsupported_spans") or []
    if not isinstance(evidence_raw, list) or not isinstance(claims_raw, list) or not isinstance(unsupported_raw, list):
        raise ValueError("evidence/claims/unsupported_spans must be arrays")
    evidence = []
    for item in evidence_raw:
        if not isinstance(item, Mapping):
            raise ValueError("evidence entry must be an object")
        evidence.append(GroundedEvidenceRecord(item.get("evidence_id"), item.get("text"), item.get("source_id")))
    claims = []
    for item in claims_raw:
        if not isinstance(item, Mapping) or not isinstance(item.get("evidence_ids") or [], list):
            raise ValueError("claim annotation is invalid")
        claims.append(GroundedClaimAnnotation(_parse_span(item.get("span")), tuple(item.get("evidence_ids") or []), bool(item.get("supported", False)), bool(item.get("contradicted", False))))
    return GroundedGenerationExample(
        example_id=value.get("example_id"), prompt=value.get("prompt"), answer=value.get("answer", ""),
        evidence=tuple(evidence), claims=tuple(claims), abstain=bool(value.get("abstain", False)),
        reflection_action=value.get("reflection_action", ReflectionAction.STOP.value),
        unsupported_spans=tuple(_parse_span(item) for item in unsupported_raw), chosen_answer=value.get("chosen_answer"),
        rejected_answer=value.get("rejected_answer"), reference_chosen_log_prob=value.get("reference_chosen_log_prob"),
        reference_rejected_log_prob=value.get("reference_rejected_log_prob"), teacher_cache_key=value.get("teacher_cache_key"),
        retriever_cache_key=value.get("retriever_cache_key"), metadata=value.get("metadata") or {},
    )


def parse_dynamic_episode_step(value: Any) -> DynamicRagEpisodeStep:
    if not isinstance(value, Mapping):
        raise ValueError("dynamic episode record must be an object")
    need = value.get("need_spans") or []
    if not isinstance(need, list):
        raise ValueError("need_spans must be an array")
    return DynamicRagEpisodeStep(
        episode_id=value.get("episode_id"), step_id=value.get("step_id"), context=value.get("context"),
        features=value.get("features") or {}, action=value.get("action"), realized_retrieval_gain=value.get("realized_retrieval_gain", 0.0),
        behavior_action_probability=value.get("behavior_action_probability"), advantage=value.get("advantage"),
        need_spans=tuple(_parse_span(item) for item in need), hidden_state_cache_key=value.get("hidden_state_cache_key"),
        terminal_utility=value.get("terminal_utility"), metadata=value.get("metadata") or {},
    )


class ManifestBoundAdvancedJsonlDataset(Dataset):
    def __init__(self, path: str | Path, *, expected_sha256: str, dataset_manifest_sha256: str, split_name: str, record_kind: str, expected_record_count: int | None = None) -> None:
        selected = Path(path).expanduser().resolve(strict=True)
        actual = sha256_file(selected)
        if actual != _sha256(expected_sha256, "expected_sha256"):
            raise ValueError("local advanced-training data digest does not match expected artifact")
        if record_kind not in {"grounded_generation", "dynamic_rag_episode"}:
            raise ValueError("record_kind must be grounded_generation or dynamic_rag_episode")
        parser = parse_grounded_example if record_kind == "grounded_generation" else parse_dynamic_episode_step
        records = []
        with selected.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
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


class TensorCacheProvider(Protocol):
    def get(self, key: str) -> Mapping[str, Any]: ...


class RetrieverBatchBuilder(Protocol):
    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class GroundedCollatorConfig:
    sequence_max_length: int = 2048
    evidence_max_length: int = 512
    evidence_limit: int = 32
    claim_limit: int = 64
    ignore_index: int = -100
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        for name in ("sequence_max_length", "evidence_max_length", "evidence_limit", "claim_limit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")


def _tokenize_with_offsets(tokenizer: Any, texts: Sequence[str], *, max_length: int, pad_to_multiple_of: int | None) -> Mapping[str, Any]:
    encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, pad_to_multiple_of=pad_to_multiple_of, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=True)
    if "input_ids" not in encoded or "attention_mask" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("tokenizer must return input_ids, attention_mask and offset_mapping")
    return encoded


def _overlap(offsets: Sequence[Sequence[int]], span: TextSpan, base: int = 0) -> list[int]:
    start, end = span.start + base, span.end + base
    return [i for i, pair in enumerate(offsets) if int(pair[1]) > int(pair[0]) and int(pair[0]) < end and int(pair[1]) > start]


def _causal_labels(input_ids: Any, attention_mask: Any, starts: Sequence[int], ignore_index: int) -> Any:
    _require_torch()
    labels = torch.full_like(input_ids, int(ignore_index))
    for row in range(input_ids.size(0)):
        valid = int(attention_mask[row].sum().item())
        start = max(1, int(starts[row]))
        for position in range(max(0, start - 1), max(0, valid - 1)):
            labels[row, position] = input_ids[row, position + 1]
    return labels


class GroundedGenerationCollator:
    def __init__(self, tokenizer: Any, config: GroundedCollatorConfig = GroundedCollatorConfig(), *, teacher_cache: TensorCacheProvider | None = None, retriever_batch_builder: RetrieverBatchBuilder | None = None) -> None:
        self.tokenizer, self.config = tokenizer, config
        self.teacher_cache, self.retriever_batch_builder = teacher_cache, retriever_batch_builder
        self._calls = 0

    def state_dict(self) -> dict[str, Any]:
        return {"schema": "rigorousrag-grounded-collator-state/v1", "calls": self._calls, "config": asdict(self.config)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "rigorousrag-grounded-collator-state/v1" or state.get("config") != asdict(self.config):
            raise ValueError("grounded collator checkpoint is incompatible")
        calls = state.get("calls")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("grounded collator call counter is invalid")
        self._calls = calls

    def _preference(self, examples: Sequence[GroundedGenerationExample], attribute: str) -> tuple[Mapping[str, Any], Any]:
        texts, char_starts = [], []
        for example in examples:
            answer = getattr(example, attribute)
            if answer is None:
                raise ValueError("preference objective requires complete chosen/rejected annotations")
            prefix = example.prompt + "\n\n"
            texts.append(prefix + answer)
            char_starts.append(len(prefix))
        encoded = dict(_tokenize_with_offsets(self.tokenizer, texts, max_length=self.config.sequence_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of))
        offsets = encoded.pop("offset_mapping").tolist()
        token_starts = []
        for row, char_start in enumerate(char_starts):
            positions = [i for i, pair in enumerate(offsets[row]) if int(pair[1]) > int(pair[0]) and int(pair[1]) > char_start]
            if not positions:
                raise ValueError("preference answer was entirely truncated")
            token_starts.append(min(positions))
        return encoded, _causal_labels(encoded["input_ids"], encoded["attention_mask"], token_starts, self.config.ignore_index)

    def __call__(self, examples: Sequence[GroundedGenerationExample]) -> dict[str, Any]:
        _require_torch()
        if not examples or any(not isinstance(item, GroundedGenerationExample) for item in examples):
            raise ValueError("grounded collator requires GroundedGenerationExample values")
        self._calls += 1
        prefixes = [item.prompt + "\n\n" for item in examples]
        texts = [prefix + item.answer for prefix, item in zip(prefixes, examples)]
        encoded = dict(_tokenize_with_offsets(self.tokenizer, texts, max_length=self.config.sequence_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of))
        offsets = encoded.pop("offset_mapping").tolist()
        selected_evidence = [item.evidence[: self.config.evidence_limit] for item in examples]
        max_evidence = max(len(group) for group in selected_evidence)
        max_claims = max(1, min(self.config.claim_limit, max((len(item.claims) for item in examples), default=0)))
        answer_starts, generation_indices = [], []
        claim_rows, claim_masks, citation_rows, support_rows, contradiction_rows, unsupported_rows = [], [], [], [], [], []
        for row, example in enumerate(examples):
            valid = [i for i, pair in enumerate(offsets[row]) if int(pair[1]) > int(pair[0]) and bool(encoded["attention_mask"][row, i].item())]
            answer_positions = [i for i in valid if int(offsets[row][i][1]) > len(prefixes[row])]
            if not valid or (example.answer and not answer_positions):
                raise ValueError("answer was entirely truncated; increase sequence_max_length")
            answer_start = min(answer_positions) if answer_positions else max(valid)
            generation_index = max(answer_positions) if answer_positions else max(valid)
            answer_starts.append(answer_start)
            generation_indices.append(generation_index)
            evidence_index = {item.evidence_id: i for i, item in enumerate(selected_evidence[row])}
            cpos, cmask, ctgt, sup, con = [], [], [], [], []
            for claim in example.claims[:max_claims]:
                positions = _overlap(offsets[row], claim.span, len(prefixes[row]))
                if not positions:
                    raise ValueError("claim annotation was truncated or has no token overlap")
                cpos.append(positions[-1]); cmask.append(True)
                available = sorted(evidence_index[eid] for eid in claim.evidence_ids if eid in evidence_index)
                ctgt.append(available[0] if available else self.config.ignore_index)
                sup.append(1.0 if claim.supported else 0.0); con.append(1.0 if claim.contradicted else 0.0)
            while len(cpos) < max_claims:
                cpos.append(generation_index); cmask.append(False); ctgt.append(self.config.ignore_index); sup.append(0.0); con.append(0.0)
            claim_rows.append(cpos); claim_masks.append(cmask); citation_rows.append(ctgt); support_rows.append(sup); contradiction_rows.append(con)
            unsupported = [False] * len(offsets[row])
            for span in example.unsupported_spans:
                positions = _overlap(offsets[row], span, len(prefixes[row]))
                if not positions:
                    raise ValueError("unsupported span was truncated or has no token overlap")
                for position in positions:
                    unsupported[position] = True
            unsupported_rows.append(unsupported)
        labels = _causal_labels(encoded["input_ids"], encoded["attention_mask"], answer_starts, self.config.ignore_index)
        flat = [item.text for group in selected_evidence for item in group]
        evidence_encoded = self.tokenizer(flat, padding=True, truncation=True, max_length=self.config.evidence_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of, return_tensors="pt", add_special_tokens=True)
        length = evidence_encoded["input_ids"].size(1)
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        evidence_ids = torch.full((len(examples), max_evidence, length), int(pad_id), dtype=evidence_encoded["input_ids"].dtype)
        evidence_mask = torch.zeros((len(examples), max_evidence, length), dtype=evidence_encoded["attention_mask"].dtype)
        cursor = 0
        for row, group in enumerate(selected_evidence):
            count = len(group)
            evidence_ids[row, :count] = evidence_encoded["input_ids"][cursor:cursor+count]
            evidence_mask[row, :count] = evidence_encoded["attention_mask"][cursor:cursor+count]
            cursor += count
        batch = {
            "example_ids": tuple(item.example_id for item in examples),
            "model_inputs": {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "claim_token_indices": torch.tensor(claim_rows, dtype=torch.long), "generation_token_index": torch.tensor(generation_indices, dtype=torch.long), "evidence_input_ids": evidence_ids, "evidence_attention_mask": evidence_mask},
            "labels": labels, "citation_targets": torch.tensor(citation_rows, dtype=torch.long),
            "support_targets": torch.tensor(support_rows, dtype=torch.float32), "contradiction_targets": torch.tensor(contradiction_rows, dtype=torch.float32),
            "claim_mask": torch.tensor(claim_masks, dtype=torch.bool), "abstention_targets": torch.tensor([1.0 if item.abstain else 0.0 for item in examples], dtype=torch.float32),
            "reflection_targets": torch.tensor([list(ReflectionAction).index(item.reflection_action) for item in examples], dtype=torch.long),
            "unsupported_token_mask": torch.tensor(unsupported_rows, dtype=torch.bool),
        }
        if any(item.chosen_answer is not None for item in examples):
            if not all(item.chosen_answer is not None for item in examples):
                raise ValueError("preference batches may not mix annotated and unannotated examples")
            chosen_inputs, chosen_labels = self._preference(examples, "chosen_answer")
            rejected_inputs, rejected_labels = self._preference(examples, "rejected_answer")
            batch["model_inputs"]["chosen_inputs"] = chosen_inputs; batch["model_inputs"]["rejected_inputs"] = rejected_inputs
            batch["chosen_labels"] = chosen_labels; batch["rejected_labels"] = rejected_labels
            if all(item.reference_chosen_log_prob is not None for item in examples):
                batch["reference_chosen_log_prob"] = torch.tensor([float(item.reference_chosen_log_prob) for item in examples], dtype=torch.float32)
                batch["reference_rejected_log_prob"] = torch.tensor([float(item.reference_rejected_log_prob) for item in examples], dtype=torch.float32)
        if self.teacher_cache is not None and all(item.teacher_cache_key is not None for item in examples):
            batch["teacher_token_logits"] = torch.stack([self.teacher_cache.get(str(item.teacher_cache_key))["teacher_token_logits"] for item in examples])
        if self.retriever_batch_builder is not None:
            retriever = dict(self.retriever_batch_builder(examples))
            if "model_inputs" not in retriever or "document_lm_log_likelihood" not in retriever:
                raise ValueError("retriever batch builder must return model_inputs and document_lm_log_likelihood")
            batch["model_inputs"]["retriever_inputs"] = retriever["model_inputs"]
            batch["document_lm_log_likelihood"] = retriever["document_lm_log_likelihood"]
            if "retriever_candidate_mask" in retriever:
                batch["retriever_candidate_mask"] = retriever["retriever_candidate_mask"]
        return batch


@dataclass(frozen=True)
class DynamicCollatorConfig:
    context_max_length: int = 2048
    ignore_index: int = -100
    pad_to_multiple_of: int | None = 8


class DynamicRagEpisodeCollator:
    def __init__(self, tokenizer: Any, architecture: DynamicPolicyArchitecture, config: DynamicCollatorConfig = DynamicCollatorConfig(), *, hidden_state_cache: TensorCacheProvider | None = None) -> None:
        if not isinstance(architecture, DynamicPolicyArchitecture):
            raise ValueError("architecture must be DynamicPolicyArchitecture")
        self.tokenizer, self.architecture, self.config = tokenizer, architecture, config
        self.hidden_state_cache = hidden_state_cache
        self._calls = 0

    def state_dict(self) -> dict[str, Any]:
        return {"schema": "rigorousrag-dynamic-collator-state/v1", "calls": self._calls, "architecture_sha256": self.architecture.architecture_sha256, "config": asdict(self.config)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "rigorousrag-dynamic-collator-state/v1" or state.get("architecture_sha256") != self.architecture.architecture_sha256 or state.get("config") != asdict(self.config):
            raise ValueError("dynamic collator checkpoint is incompatible")
        calls = state.get("calls")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise ValueError("dynamic collator calls is invalid")
        self._calls = calls

    def __call__(self, examples: Sequence[DynamicRagEpisodeStep]) -> dict[str, Any]:
        _require_torch()
        if not examples or any(not isinstance(item, DynamicRagEpisodeStep) for item in examples):
            raise ValueError("dynamic collator requires DynamicRagEpisodeStep values")
        self._calls += 1
        actions = {action: index for index, action in enumerate(self.architecture.actions)}
        batch = {
            "episode_step_ids": tuple(f"{item.episode_id}:{item.step_id}" for item in examples),
            "features": torch.tensor([[float(item.features[name]) for name in self.architecture.feature_names] for item in examples], dtype=torch.float32),
            "action_targets": torch.tensor([actions[item.action] for item in examples], dtype=torch.long),
            "logged_action_indices": torch.tensor([actions[item.action] for item in examples], dtype=torch.long),
            "realized_retrieval_gain": torch.tensor([item.realized_retrieval_gain for item in examples], dtype=torch.float32),
        }
        if all(item.advantage is not None for item in examples):
            batch["advantage"] = torch.tensor([float(item.advantage) for item in examples], dtype=torch.float32)
        if all(item.behavior_action_probability is not None for item in examples):
            batch["importance_ratio"] = torch.ones(len(examples), dtype=torch.float32)
        need_annotated = any(item.need_spans for item in examples)
        hidden_available = self.hidden_state_cache is not None and all(item.hidden_state_cache_key is not None for item in examples)
        if need_annotated or hidden_available:
            encoded = dict(_tokenize_with_offsets(self.tokenizer, [item.context for item in examples], max_length=self.config.context_max_length, pad_to_multiple_of=self.config.pad_to_multiple_of))
            offsets = encoded.pop("offset_mapping").tolist()
            target = torch.zeros_like(encoded["attention_mask"], dtype=torch.float32)
            valid = encoded["attention_mask"].to(dtype=torch.bool)
            for row, item in enumerate(examples):
                for span in item.need_spans:
                    positions = _overlap(offsets[row], span)
                    if not positions:
                        raise ValueError("information-need annotation was truncated")
                    for position in positions:
                        target[row, position] = 1.0
            batch["need_target_mask"] = target; batch["need_valid_mask"] = valid; batch["attention_mask"] = valid
        if hidden_available:
            cached = [self.hidden_state_cache.get(str(item.hidden_state_cache_key)) for item in examples]
            batch["token_hidden"] = torch.stack([item["token_hidden"] for item in cached])
            batch["state_hidden"] = torch.stack([item["state_hidden"] for item in cached])
        return batch


__all__ = ["AdvancedDatasetBinding", "DynamicCollatorConfig", "DynamicRagEpisodeCollator", "DynamicRagEpisodeStep", "GroundedClaimAnnotation", "GroundedCollatorConfig", "GroundedEvidenceRecord", "GroundedGenerationCollator", "GroundedGenerationExample", "ManifestBoundAdvancedJsonlDataset", "RetrieverBatchBuilder", "TensorCacheProvider", "TextSpan", "parse_dynamic_episode_step", "parse_grounded_example", "sha256_file"]
