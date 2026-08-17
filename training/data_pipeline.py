"""Local-only, manifest-bound datasets, resumable sampling and retrieval collators.

Training data is never downloaded here.  The loader consumes operator-provided local
JSONL whose byte digest is pinned before parsing.  Query/document identifiers remain
first-class so false negatives can be excluded and hard-negative generations can be
reproduced.  Sampler state is serializable for exact mid-epoch checkpoint/resume.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except Exception:  # pragma: no cover - optional training dependency.
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass

    class Sampler:  # type: ignore[no-redef]
        pass

_MAX_FILE_BYTES = 100 * 1024 * 1024 * 1024
_MAX_RECORDS = 100_000_000
_MAX_TEXT = 1_000_000


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{label} is invalid")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > _MAX_TEXT or "\x00" in result:
        raise ValueError(f"{label} is empty, too long, or contains NUL")
    return result


def _sha256(value: Any, label: str) -> str:
    digest = _identifier(value, label, 64).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be SHA-256")
    return digest


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    selected = Path(path).expanduser().resolve(strict=True)
    if not selected.is_file() or selected.is_symlink():
        raise ValueError("training dataset path must be a regular non-symlink file")
    size = selected.stat().st_size
    if size > _MAX_FILE_BYTES:
        raise ValueError("training dataset exceeds configured byte safety bound")
    digest = hashlib.sha256()
    with selected.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainingDocument:
    document_id: str
    text: str
    relevance: float = 0.0
    source_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    teacher_score: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        object.__setattr__(self, "text", _text(self.text, "document text"))
        try:
            relevance = float(self.relevance)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("relevance must be numeric") from exc
        if not relevance >= 0.0:
            raise ValueError("relevance must be non-negative")
        object.__setattr__(self, "relevance", relevance)
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        if self.teacher_score is not None:
            try:
                teacher_score = float(self.teacher_score)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("teacher_score must be numeric") from exc
            if teacher_score != teacher_score or abs(teacher_score) == float("inf"):
                raise ValueError("teacher_score must be finite")
            object.__setattr__(self, "teacher_score", teacher_score)
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 2_000:
            raise ValueError("document metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in self.metadata.items()
            },
        )


@dataclass(frozen=True)
class RetrievalTrainingExample:
    query_id: str
    query: str
    positives: tuple[TrainingDocument, ...]
    negatives: tuple[TrainingDocument, ...] = ()
    domain: str | None = None
    language: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _identifier(self.query_id, "query_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        if not self.positives or len(self.positives) > 10_000:
            raise ValueError("each query requires at least one bounded positive set")
        if len(self.negatives) > 100_000:
            raise ValueError("negative set exceeds safety bound")
        if any(not isinstance(value, TrainingDocument) for value in (*self.positives, *self.negatives)):
            raise ValueError("positives/negatives must contain TrainingDocument values")
        positive_ids = {value.document_id for value in self.positives}
        negative_ids = {value.document_id for value in self.negatives}
        overlap = positive_ids & negative_ids
        if overlap:
            raise ValueError(f"documents cannot be both positive and negative: {sorted(overlap)[:20]}")
        if len(positive_ids) != len(self.positives) or len(negative_ids) != len(self.negatives):
            raise ValueError("duplicate document ids are not allowed within a training example")
        for name in ("domain", "language"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name, 500))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 2_000:
            raise ValueError("example metadata must be a bounded mapping")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in self.metadata.items()
            },
        )

    @property
    def positive_document_ids(self) -> frozenset[str]:
        return frozenset(value.document_id for value in self.positives)


@dataclass(frozen=True)
class LocalDatasetBinding:
    path: str
    content_sha256: str
    dataset_manifest_digest: str
    split_name: str
    record_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", str(Path(self.path).expanduser().resolve()))
        object.__setattr__(self, "content_sha256", _sha256(self.content_sha256, "content_sha256"))
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            _sha256(self.dataset_manifest_digest, "dataset_manifest_digest"),
        )
        object.__setattr__(self, "split_name", _identifier(self.split_name, "split_name", 300))
        if isinstance(self.record_count, bool) or not isinstance(self.record_count, int) or not 0 <= self.record_count <= _MAX_RECORDS:
            raise ValueError("record_count is invalid")


def _parse_document(value: Any, *, default_relevance: float) -> TrainingDocument:
    if not isinstance(value, Mapping):
        raise ValueError("training document must be an object")
    return TrainingDocument(
        document_id=value.get("document_id"),
        text=value.get("text"),
        relevance=value.get("relevance", default_relevance),
        source_id=value.get("source_id"),
        metadata=value.get("metadata") or {},
        teacher_score=value.get("teacher_score"),
    )


def parse_training_example(value: Any) -> RetrievalTrainingExample:
    if not isinstance(value, Mapping):
        raise ValueError("training example must be an object")
    positives_raw = value.get("positives")
    negatives_raw = value.get("negatives") or []
    if not isinstance(positives_raw, list) or not isinstance(negatives_raw, list):
        raise ValueError("positives and negatives must be JSON arrays")
    return RetrievalTrainingExample(
        query_id=value.get("query_id"),
        query=value.get("query"),
        positives=tuple(_parse_document(item, default_relevance=1.0) for item in positives_raw),
        negatives=tuple(_parse_document(item, default_relevance=0.0) for item in negatives_raw),
        domain=value.get("domain"),
        language=value.get("language"),
        metadata=value.get("metadata") or {},
    )


class ManifestBoundJsonlDataset(Dataset):
    """Eager local JSONL dataset whose bytes are verified before any record is parsed."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str,
        dataset_manifest_digest: str,
        split_name: str,
        expected_record_count: int | None = None,
    ) -> None:
        selected = Path(path).expanduser().resolve(strict=True)
        actual_digest = sha256_file(selected)
        expected = _sha256(expected_sha256, "expected_sha256")
        if actual_digest != expected:
            raise ValueError("local training data digest does not match expected split artifact")
        examples: list[RetrievalTrainingExample] = []
        with selected.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(examples) >= _MAX_RECORDS:
                    raise ValueError("training dataset exceeds record safety bound")
                try:
                    payload = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                except Exception as exc:
                    raise ValueError(f"invalid JSON at training data line {line_number}") from exc
                examples.append(parse_training_example(payload))
        if expected_record_count is not None and len(examples) != expected_record_count:
            raise ValueError("training dataset record count does not match manifest expectation")
        query_ids = [value.query_id for value in examples]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_id must be unique within the governed training split")
        self._examples = tuple(examples)
        self.binding = LocalDatasetBinding(
            path=str(selected),
            content_sha256=actual_digest,
            dataset_manifest_digest=dataset_manifest_digest,
            split_name=split_name,
            record_count=len(examples),
        )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> RetrievalTrainingExample:
        return self._examples[index]


@dataclass(frozen=True)
class SamplerState:
    epoch: int
    cursor: int
    seed: int
    dataset_size: int
    shuffle: bool


class ResumableDeterministicSampler(Sampler):
    """Epoch-seeded deterministic sampler with exact mid-epoch cursor restoration."""

    def __init__(self, dataset_size: int, *, seed: int = 0, shuffle: bool = True) -> None:
        if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size < 0:
            raise ValueError("dataset_size must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
            raise ValueError("seed must be a non-negative 63-bit integer")
        self.dataset_size = dataset_size
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.epoch = 0
        self.cursor = 0

    def _order(self) -> list[int]:
        order = list(range(self.dataset_size))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        while self.cursor < len(order):
            value = order[self.cursor]
            self.cursor += 1
            yield value
        self.epoch += 1
        self.cursor = 0

    def __len__(self) -> int:
        return max(0, self.dataset_size - self.cursor)

    def state_dict(self) -> dict[str, Any]:
        return asdict(SamplerState(self.epoch, self.cursor, self.seed, self.dataset_size, self.shuffle))

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        candidate = SamplerState(
            epoch=int(state["epoch"]),
            cursor=int(state["cursor"]),
            seed=int(state["seed"]),
            dataset_size=int(state["dataset_size"]),
            shuffle=bool(state["shuffle"]),
        )
        if candidate.seed != self.seed or candidate.dataset_size != self.dataset_size or candidate.shuffle != self.shuffle:
            raise ValueError("sampler checkpoint is incompatible with configured dataset/seed/shuffle")
        if candidate.epoch < 0 or not 0 <= candidate.cursor <= self.dataset_size:
            raise ValueError("sampler checkpoint cursor/epoch is invalid")
        self.epoch = candidate.epoch
        self.cursor = candidate.cursor


@dataclass(frozen=True)
class CollatorConfig:
    query_max_length: int = 64
    document_max_length: int = 512
    negatives_per_query: int = 8
    positive_selection_seed: int = 0
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        for name in ("query_max_length", "document_max_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.negatives_per_query, bool) or not isinstance(self.negatives_per_query, int) or self.negatives_per_query < 0:
            raise ValueError("negatives_per_query must be non-negative")
        if self.pad_to_multiple_of is not None and (
            isinstance(self.pad_to_multiple_of, bool)
            or not isinstance(self.pad_to_multiple_of, int)
            or self.pad_to_multiple_of <= 0
        ):
            raise ValueError("pad_to_multiple_of must be positive or None")


def _tokenize(tokenizer: Any, texts: Sequence[str], *, max_length: int, pad_to_multiple_of: int | None) -> Mapping[str, Any]:
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )


class BiEncoderCollator:
    """Build one positive plus deterministic hard negatives per query.

    Candidate documents are concatenated by query. ``positive_indices`` points to the
    selected positive inside the global document batch.  ``false_negative_mask`` marks
    global candidates known positive for each query so a training step can mask them
    rather than accidentally train them as negatives.
    """

    def __init__(self, tokenizer: Any, config: CollatorConfig = CollatorConfig()) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self._calls = 0

    def __call__(self, examples: Sequence[RetrievalTrainingExample]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("collation requires optional PyTorch training dependencies")
        if not examples:
            raise ValueError("cannot collate an empty batch")
        rng = random.Random(self.config.positive_selection_seed + self._calls)
        self._calls += 1
        queries: list[str] = []
        documents: list[TrainingDocument] = []
        positive_indices: list[int] = []
        known_positive_ids: list[frozenset[str]] = []
        candidate_ranges: list[tuple[int, int]] = []
        for example in examples:
            queries.append(example.query)
            selected_positive = example.positives[rng.randrange(len(example.positives))]
            negatives = list(example.negatives)
            rng.shuffle(negatives)
            negatives = negatives[: self.config.negatives_per_query]
            start = len(documents)
            documents.append(selected_positive)
            documents.extend(negatives)
            positive_indices.append(start)
            candidate_ranges.append((start, len(documents)))
            known_positive_ids.append(example.positive_document_ids)
        query_inputs = _tokenize(
            self.tokenizer,
            queries,
            max_length=self.config.query_max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        document_inputs = _tokenize(
            self.tokenizer,
            [value.text for value in documents],
            max_length=self.config.document_max_length,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
        )
        false_negative_mask = torch.zeros(len(examples), len(documents), dtype=torch.bool)
        for row, positives in enumerate(known_positive_ids):
            for column, document in enumerate(documents):
                if document.document_id in positives and column != positive_indices[row]:
                    false_negative_mask[row, column] = True
        teacher_scores = torch.full((len(examples), len(documents)), float("nan"), dtype=torch.float32)
        for row, (start, end) in enumerate(candidate_ranges):
            for column in range(start, end):
                value = documents[column].teacher_score
                if value is not None:
                    teacher_scores[row, column] = value
        return {
            "query_inputs": query_inputs,
            "document_inputs": document_inputs,
            "positive_indices": torch.tensor(positive_indices, dtype=torch.long),
            "false_negative_mask": false_negative_mask,
            "teacher_scores": teacher_scores,
            "query_ids": tuple(value.query_id for value in examples),
            "document_ids": tuple(value.document_id for value in documents),
            "candidate_ranges": tuple(candidate_ranges),
        }

    def state_dict(self) -> dict[str, int]:
        return {"calls": self._calls}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        calls = int(state.get("calls", 0))
        if calls < 0:
            raise ValueError("collator call counter may not be negative")
        self._calls = calls


class CrossEncoderCollator:
    """Tokenize query-document pairs and preserve grouped graded relevance labels."""

    def __init__(self, tokenizer: Any, *, max_length: int = 512, negatives_per_query: int = 8, seed: int = 0) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.negatives_per_query = int(negatives_per_query)
        self.seed = int(seed)
        self._calls = 0
        if self.max_length <= 0 or self.negatives_per_query < 0:
            raise ValueError("invalid cross-encoder collator lengths")

    def __call__(self, examples: Sequence[RetrievalTrainingExample]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("collation requires optional PyTorch training dependencies")
        rng = random.Random(self.seed + self._calls)
        self._calls += 1
        queries: list[str] = []
        documents: list[str] = []
        relevance: list[float] = []
        teacher_scores: list[float] = []
        group_sizes: list[int] = []
        pair_ids: list[tuple[str, str]] = []
        for example in examples:
            candidates = list(example.positives) + list(example.negatives)
            positives = list(example.positives)
            negatives = list(example.negatives)
            rng.shuffle(negatives)
            candidates = positives + negatives[: self.negatives_per_query]
            if not candidates:
                raise ValueError("cross-encoder example has no candidates")
            group_sizes.append(len(candidates))
            for document in candidates:
                queries.append(example.query)
                documents.append(document.text)
                relevance.append(document.relevance)
                teacher_scores.append(float("nan") if document.teacher_score is None else document.teacher_score)
                pair_ids.append((example.query_id, document.document_id))
        pair_inputs = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pair_inputs": pair_inputs,
            "relevance": torch.tensor(relevance, dtype=torch.float32),
            "teacher_scores": torch.tensor(teacher_scores, dtype=torch.float32),
            "group_sizes": tuple(group_sizes),
            "pair_ids": tuple(pair_ids),
        }

    def state_dict(self) -> dict[str, int]:
        return {"calls": self._calls}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        calls = int(state.get("calls", 0))
        if calls < 0:
            raise ValueError("collator call counter may not be negative")
        self._calls = calls


__all__ = [
    "BiEncoderCollator",
    "CollatorConfig",
    "CrossEncoderCollator",
    "LocalDatasetBinding",
    "ManifestBoundJsonlDataset",
    "ResumableDeterministicSampler",
    "RetrievalTrainingExample",
    "SamplerState",
    "TrainingDocument",
    "parse_training_example",
    "sha256_file",
]
