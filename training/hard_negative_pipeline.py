"""Reproducible hard-negative and teacher-distillation data generation.

The miner executes only when explicitly called.  It accepts an injected retriever and
optional teacher scorer, filters all known positives before negative selection, supports
rank-based, semi-hard and teacher-difficulty curricula, writes immutable local JSONL,
and emits a manifest binding every generation to the source split and retriever/teacher
artifacts.  It never downloads a corpus or model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from training.data_pipeline import RetrievalTrainingExample, TrainingDocument

_HEX = frozenset("0123456789abcdef")


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _finite(value: Any, label: str) -> float:
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class RetrievedCandidate:
    document_id: str
    text: str
    rank: int
    retriever_score: float
    source_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _identifier(self.document_id, "document_id"))
        if not isinstance(self.text, str) or not self.text.strip() or "\x00" in self.text:
            raise ValueError("candidate text is invalid")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("candidate rank must be positive")
        object.__setattr__(self, "retriever_score", _finite(self.retriever_score, "retriever_score"))
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        if len(self.metadata) > 2_000:
            raise ValueError("candidate metadata is too large")
        object.__setattr__(
            self,
            "metadata",
            {
                _identifier(key, "metadata key", 300): _identifier(value, "metadata value", 10_000)
                for key, value in self.metadata.items()
            },
        )


class NegativeRetriever(Protocol):
    @property
    def artifact_digest(self) -> str: ...

    def retrieve(self, query: str, *, top_k: int) -> Sequence[RetrievedCandidate]: ...


class TeacherScorer(Protocol):
    @property
    def artifact_digest(self) -> str: ...

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True)
class HardNegativeMiningConfig:
    retrieval_depth: int = 200
    negatives_per_query: int = 16
    skip_top_ranks: int = 0
    selection: str = "rank"
    minimum_teacher_score: float | None = None
    maximum_teacher_score: float | None = None
    semi_hard_margin: float = 0.0

    def __post_init__(self) -> None:
        for name, minimum in (("retrieval_depth", 1), ("negatives_per_query", 1), ("skip_top_ranks", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} is invalid")
        if self.negatives_per_query > self.retrieval_depth:
            raise ValueError("negatives_per_query may not exceed retrieval_depth")
        if self.selection not in {"rank", "teacher_hard", "teacher_semi_hard"}:
            raise ValueError("selection must be rank, teacher_hard, or teacher_semi_hard")
        for name in ("minimum_teacher_score", "maximum_teacher_score"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        if self.minimum_teacher_score is not None and self.maximum_teacher_score is not None:
            if self.minimum_teacher_score > self.maximum_teacher_score:
                raise ValueError("teacher-score interval is reversed")
        if _finite(self.semi_hard_margin, "semi_hard_margin") < 0.0:
            raise ValueError("semi_hard_margin must be non-negative")


@dataclass(frozen=True)
class HardNegativeRecord:
    query_id: str
    query: str
    positives: tuple[TrainingDocument, ...]
    negatives: tuple[TrainingDocument, ...]


@dataclass(frozen=True)
class HardNegativeGenerationManifest:
    generation_id: str
    source_dataset_manifest_digest: str
    source_split_digest: str
    retriever_artifact_digest: str
    teacher_artifact_digest: str | None
    config_digest: str
    record_count: int
    output_sha256: str
    rejected_known_positive_count: int
    rejected_duplicate_count: int
    insufficient_negative_query_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation_id", _identifier(self.generation_id, "generation_id"))
        for name in (
            "source_dataset_manifest_digest",
            "source_split_digest",
            "retriever_artifact_digest",
            "config_digest",
            "output_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name))
        if self.teacher_artifact_digest is not None:
            object.__setattr__(
                self,
                "teacher_artifact_digest",
                _sha256(self.teacher_artifact_digest, "teacher_artifact_digest"),
            )
        for name in (
            "record_count",
            "rejected_known_positive_count",
            "rejected_duplicate_count",
            "insufficient_negative_query_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def digest(self) -> str:
        return canonical_digest(asdict(self))


def _candidate_training_document(candidate: RetrievedCandidate, teacher_score: float | None) -> TrainingDocument:
    return TrainingDocument(
        document_id=candidate.document_id,
        text=candidate.text,
        relevance=0.0,
        source_id=candidate.source_id,
        metadata={**candidate.metadata, "retriever_rank": str(candidate.rank)},
        teacher_score=teacher_score,
    )


def mine_one(
    example: RetrievalTrainingExample,
    retriever: NegativeRetriever,
    *,
    config: HardNegativeMiningConfig,
    teacher: TeacherScorer | None = None,
) -> tuple[HardNegativeRecord, int, int, bool]:
    if config.selection != "rank" and teacher is None:
        raise ValueError("teacher-based hard-negative selection requires a teacher scorer")
    retrieved = list(retriever.retrieve(example.query, top_k=config.retrieval_depth))
    if len(retrieved) > config.retrieval_depth:
        raise ValueError("retriever returned more candidates than requested")
    if any(not isinstance(candidate, RetrievedCandidate) for candidate in retrieved):
        raise ValueError("retriever returned an invalid candidate type")
    retrieved.sort(key=lambda value: (value.rank, value.document_id))
    positive_ids = example.positive_document_ids
    filtered: list[RetrievedCandidate] = []
    seen: set[str] = set()
    rejected_positive = 0
    rejected_duplicate = 0
    for candidate in retrieved:
        if candidate.rank <= config.skip_top_ranks:
            continue
        if candidate.document_id in positive_ids:
            rejected_positive += 1
            continue
        if candidate.document_id in seen:
            rejected_duplicate += 1
            continue
        seen.add(candidate.document_id)
        filtered.append(candidate)

    teacher_scores: list[float | None] = [None] * len(filtered)
    positive_reference_score: float | None = None
    if teacher is not None and filtered:
        values = teacher.score(example.query, [candidate.text for candidate in filtered])
        if len(values) != len(filtered):
            raise ValueError("teacher score cardinality differs from candidate cardinality")
        teacher_scores = [_finite(value, "teacher score") for value in values]
        if config.selection == "teacher_semi_hard":
            positive_values = teacher.score(example.query, [document.text for document in example.positives])
            if len(positive_values) != len(example.positives):
                raise ValueError("teacher positive score cardinality mismatch")
            positive_reference_score = min(_finite(value, "positive teacher score") for value in positive_values)

    candidates: list[tuple[RetrievedCandidate, float | None]] = []
    for candidate, teacher_score in zip(filtered, teacher_scores):
        if teacher_score is not None:
            if config.minimum_teacher_score is not None and teacher_score < config.minimum_teacher_score:
                continue
            if config.maximum_teacher_score is not None and teacher_score > config.maximum_teacher_score:
                continue
            if config.selection == "teacher_semi_hard" and positive_reference_score is not None:
                if teacher_score > positive_reference_score + config.semi_hard_margin:
                    continue
        candidates.append((candidate, teacher_score))

    if config.selection == "teacher_hard":
        candidates.sort(key=lambda pair: (-(pair[1] if pair[1] is not None else float("-inf")), pair[0].rank, pair[0].document_id))
    elif config.selection == "teacher_semi_hard":
        candidates.sort(
            key=lambda pair: (
                abs((positive_reference_score or 0.0) - (pair[1] if pair[1] is not None else float("-inf"))),
                pair[0].rank,
                pair[0].document_id,
            )
        )
    else:
        candidates.sort(key=lambda pair: (pair[0].rank, pair[0].document_id))

    selected = candidates[: config.negatives_per_query]
    record = HardNegativeRecord(
        query_id=example.query_id,
        query=example.query,
        positives=example.positives,
        negatives=tuple(_candidate_training_document(candidate, score) for candidate, score in selected),
    )
    return record, rejected_positive, rejected_duplicate, len(selected) < config.negatives_per_query


def _record_json(record: HardNegativeRecord) -> dict[str, Any]:
    return {
        "query_id": record.query_id,
        "query": record.query,
        "positives": [asdict(value) for value in record.positives],
        "negatives": [asdict(value) for value in record.negatives],
    }


def _write_atomic_jsonl(path: Path, records: Sequence[HardNegativeRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                line = _canonical(_record_json(record)) + b"\n"
                digest.update(line)
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return digest.hexdigest()


def mine_generation(
    examples: Sequence[RetrievalTrainingExample],
    retriever: NegativeRetriever,
    *,
    output_path: str | Path,
    generation_id: str,
    source_dataset_manifest_digest: str,
    source_split_digest: str,
    config: HardNegativeMiningConfig = HardNegativeMiningConfig(),
    teacher: TeacherScorer | None = None,
) -> HardNegativeGenerationManifest:
    if not examples:
        raise ValueError("hard-negative mining requires at least one training example")
    retriever_digest = _sha256(retriever.artifact_digest, "retriever artifact digest")
    teacher_digest = None if teacher is None else _sha256(teacher.artifact_digest, "teacher artifact digest")
    records: list[HardNegativeRecord] = []
    rejected_positive = 0
    rejected_duplicate = 0
    insufficient = 0
    for example in examples:
        record, positive_count, duplicate_count, short = mine_one(
            example,
            retriever,
            config=config,
            teacher=teacher,
        )
        records.append(record)
        rejected_positive += positive_count
        rejected_duplicate += duplicate_count
        insufficient += int(short)
    destination = Path(output_path).expanduser().resolve()
    output_digest = _write_atomic_jsonl(destination, records)
    manifest = HardNegativeGenerationManifest(
        generation_id=generation_id,
        source_dataset_manifest_digest=source_dataset_manifest_digest,
        source_split_digest=source_split_digest,
        retriever_artifact_digest=retriever_digest,
        teacher_artifact_digest=teacher_digest,
        config_digest=canonical_digest(asdict(config)),
        record_count=len(records),
        output_sha256=output_digest,
        rejected_known_positive_count=rejected_positive,
        rejected_duplicate_count=rejected_duplicate,
        insufficient_negative_query_count=insufficient,
    )
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    payload = _canonical(asdict(manifest)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{manifest_path.name}-", suffix=".tmp", dir=manifest_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return manifest


__all__ = [
    "HardNegativeGenerationManifest",
    "HardNegativeMiningConfig",
    "HardNegativeRecord",
    "NegativeRetriever",
    "RetrievedCandidate",
    "TeacherScorer",
    "canonical_digest",
    "mine_generation",
    "mine_one",
]
