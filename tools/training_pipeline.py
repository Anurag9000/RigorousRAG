"""Governed training-pipeline contracts for retrievers, rerankers and routers.

This module implements dataset records, leakage-resistant grouped splits, hard-negative
attachment, objective/config manifests, trainer/export protocols and immutable artifact
manifests.  It intentionally does not train or download a model.  Production trainers
plug into these contracts and must emit artifacts bound to dataset/config fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.capability_registry import CapabilityDescriptor, ResourceEnvelope

_MAX_TEXT = 100_000
_MAX_RECORDS = 2_000_000
_ALLOWED_TASKS = frozenset({"reranker", "dense_retriever", "sparse_retriever", "late_interaction", "router", "planner", "entailment"})
_ALLOWED_OBJECTIVES = frozenset({"pointwise", "pairwise", "listwise", "contrastive", "in_batch_contrastive", "distillation", "matryoshka", "classification", "regression"})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = value.replace("\x00", " ").strip()
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _finite(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} is below its minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} exceeds its maximum")
    return result


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    result = value.strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return result


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class TrainingExample:
    """Text-bearing governed example; never reconstructed from privacy hashes."""

    example_id: str
    group_id: str
    query: str
    positive: str
    negatives: tuple[str, ...] = ()
    teacher_scores: tuple[float, ...] = ()
    label: float | None = None
    source_dataset_id: str = ""
    source_record_sha256: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _text(self.example_id, "example_id", 256))
        object.__setattr__(self, "group_id", _text(self.group_id, "group_id", 256))
        object.__setattr__(self, "query", _text(self.query, "query", _MAX_TEXT))
        object.__setattr__(self, "positive", _text(self.positive, "positive", _MAX_TEXT))
        if len(self.negatives) > 1024:
            raise ValueError("negatives exceed the item limit")
        object.__setattr__(self, "negatives", tuple(_text(item, "negative", _MAX_TEXT) for item in self.negatives))
        if self.teacher_scores:
            expected = 1 + len(self.negatives)
            if len(self.teacher_scores) != expected:
                raise ValueError("teacher_scores must align with positive plus negatives")
            object.__setattr__(self, "teacher_scores", tuple(_finite(item, "teacher score") for item in self.teacher_scores))
        if self.label is not None:
            object.__setattr__(self, "label", _finite(self.label, "label"))
        if self.source_dataset_id:
            object.__setattr__(self, "source_dataset_id", _text(self.source_dataset_id, "source_dataset_id", 256))
        if self.source_record_sha256:
            object.__setattr__(self, "source_record_sha256", _digest(self.source_record_sha256, "source_record_sha256"))
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {_text(str(k), "metadata key", 100): _text(str(v), "metadata value", 1000) for k, v in self.metadata.items()})

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class TrainingDataset:
    dataset_id: str
    examples: tuple[TrainingExample, ...]
    license_id: str
    governance_sha256: str
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", _text(self.dataset_id, "dataset_id", 256))
        if not 1 <= len(self.examples) <= _MAX_RECORDS:
            raise ValueError("examples are empty or exceed the dataset limit")
        if any(not isinstance(item, TrainingExample) for item in self.examples):
            raise ValueError("examples must contain TrainingExample objects")
        ids = [item.example_id for item in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate training example IDs")
        object.__setattr__(self, "license_id", _text(self.license_id, "license_id", 256))
        object.__setattr__(self, "governance_sha256", _digest(self.governance_sha256, "governance_sha256"))
        object.__setattr__(self, "version", _text(self.version, "version", 64))

    @property
    def fingerprint(self) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "license_id": self.license_id,
            "governance_sha256": self.governance_sha256,
            "examples": [item.fingerprint for item in self.examples],
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class DatasetSplit:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    seed: int
    fingerprint: str


def grouped_split(
    dataset: TrainingDataset,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 17,
) -> DatasetSplit:
    validation = _finite(validation_fraction, "validation_fraction", minimum=0.0, maximum=0.49)
    test = _finite(test_fraction, "test_fraction", minimum=0.0, maximum=0.49)
    if validation + test >= 0.9:
        raise ValueError("validation and test fractions leave too little training data")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    groups: dict[str, list[str]] = {}
    for example in dataset.examples:
        groups.setdefault(example.group_id, []).append(example.example_id)
    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    group_count = len(keys)
    test_count = int(round(group_count * test))
    validation_count = int(round(group_count * validation))
    test_groups = set(keys[:test_count])
    validation_groups = set(keys[test_count : test_count + validation_count])
    train_groups = set(keys[test_count + validation_count :])
    if not train_groups:
        raise ValueError("grouped split produced no training groups")
    train_ids = tuple(example.example_id for example in dataset.examples if example.group_id in train_groups)
    validation_ids = tuple(example.example_id for example in dataset.examples if example.group_id in validation_groups)
    test_ids = tuple(example.example_id for example in dataset.examples if example.group_id in test_groups)
    payload = {"dataset": dataset.fingerprint, "train": train_ids, "validation": validation_ids, "test": test_ids, "seed": seed}
    return DatasetSplit(train_ids, validation_ids, test_ids, seed, hashlib.sha256(_canonical(payload)).hexdigest())


@dataclass(frozen=True)
class HardNegative:
    query_group_id: str
    text: str
    source_id: str
    rank: int
    score: float
    miner_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_group_id", _text(self.query_group_id, "query_group_id", 256))
        object.__setattr__(self, "text", _text(self.text, "negative text", _MAX_TEXT))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 500))
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or not 1 <= self.rank <= 100_000:
            raise ValueError("rank is invalid")
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "miner_id", _text(self.miner_id, "miner_id", 256))


def attach_hard_negatives(
    dataset: TrainingDataset,
    negatives: Sequence[HardNegative],
    *,
    max_per_example: int = 8,
) -> TrainingDataset:
    if isinstance(max_per_example, bool) or not isinstance(max_per_example, int) or not 1 <= max_per_example <= 128:
        raise ValueError("max_per_example is invalid")
    by_group: dict[str, list[HardNegative]] = {}
    for item in negatives:
        by_group.setdefault(item.query_group_id, []).append(item)
    updated: list[TrainingExample] = []
    for example in dataset.examples:
        candidates = sorted(by_group.get(example.group_id, ()), key=lambda item: (item.rank, -item.score, item.source_id))
        texts = list(example.negatives)
        seen = {example.positive.casefold(), *(item.casefold() for item in texts)}
        for item in candidates:
            if len(texts) >= max_per_example:
                break
            key = item.text.casefold()
            if key not in seen:
                seen.add(key)
                texts.append(item.text)
        updated.append(
            TrainingExample(
                example_id=example.example_id,
                group_id=example.group_id,
                query=example.query,
                positive=example.positive,
                negatives=tuple(texts),
                teacher_scores=(),
                label=example.label,
                source_dataset_id=example.source_dataset_id,
                source_record_sha256=example.source_record_sha256,
                metadata=example.metadata,
            )
        )
    return TrainingDataset(dataset.dataset_id, tuple(updated), dataset.license_id, dataset.governance_sha256, dataset.version)


@dataclass(frozen=True)
class ObjectiveConfig:
    name: str
    weight: float = 1.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _text(self.name, "objective name", 64).lower()
        if name not in _ALLOWED_OBJECTIVES:
            raise ValueError("unsupported objective")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "weight", _finite(self.weight, "objective weight", minimum=0.0, maximum=1000.0))
        if not isinstance(self.parameters, Mapping) or len(self.parameters) > 64:
            raise ValueError("parameters must be a bounded mapping")
        # Parameters are config metadata only; force JSON-safe canonical representation.
        try:
            _canonical(dict(self.parameters))
        except (TypeError, ValueError) as exc:
            raise ValueError("objective parameters must be strict JSON values") from exc


@dataclass(frozen=True)
class TrainingConfig:
    task: str
    base_model_id: str
    objectives: tuple[ObjectiveConfig, ...]
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    max_sequence_length: int
    gradient_accumulation: int = 1
    mixed_precision: str = "none"
    output_format: str = "safetensors"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = _text(self.task, "task", 64).lower()
        if task not in _ALLOWED_TASKS:
            raise ValueError("unsupported training task")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "base_model_id", _text(self.base_model_id, "base_model_id", 500))
        if not self.objectives or len(self.objectives) > 16 or any(not isinstance(item, ObjectiveConfig) for item in self.objectives):
            raise ValueError("objectives are invalid")
        for name, minimum, maximum in (("seed", -2**31, 2**31-1), ("epochs", 1, 100_000), ("batch_size", 1, 1_000_000), ("max_sequence_length", 1, 1_000_000), ("gradient_accumulation", 1, 1_000_000)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} is invalid")
        object.__setattr__(self, "learning_rate", _finite(self.learning_rate, "learning_rate", minimum=0.0, maximum=10.0))
        precision = _text(self.mixed_precision, "mixed_precision", 16).lower()
        if precision not in {"none", "fp16", "bf16"}:
            raise ValueError("mixed_precision is invalid")
        object.__setattr__(self, "mixed_precision", precision)
        output_format = _text(self.output_format, "output_format", 32).lower()
        if output_format not in {"safetensors", "onnx", "torchscript", "openvino", "custom"}:
            raise ValueError("output_format is invalid")
        object.__setattr__(self, "output_format", output_format)
        if not isinstance(self.extra, Mapping) or len(self.extra) > 128:
            raise ValueError("extra must be a bounded mapping")
        try:
            _canonical(dict(self.extra))
        except (TypeError, ValueError) as exc:
            raise ValueError("extra must contain strict JSON values") from exc

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class TrainingPlan:
    plan_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    config: TrainingConfig
    parent_artifact_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id", 256))
        object.__setattr__(self, "dataset_fingerprint", _digest(self.dataset_fingerprint, "dataset_fingerprint"))
        object.__setattr__(self, "split_fingerprint", _digest(self.split_fingerprint, "split_fingerprint"))
        if not isinstance(self.config, TrainingConfig):
            raise ValueError("config must be TrainingConfig")
        if self.parent_artifact_sha256:
            object.__setattr__(self, "parent_artifact_sha256", _digest(self.parent_artifact_sha256, "parent_artifact_sha256"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical({"plan_id": self.plan_id, "dataset": self.dataset_fingerprint, "split": self.split_fingerprint, "config": self.config.fingerprint, "parent": self.parent_artifact_sha256})).hexdigest()


@dataclass(frozen=True)
class ModelArtifactManifest:
    plan_fingerprint: str
    artifact_sha256: str
    model_id: str
    model_version: str
    task: str
    format: str
    size_bytes: int
    metrics_sha256: str = ""
    calibration_sha256: str = ""
    model_card_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("plan_fingerprint", "artifact_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("metrics_sha256", "calibration_sha256", "model_card_sha256"):
            value = getattr(self, name)
            if value:
                object.__setattr__(self, name, _digest(value, name))
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id", 500))
        object.__setattr__(self, "model_version", _text(self.model_version, "model_version", 100))
        task = _text(self.task, "task", 64).lower()
        if task not in _ALLOWED_TASKS:
            raise ValueError("unsupported artifact task")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "format", _text(self.format, "format", 32).lower())
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or not 0 <= self.size_bytes <= 10**15:
            raise ValueError("size_bytes is invalid")

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()

    def capability_descriptor(self, *, kind: str, provider: str, modalities: Sequence[str] = ("text",), permissions: Sequence[str] = (), trust_level: str = "local") -> CapabilityDescriptor:
        return CapabilityDescriptor(
            capability_id=self.model_id,
            version=self.model_version,
            kind=kind,
            provider=provider,
            modalities=tuple(modalities),
            permissions=tuple(permissions),
            trust_level=trust_level,
            resources=ResourceEnvelope(),
            artifact_sha256=self.artifact_sha256,
        )


class TrainerBackend(Protocol):
    @property
    def trainer_id(self) -> str: ...
    def prepare(self, plan: TrainingPlan, dataset: TrainingDataset, split: DatasetSplit) -> Mapping[str, Any]: ...
    def train(self, prepared: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def export(self, trained: Mapping[str, Any], *, output_format: str) -> ModelArtifactManifest: ...


class ArtifactWriter(Protocol):
    def write_bytes(self, name: str, payload: bytes, *, metadata: Mapping[str, str]) -> str: ...


def model_card_payload(plan: TrainingPlan, dataset: TrainingDataset, split: DatasetSplit, manifest: ModelArtifactManifest | None = None) -> Mapping[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "plan_fingerprint": plan.fingerprint,
        "task": plan.config.task,
        "base_model_id": plan.config.base_model_id,
        "dataset_id": dataset.dataset_id,
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_license": dataset.license_id,
        "governance_sha256": dataset.governance_sha256,
        "split_fingerprint": split.fingerprint,
        "train_examples": len(split.train_ids),
        "validation_examples": len(split.validation_ids),
        "test_examples": len(split.test_ids),
        "artifact": asdict(manifest) if manifest is not None else None,
        "non_claims": [
            "This manifest does not establish model quality or safety.",
            "Training data provenance and licensing remain operator-governed.",
            "Promotion requires separate evaluation and policy approval.",
        ],
    }


__all__ = [
    "ArtifactWriter",
    "DatasetSplit",
    "HardNegative",
    "ModelArtifactManifest",
    "ObjectiveConfig",
    "TrainerBackend",
    "TrainingConfig",
    "TrainingDataset",
    "TrainingExample",
    "TrainingPlan",
    "attach_hard_negatives",
    "grouped_split",
    "model_card_payload",
]
