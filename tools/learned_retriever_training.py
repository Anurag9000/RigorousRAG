"""Framework-neutral training objectives and resumable plans for learned retrieval.

This module intentionally does *not* import a model framework or download weights.  It
owns the reproducible mathematics and state contracts that are otherwise easy to lose
inside a one-off training script:

* SPLADE/uniCOIL-style contrastive objectives with sparse FLOPS/L1 regularisation;
* ColBERT-style in-batch contrastive and optional teacher-distillation objectives;
* cross-encoder pairwise and listwise reranker objectives;
* hard-negative curriculum metadata;
* immutable training plans and stage boundaries; and
* content-addressed checkpoint manifests with strict resume compatibility checks.

A PyTorch/JAX/Transformers adapter can implement ``TrainingBackend`` and feed the scalar
loss terms back into its autograd graph.  The pure-Python helpers remain useful for
reference calculations, dry planning, audit reports and compatibility validation.
No dataset or model execution occurs merely by importing this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

_EPS = 1e-12
_MAX_BATCH = 65_536
_MAX_CANDIDATES = 65_536
_MAX_TERMS = 1_000_000
_MANIFEST_VERSION = 1


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive(value: Any, label: str, maximum: float = 1e9) -> float:
    result = _finite(value, label)
    if not 0.0 < result <= maximum:
        raise ValueError(f"{label} must be positive and bounded")
    return result


def _nonnegative(value: Any, label: str, maximum: float = 1e9) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= maximum:
        raise ValueError(f"{label} must be non-negative and bounded")
    return result


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or len(result) > maximum:
        raise ValueError(f"{label} is empty or too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in result):
        raise ValueError(f"{label} contains control characters")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 128).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON-compatible object."""

    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _softmax(logits: Sequence[float], temperature: float = 1.0) -> tuple[float, ...]:
    selected_temperature = _positive(temperature, "temperature", 1_000.0)
    if not logits or len(logits) > _MAX_CANDIDATES:
        raise ValueError("logits must be a non-empty bounded sequence")
    values = tuple(_finite(value, "logit") / selected_temperature for value in logits)
    normalizer = _logsumexp(values)
    return tuple(math.exp(value - normalizer) for value in values)


class RetrieverArchitecture(str, Enum):
    SPLADE = "splade"
    UNICOIL = "unicoil"
    COLBERT = "colbert"
    CROSS_ENCODER = "cross_encoder"
    LISTWISE_RERANKER = "listwise_reranker"
    DENSE_BIENCODER = "dense_biencoder"


class TrainingStageKind(str, Enum):
    WARMUP = "warmup"
    IN_BATCH = "in_batch"
    HARD_NEGATIVE = "hard_negative"
    DISTILLATION = "distillation"
    DOMAIN_ADAPTATION = "domain_adaptation"
    CALIBRATION = "calibration"


@dataclass(frozen=True)
class SparseRegularization:
    query_l1: float = 0.0
    document_l1: float = 0.0
    query_flops: float = 0.0
    document_flops: float = 0.0

    def __post_init__(self) -> None:
        for name in ("query_l1", "document_l1", "query_flops", "document_flops"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name, 1e6))


@dataclass(frozen=True)
class ObjectiveConfig:
    architecture: RetrieverArchitecture
    temperature: float = 1.0
    margin: float = 1.0
    sparse: SparseRegularization = field(default_factory=SparseRegularization)
    distillation_weight: float = 0.0
    teacher_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, RetrieverArchitecture):
            object.__setattr__(self, "architecture", RetrieverArchitecture(self.architecture))
        object.__setattr__(self, "temperature", _positive(self.temperature, "temperature", 1_000.0))
        object.__setattr__(self, "margin", _nonnegative(self.margin, "margin", 1e6))
        if not isinstance(self.sparse, SparseRegularization):
            raise ValueError("sparse must be SparseRegularization")
        weight = _nonnegative(self.distillation_weight, "distillation_weight", 1_000.0)
        object.__setattr__(self, "distillation_weight", weight)
        object.__setattr__(
            self,
            "teacher_temperature",
            _positive(self.teacher_temperature, "teacher_temperature", 1_000.0),
        )


@dataclass(frozen=True)
class LossBreakdown:
    total: float
    retrieval: float
    query_l1: float = 0.0
    document_l1: float = 0.0
    query_flops: float = 0.0
    document_flops: float = 0.0
    distillation: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "total",
            "retrieval",
            "query_l1",
            "document_l1",
            "query_flops",
            "document_flops",
            "distillation",
        ):
            value = _nonnegative(getattr(self, name), name, 1e30)
            object.__setattr__(self, name, value)


def in_batch_contrastive_loss(
    score_rows: Sequence[Sequence[Any]],
    positive_indices: Sequence[int],
    *,
    temperature: float = 1.0,
) -> float:
    """Mean InfoNCE/cross-entropy loss for query-by-candidate score rows."""

    if not score_rows or len(score_rows) > _MAX_BATCH:
        raise ValueError("score_rows must be a non-empty bounded batch")
    if len(positive_indices) != len(score_rows):
        raise ValueError("positive_indices must align with score_rows")
    selected_temperature = _positive(temperature, "temperature", 1_000.0)
    losses: list[float] = []
    for row_index, raw_row in enumerate(score_rows):
        if not raw_row or len(raw_row) > _MAX_CANDIDATES:
            raise ValueError("each score row must contain bounded candidates")
        row = [_finite(value, "score") / selected_temperature for value in raw_row]
        positive = positive_indices[row_index]
        if isinstance(positive, bool) or not isinstance(positive, int) or not 0 <= positive < len(row):
            raise ValueError("positive index is outside its score row")
        losses.append(_logsumexp(row) - row[positive])
    return sum(losses) / len(losses)


def pairwise_softplus_loss(
    positive_scores: Sequence[Any],
    negative_scores: Sequence[Any],
    *,
    margin: float = 0.0,
) -> float:
    """Stable pairwise logistic loss: softplus(margin - positive + negative)."""

    if not positive_scores or len(positive_scores) != len(negative_scores):
        raise ValueError("positive_scores and negative_scores must align")
    if len(positive_scores) > _MAX_BATCH:
        raise ValueError("pairwise batch is too large")
    selected_margin = _nonnegative(margin, "margin", 1e6)
    losses: list[float] = []
    for positive, negative in zip(positive_scores, negative_scores):
        value = selected_margin - _finite(positive, "positive score") + _finite(negative, "negative score")
        losses.append(max(value, 0.0) + math.log1p(math.exp(-abs(value))))
    return sum(losses) / len(losses)


def listwise_cross_entropy(
    logits: Sequence[Any],
    relevance: Sequence[Any],
    *,
    temperature: float = 1.0,
) -> float:
    """Cross entropy between model softmax and normalized graded relevance targets."""

    if not logits or len(logits) != len(relevance) or len(logits) > _MAX_CANDIDATES:
        raise ValueError("logits and relevance must be aligned bounded sequences")
    model = _softmax([_finite(value, "logit") for value in logits], temperature)
    labels = [_nonnegative(value, "relevance", 1e9) for value in relevance]
    total = sum(labels)
    if total <= 0.0:
        raise ValueError("at least one relevance value must be positive")
    target = [value / total for value in labels]
    return -sum(expected * math.log(max(observed, _EPS)) for expected, observed in zip(target, model))


def distillation_kl(
    student_logits: Sequence[Any],
    teacher_logits: Sequence[Any],
    *,
    temperature: float = 1.0,
) -> float:
    """Temperature-scaled KL(teacher || student), including the conventional T^2 factor."""

    if not student_logits or len(student_logits) != len(teacher_logits):
        raise ValueError("student and teacher logits must align")
    selected_temperature = _positive(temperature, "temperature", 1_000.0)
    student = _softmax([_finite(value, "student logit") for value in student_logits], selected_temperature)
    teacher = _softmax([_finite(value, "teacher logit") for value in teacher_logits], selected_temperature)
    divergence = sum(
        expected * (math.log(max(expected, _EPS)) - math.log(max(observed, _EPS)))
        for expected, observed in zip(teacher, student)
    )
    return divergence * selected_temperature * selected_temperature


def sparse_activation_penalties(batch_weights: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Return mean L1 activation and SPLADE FLOPS regularizer for sparse expansions.

    FLOPS follows the common SPLADE proxy: for each vocabulary term, square its mean
    activation over the batch and sum across terms.
    """

    if not batch_weights or len(batch_weights) > _MAX_BATCH:
        raise ValueError("batch_weights must be a non-empty bounded batch")
    term_sums: dict[str, float] = {}
    l1_total = 0.0
    for weights in batch_weights:
        if not isinstance(weights, Mapping) or len(weights) > _MAX_TERMS:
            raise ValueError("sparse expansion must be a bounded mapping")
        row_l1 = 0.0
        for term, raw_value in weights.items():
            _identifier(term, "term")
            value = _nonnegative(raw_value, "activation", 1e9)
            row_l1 += value
            term_sums[term] = term_sums.get(term, 0.0) + value
        l1_total += row_l1
    count = float(len(batch_weights))
    mean_l1 = l1_total / count
    flops = sum((total / count) ** 2 for total in term_sums.values())
    return mean_l1, flops


def sparse_retriever_loss(
    score_rows: Sequence[Sequence[Any]],
    positive_indices: Sequence[int],
    *,
    query_weights: Sequence[Mapping[str, Any]],
    document_weights: Sequence[Mapping[str, Any]],
    config: ObjectiveConfig,
    student_logits: Sequence[Any] | None = None,
    teacher_logits: Sequence[Any] | None = None,
) -> LossBreakdown:
    """Compose SPLADE/uniCOIL retrieval, sparsity and optional distillation losses."""

    if config.architecture not in {RetrieverArchitecture.SPLADE, RetrieverArchitecture.UNICOIL}:
        raise ValueError("sparse_retriever_loss requires a sparse architecture")
    retrieval = in_batch_contrastive_loss(score_rows, positive_indices, temperature=config.temperature)
    query_l1, query_flops = sparse_activation_penalties(query_weights)
    document_l1, document_flops = sparse_activation_penalties(document_weights)
    distillation = 0.0
    if student_logits is not None or teacher_logits is not None:
        if student_logits is None or teacher_logits is None:
            raise ValueError("both student_logits and teacher_logits are required for distillation")
        distillation = distillation_kl(
            student_logits,
            teacher_logits,
            temperature=config.teacher_temperature,
        )
    sparse = config.sparse
    total = (
        retrieval
        + sparse.query_l1 * query_l1
        + sparse.document_l1 * document_l1
        + sparse.query_flops * query_flops
        + sparse.document_flops * document_flops
        + config.distillation_weight * distillation
    )
    return LossBreakdown(
        total=total,
        retrieval=retrieval,
        query_l1=query_l1,
        document_l1=document_l1,
        query_flops=query_flops,
        document_flops=document_flops,
        distillation=distillation,
    )


def late_interaction_loss(
    score_rows: Sequence[Sequence[Any]],
    positive_indices: Sequence[int],
    *,
    config: ObjectiveConfig,
    student_logits: Sequence[Any] | None = None,
    teacher_logits: Sequence[Any] | None = None,
) -> LossBreakdown:
    """ColBERT/dense in-batch retrieval loss with optional teacher distillation."""

    if config.architecture not in {RetrieverArchitecture.COLBERT, RetrieverArchitecture.DENSE_BIENCODER}:
        raise ValueError("late_interaction_loss requires ColBERT or dense bi-encoder architecture")
    retrieval = in_batch_contrastive_loss(score_rows, positive_indices, temperature=config.temperature)
    distillation = 0.0
    if student_logits is not None or teacher_logits is not None:
        if student_logits is None or teacher_logits is None:
            raise ValueError("both student and teacher logits are required")
        distillation = distillation_kl(student_logits, teacher_logits, temperature=config.teacher_temperature)
    return LossBreakdown(
        total=retrieval + config.distillation_weight * distillation,
        retrieval=retrieval,
        distillation=distillation,
    )


@dataclass(frozen=True)
class HardNegativeCurriculum:
    warmup_steps: int = 0
    negatives_per_query: int = 8
    refresh_every_steps: int = 1_000
    teacher_mining: bool = False

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("warmup_steps", 0, 10**12),
            ("negatives_per_query", 1, 100_000),
            ("refresh_every_steps", 1, 10**12),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if not isinstance(self.teacher_mining, bool):
            raise ValueError("teacher_mining must be boolean")


@dataclass(frozen=True)
class TrainingStage:
    name: str
    kind: TrainingStageKind
    max_steps: int
    learning_rate: float
    checkpoint_every_steps: int
    objective: ObjectiveConfig
    hard_negatives: HardNegativeCurriculum | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "stage name", 120))
        if not isinstance(self.kind, TrainingStageKind):
            object.__setattr__(self, "kind", TrainingStageKind(self.kind))
        for name, minimum in (("max_steps", 1), ("checkpoint_every_steps", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > 10**12:
                raise ValueError(f"{name} is invalid")
        object.__setattr__(self, "learning_rate", _positive(self.learning_rate, "learning_rate", 100.0))
        if not isinstance(self.objective, ObjectiveConfig):
            raise ValueError("objective must be ObjectiveConfig")
        if self.hard_negatives is not None and not isinstance(self.hard_negatives, HardNegativeCurriculum):
            raise ValueError("hard_negatives must be HardNegativeCurriculum")


@dataclass(frozen=True)
class TrainingPlan:
    run_id: str
    architecture: RetrieverArchitecture
    base_model: str
    dataset_manifest_digest: str
    stages: tuple[TrainingStage, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        if not isinstance(self.architecture, RetrieverArchitecture):
            object.__setattr__(self, "architecture", RetrieverArchitecture(self.architecture))
        object.__setattr__(self, "base_model", _identifier(self.base_model, "base_model", 1_000))
        object.__setattr__(
            self,
            "dataset_manifest_digest",
            _digest(self.dataset_manifest_digest, "dataset_manifest_digest"),
        )
        if not self.stages or len(self.stages) > 100:
            raise ValueError("stages must be a non-empty bounded tuple")
        if any(stage.objective.architecture != self.architecture for stage in self.stages):
            raise ValueError("all stage objectives must match the plan architecture")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed <= 2**63 - 1:
            raise ValueError("seed must be a non-negative 63-bit integer")

    @property
    def config_digest(self) -> str:
        return sha256_json(asdict(self))


@dataclass(frozen=True)
class TrainingCursor:
    stage_index: int
    epoch: int
    global_step: int
    examples_seen: int
    tokens_seen: int = 0

    def __post_init__(self) -> None:
        for name in ("stage_index", "epoch", "global_step", "examples_seen", "tokens_seen"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > 10**18:
                raise ValueError(f"{name} must be a bounded non-negative integer")


@dataclass(frozen=True)
class CheckpointManifest:
    run_id: str
    architecture: RetrieverArchitecture
    base_model: str
    dataset_manifest_digest: str
    training_config_digest: str
    source_commit: str
    model_artifact_digest: str
    optimizer_artifact_digest: str
    scheduler_artifact_digest: str
    rng_state_digest: str
    data_cursor_digest: str
    cursor: TrainingCursor
    parent_checkpoint_digest: str | None = None
    version: int = _MANIFEST_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _identifier(self.run_id, "run_id"))
        if not isinstance(self.architecture, RetrieverArchitecture):
            object.__setattr__(self, "architecture", RetrieverArchitecture(self.architecture))
        object.__setattr__(self, "base_model", _identifier(self.base_model, "base_model", 1_000))
        for name in (
            "dataset_manifest_digest",
            "training_config_digest",
            "source_commit",
            "model_artifact_digest",
            "optimizer_artifact_digest",
            "scheduler_artifact_digest",
            "rng_state_digest",
            "data_cursor_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.parent_checkpoint_digest is not None:
            object.__setattr__(
                self,
                "parent_checkpoint_digest",
                _digest(self.parent_checkpoint_digest, "parent_checkpoint_digest"),
            )
        if not isinstance(self.cursor, TrainingCursor):
            raise ValueError("cursor must be TrainingCursor")
        if self.version != _MANIFEST_VERSION:
            raise ValueError(f"unsupported checkpoint manifest version {self.version}")

    def canonical_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["architecture"] = self.architecture.value
        return value

    @property
    def checkpoint_digest(self) -> str:
        return sha256_json(self.canonical_dict())

    def assert_resume_compatible(self, plan: TrainingPlan, source_commit: str) -> None:
        """Reject accidental resume across model, data, config, architecture or code changes."""

        if self.run_id != plan.run_id:
            raise ValueError("checkpoint run_id does not match training plan")
        if self.architecture != plan.architecture:
            raise ValueError("checkpoint architecture does not match training plan")
        if self.base_model != plan.base_model:
            raise ValueError("checkpoint base model does not match training plan")
        if self.dataset_manifest_digest != plan.dataset_manifest_digest:
            raise ValueError("checkpoint dataset manifest does not match training plan")
        if self.training_config_digest != plan.config_digest:
            raise ValueError("checkpoint training configuration does not match training plan")
        if self.source_commit != _digest(source_commit, "source_commit"):
            raise ValueError("checkpoint source commit does not match requested resume commit")
        if self.cursor.stage_index >= len(plan.stages):
            raise ValueError("checkpoint stage is outside the training plan")


class TrainingBackend(Protocol):
    """Framework adapter used by an execution script, not by repository import."""

    def train_step(self, batch: Any, *, stage: TrainingStage) -> Mapping[str, float]: ...

    def snapshot_artifacts(self) -> Mapping[str, str]: ...


class CheckpointSink(Protocol):
    def save(self, manifest: CheckpointManifest) -> str: ...


class JsonCheckpointSink:
    """Atomically persist checkpoint manifests; heavy tensor artifacts stay provider-owned."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, manifest: CheckpointManifest) -> str:
        if not isinstance(manifest, CheckpointManifest):
            raise ValueError("manifest must be CheckpointManifest")
        digest = manifest.checkpoint_digest
        destination = self.root / f"{digest}.json"
        payload = _canonical_json(manifest.canonical_dict()) + b"\n"
        if destination.exists():
            existing = destination.read_bytes()
            if existing != payload:
                raise RuntimeError("content-addressed checkpoint manifest collision")
            return digest
        descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
            try:
                directory_descriptor = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
            except OSError:
                pass
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return digest


def build_default_training_plan(
    *,
    run_id: str,
    architecture: RetrieverArchitecture,
    base_model: str,
    dataset_manifest_digest: str,
    seed: int = 0,
) -> TrainingPlan:
    """Build a conservative two-stage plan that must be explicitly executed elsewhere."""

    selected = RetrieverArchitecture(architecture)
    sparse = SparseRegularization(
        query_l1=1e-5 if selected in {RetrieverArchitecture.SPLADE, RetrieverArchitecture.UNICOIL} else 0.0,
        document_l1=1e-5 if selected in {RetrieverArchitecture.SPLADE, RetrieverArchitecture.UNICOIL} else 0.0,
        query_flops=1e-4 if selected == RetrieverArchitecture.SPLADE else 0.0,
        document_flops=1e-4 if selected == RetrieverArchitecture.SPLADE else 0.0,
    )
    objective = ObjectiveConfig(architecture=selected, temperature=0.05, sparse=sparse)
    stages = (
        TrainingStage(
            name="in_batch_warmup",
            kind=TrainingStageKind.IN_BATCH,
            max_steps=10_000,
            learning_rate=2e-5,
            checkpoint_every_steps=500,
            objective=objective,
        ),
        TrainingStage(
            name="hard_negative_finetune",
            kind=TrainingStageKind.HARD_NEGATIVE,
            max_steps=20_000,
            learning_rate=1e-5,
            checkpoint_every_steps=500,
            objective=objective,
            hard_negatives=HardNegativeCurriculum(
                warmup_steps=0,
                negatives_per_query=16,
                refresh_every_steps=2_000,
                teacher_mining=True,
            ),
        ),
    )
    return TrainingPlan(
        run_id=run_id,
        architecture=selected,
        base_model=base_model,
        dataset_manifest_digest=dataset_manifest_digest,
        stages=stages,
        seed=seed,
    )


__all__ = [
    "CheckpointManifest",
    "CheckpointSink",
    "HardNegativeCurriculum",
    "JsonCheckpointSink",
    "LossBreakdown",
    "ObjectiveConfig",
    "RetrieverArchitecture",
    "SparseRegularization",
    "TrainingBackend",
    "TrainingCursor",
    "TrainingPlan",
    "TrainingStage",
    "TrainingStageKind",
    "build_default_training_plan",
    "distillation_kl",
    "in_batch_contrastive_loss",
    "late_interaction_loss",
    "listwise_cross_entropy",
    "pairwise_softplus_loss",
    "sha256_json",
    "sparse_activation_penalties",
    "sparse_retriever_loss",
]
