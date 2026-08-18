"""Turnkey source-only training composition for grounded generation and dynamic RAG.

Callers supply local dataset artifacts and already-admitted model/tokenizer objects. This
module binds datasets, deterministic samplers, strict collators, stage objectives, evaluator,
checkpoint manager and exact-resume trainer. It performs no work merely on import.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from training.advanced_rag_data import (
    DynamicCollatorConfig,
    GroundedCollatorConfig,
    ManifestBoundAdvancedJsonlDataset,
    RetrieverBatchBuilder,
    TensorCacheProvider,
)
from training.advanced_rag_evaluators import DynamicPolicyValidationEvaluator, GroundedValidationEvaluator, ValidationLimits
from training.advanced_rag_models import DynamicRagPolicyModel, GroundedGeneratorTrainingModule
from training.advanced_rag_steps import (
    DynamicPolicyStepConfig,
    GroundedGenerationStep,
    GroundedStepConfig,
    dynamic_plan_to_trainer_config,
    grounded_plan_to_trainer_config,
)
from training.advanced_rag_strict import StrictDynamicRagEpisodeCollator, StrictDynamicRetrievalPolicyStep
from training.checkpointing import CheckpointManager
from training.data_pipeline import ResumableDeterministicSampler
from training.dynamic_retrieval_policy import DynamicPolicyTrainingPlan
from training.grounded_generation import GroundedTrainingPlan
from training.grounded_supervision_pipeline import CompleteGroundedGenerationCollator
from training.torch_engine import StageRuntime, TorchTrainingEngine, TrainerConfig, TrainingSummary


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise RuntimeError("advanced RAG training runners require optional PyTorch")


@dataclass(frozen=True)
class LocalTrainingSplit:
    path: str
    content_sha256: str
    split_name: str
    expected_record_count: int | None = None

    def __post_init__(self) -> None:
        selected = Path(self.path).expanduser().resolve(strict=True)
        if not selected.is_file() or selected.is_symlink():
            raise ValueError("training split path must be a regular non-symlink file")
        object.__setattr__(self, "path", str(selected))
        digest = self.content_sha256.strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("training split content_sha256 must be SHA-256")
        object.__setattr__(self, "content_sha256", digest)
        if not isinstance(self.split_name, str) or not self.split_name.strip():
            raise ValueError("split_name is required")
        if self.expected_record_count is not None and (isinstance(self.expected_record_count, bool) or not isinstance(self.expected_record_count, int) or self.expected_record_count < 0):
            raise ValueError("expected_record_count must be non-negative or None")


@dataclass(frozen=True)
class TrainingExecutionConfig:
    train_batch_size: int = 4
    validation_batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "auto"
    precision: str = "fp32"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float | None = 1.0
    seed: int = 0
    deterministic_algorithms: bool = False
    ddp: bool = False
    weight_decay: float = 0.01
    scheduler: str = "linear"
    warmup_steps: int = 0
    evaluate_every_steps: int = 500
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 0.0
    validation_maximum_batches: int | None = None

    def __post_init__(self) -> None:
        for name in ("train_batch_size", "validation_batch_size", "evaluate_every_steps", "gradient_accumulation_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.scheduler not in {"constant", "linear", "cosine"}:
            raise ValueError("scheduler must be constant, linear, or cosine")


@dataclass(frozen=True)
class ParameterTrainabilityPolicy:
    """Stage-local parameter policy. Empty prefixes mean train every parameter.

    Optimizer/scheduler state intentionally continues across stages in the generic engine.
    Per-stage learning rates are still applied by ``TrainingStageSpec``; newly unfrozen
    parameters enter the already-bound optimizer with no historical moment state while
    previously trained parameters retain their optimizer state. This policy is explicit and
    deterministic across checkpoint/resume.
    """
    trainable_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        prefixes = tuple(str(value).strip() for value in self.trainable_prefixes)
        if any(not value for value in prefixes) or len(set(prefixes)) != len(prefixes):
            raise ValueError("trainable prefixes must be unique and non-empty")
        object.__setattr__(self, "trainable_prefixes", prefixes)

    def apply(self, model: Any) -> int:
        module = model.module if hasattr(model, "module") else model
        matched = 0
        for name, parameter in module.named_parameters():
            trainable = not self.trainable_prefixes or any(name == prefix or name.startswith(prefix + ".") for prefix in self.trainable_prefixes)
            parameter.requires_grad_(trainable)
            if trainable:
                matched += int(parameter.numel())
        if matched <= 0:
            raise ValueError("stage trainability policy matched no parameters")
        return matched


class _TrainabilityStep:
    def __init__(self, inner: Any, policy: ParameterTrainabilityPolicy) -> None:
        self.inner, self.policy, self._applied = inner, policy, False

    def __call__(self, model: Any, batch: Mapping[str, Any]) -> Any:
        if not self._applied:
            self.policy.apply(model)
            self._applied = True
        return self.inner(model, batch)


def _trainer_with_evaluation(config: TrainerConfig, execution: TrainingExecutionConfig) -> TrainerConfig:
    stages = tuple(replace(stage, evaluate_every_steps=min(execution.evaluate_every_steps, stage.max_optimizer_steps)) for stage in config.stages)
    return replace(
        config,
        stages=stages,
        early_stopping_metric="validation_primary",
        early_stopping_mode="max",
        early_stopping_patience=execution.early_stopping_patience,
        early_stopping_min_delta=execution.early_stopping_min_delta,
    )


def _loader(dataset: Any, sampler: Any, collator: Any, *, batch_size: int, execution: TrainingExecutionConfig) -> Any:
    _require_torch()
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=execution.num_workers, pin_memory=execution.pin_memory, drop_last=False)


@dataclass(frozen=True)
class AdvancedTrainingRunResult:
    summary: TrainingSummary
    plan_sha256: str
    training_data_sha256: str
    validation_data_sha256: str
    checkpoint_root: str


class GroundedGeneratorTrainingRunner:
    def __init__(self, *, plan: GroundedTrainingPlan, base_model: Any, tokenizer: Any, train_split: LocalTrainingSplit, validation_split: LocalTrainingSplit, checkpoint_root: str | Path, execution: TrainingExecutionConfig = TrainingExecutionConfig(), collator_config: GroundedCollatorConfig = GroundedCollatorConfig(), retriever_model: Any | None = None, teacher_cache: TensorCacheProvider | None = None, reference_cache: TensorCacheProvider | None = None, retriever_batch_builder: RetrieverBatchBuilder | None = None, trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None) -> None:
        _require_torch()
        if not isinstance(plan, GroundedTrainingPlan):
            raise ValueError("plan must be GroundedTrainingPlan")
        self.plan, self.base_model, self.tokenizer = plan, base_model, tokenizer
        self.train_split, self.validation_split = train_split, validation_split
        self.checkpoint_root, self.execution, self.collator_config = checkpoint_root, execution, collator_config
        self.retriever_model, self.teacher_cache, self.reference_cache, self.retriever_batch_builder = retriever_model, teacher_cache, reference_cache, retriever_batch_builder
        self.trainability = dict(trainability or {})

    def _collator(self) -> CompleteGroundedGenerationCollator:
        return CompleteGroundedGenerationCollator(
            self.tokenizer,
            self.collator_config,
            teacher_cache=self.teacher_cache,
            reference_cache=self.reference_cache,
            retriever_batch_builder=self.retriever_batch_builder,
        )

    def run(self, *, resume_checkpoint_digest: str | None = None, event_sink: Any | None = None) -> AdvancedTrainingRunResult:
        train_dataset = ManifestBoundAdvancedJsonlDataset(self.train_split.path, expected_sha256=self.train_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.train_split.split_name, record_kind="grounded_generation", expected_record_count=self.train_split.expected_record_count)
        validation_dataset = ManifestBoundAdvancedJsonlDataset(self.validation_split.path, expected_sha256=self.validation_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.validation_split.split_name, record_kind="grounded_generation", expected_record_count=self.validation_split.expected_record_count)
        model = GroundedGeneratorTrainingModule(base_model=self.base_model, config=self.plan.architecture, retriever_model=self.retriever_model)
        trainer_config = grounded_plan_to_trainer_config(self.plan, device=self.execution.device, precision=self.execution.precision, gradient_accumulation_steps=self.execution.gradient_accumulation_steps, max_grad_norm=self.execution.max_grad_norm, seed=self.execution.seed, deterministic_algorithms=self.execution.deterministic_algorithms, ddp=self.execution.ddp, weight_decay=self.execution.weight_decay, scheduler=self.execution.scheduler, warmup_steps=self.execution.warmup_steps)
        trainer_config = _trainer_with_evaluation(trainer_config, self.execution)
        runtimes, validation_steps = [], []
        for stage_index, stage in enumerate(self.plan.stages):
            sampler = ResumableDeterministicSampler(len(train_dataset), seed=self.execution.seed + stage_index, shuffle=True)
            collator = self._collator()
            step = GroundedGenerationStep(GroundedStepConfig(stage.objective))
            validation_steps.append(GroundedGenerationStep(GroundedStepConfig(stage.objective)))
            wrapped = _TrainabilityStep(step, self.trainability.get(stage.name, ParameterTrainabilityPolicy()))
            runtimes.append(StageRuntime(dataloader=_loader(train_dataset, sampler, collator, batch_size=self.execution.train_batch_size, execution=self.execution), step=wrapped, sampler=sampler, collator=collator))
        validation_collator = self._collator()
        validation_loader = _loader(validation_dataset, None, validation_collator, batch_size=self.execution.validation_batch_size, execution=self.execution)
        evaluator = GroundedValidationEvaluator(validation_loader, validation_steps, ValidationLimits(self.execution.validation_maximum_batches))
        engine = TorchTrainingEngine(model, trainer_config, CheckpointManager(self.checkpoint_root))
        summary = engine.fit(runtimes, resume_checkpoint_digest=resume_checkpoint_digest, evaluator=evaluator, event_sink=event_sink)
        return AdvancedTrainingRunResult(summary, self.plan.plan_sha256, train_dataset.binding.content_sha256, validation_dataset.binding.content_sha256, str(Path(self.checkpoint_root).expanduser().resolve()))


class DynamicRagPolicyTrainingRunner:
    def __init__(self, *, plan: DynamicPolicyTrainingPlan, tokenizer: Any, train_split: LocalTrainingSplit, validation_split: LocalTrainingSplit, checkpoint_root: str | Path, execution: TrainingExecutionConfig = TrainingExecutionConfig(), collator_config: DynamicCollatorConfig = DynamicCollatorConfig(), hidden_state_cache: TensorCacheProvider | None = None, trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None) -> None:
        _require_torch()
        if not isinstance(plan, DynamicPolicyTrainingPlan):
            raise ValueError("plan must be DynamicPolicyTrainingPlan")
        self.plan, self.tokenizer = plan, tokenizer
        self.train_split, self.validation_split = train_split, validation_split
        self.checkpoint_root, self.execution, self.collator_config = checkpoint_root, execution, collator_config
        self.hidden_state_cache, self.trainability = hidden_state_cache, dict(trainability or {})

    def run(self, *, resume_checkpoint_digest: str | None = None, event_sink: Any | None = None) -> AdvancedTrainingRunResult:
        train_dataset = ManifestBoundAdvancedJsonlDataset(self.train_split.path, expected_sha256=self.train_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.train_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=self.train_split.expected_record_count)
        validation_dataset = ManifestBoundAdvancedJsonlDataset(self.validation_split.path, expected_sha256=self.validation_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.validation_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=self.validation_split.expected_record_count)
        model = DynamicRagPolicyModel(self.plan.architecture)
        trainer_config = dynamic_plan_to_trainer_config(self.plan, device=self.execution.device, precision=self.execution.precision, gradient_accumulation_steps=self.execution.gradient_accumulation_steps, max_grad_norm=self.execution.max_grad_norm, seed=self.execution.seed, deterministic_algorithms=self.execution.deterministic_algorithms, ddp=self.execution.ddp, weight_decay=self.execution.weight_decay, scheduler=self.execution.scheduler, warmup_steps=self.execution.warmup_steps)
        trainer_config = _trainer_with_evaluation(trainer_config, self.execution)
        runtimes, validation_steps = [], []
        for stage_index, stage in enumerate(self.plan.stages):
            sampler = ResumableDeterministicSampler(len(train_dataset), seed=self.execution.seed + stage_index, shuffle=True)
            collator = StrictDynamicRagEpisodeCollator(self.tokenizer, self.plan.architecture, self.collator_config, hidden_state_cache=self.hidden_state_cache)
            step = StrictDynamicRetrievalPolicyStep(DynamicPolicyStepConfig(stage.objective), actions=self.plan.architecture.actions)
            validation_steps.append(StrictDynamicRetrievalPolicyStep(DynamicPolicyStepConfig(stage.objective), actions=self.plan.architecture.actions))
            wrapped = _TrainabilityStep(step, self.trainability.get(stage.name, ParameterTrainabilityPolicy()))
            runtimes.append(StageRuntime(dataloader=_loader(train_dataset, sampler, collator, batch_size=self.execution.train_batch_size, execution=self.execution), step=wrapped, sampler=sampler, collator=collator))
        validation_collator = StrictDynamicRagEpisodeCollator(self.tokenizer, self.plan.architecture, self.collator_config, hidden_state_cache=self.hidden_state_cache)
        validation_loader = _loader(validation_dataset, None, validation_collator, batch_size=self.execution.validation_batch_size, execution=self.execution)
        evaluator = DynamicPolicyValidationEvaluator(validation_loader, validation_steps, ValidationLimits(self.execution.validation_maximum_batches))
        engine = TorchTrainingEngine(model, trainer_config, CheckpointManager(self.checkpoint_root))
        summary = engine.fit(runtimes, resume_checkpoint_digest=resume_checkpoint_digest, evaluator=evaluator, event_sink=event_sink)
        return AdvancedTrainingRunResult(summary, self.plan.plan_sha256, train_dataset.binding.content_sha256, validation_dataset.binding.content_sha256, str(Path(self.checkpoint_root).expanduser().resolve()))


__all__ = ["AdvancedTrainingRunResult", "DynamicRagPolicyTrainingRunner", "GroundedGeneratorTrainingRunner", "LocalTrainingSplit", "ParameterTrainabilityPolicy", "TrainingExecutionConfig"]
