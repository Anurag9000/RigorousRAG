"""Authoritative ready-to-train composition for advanced RAG.

This is the launch path used by the CLI/config layer. Older modules remain reusable research
primitives; configuration-driven training always uses padding-safe supervision, exact input
identities, strict off-policy learning and an explicit causal/seq2seq generator family.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.advanced_rag_data import ManifestBoundAdvancedJsonlDataset
from training.advanced_rag_evaluators import DynamicPolicyValidationEvaluator, GroundedValidationEvaluator, ValidationLimits
from training.advanced_rag_final_collation import FinalCausalGroundedCollator, FinalDynamicRagEpisodeCollator, FinalSeq2SeqGroundedCollator
from training.advanced_rag_final_objectives import AuthoritativeGroundedGenerationStep
from training.advanced_rag_identity import AdvancedTrainingInputIdentity, dataclass_sha256, provider_identity_sha256, trainability_sha256
from training.advanced_rag_models import DynamicRagPolicyModel, GroundedGeneratorTrainingModule
from training.advanced_rag_runner import (
    AdvancedTrainingRunResult,
    DynamicRagPolicyTrainingRunner,
    GroundedGeneratorTrainingRunner,
    _TrainabilityStep,
    _effective_trainability,
    _loader,
    _trainer_with_evaluation,
)
from training.advanced_rag_steps import DynamicPolicyStepConfig, GroundedStepConfig, dynamic_plan_to_trainer_config, grounded_plan_to_trainer_config
from training.advanced_rag_strict import StrictDynamicRetrievalPolicyStep
from training.checkpointing import CheckpointManager
from training.data_pipeline import ResumableDeterministicSampler
from training.seq2seq_grounded import Seq2SeqGroundedGeneratorTrainingModule
from training.torch_engine import StageRuntime, TorchTrainingEngine


@dataclass(frozen=True)
class GroundedGeneratorPathBinding:
    generator_family: str
    collator: Any

    def __post_init__(self) -> None:
        if self.generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("generator_family must be causal_lm or seq2seq_lm")


class AuthoritativeGroundedGeneratorTrainingRunner(GroundedGeneratorTrainingRunner):
    """Final grounded runner supporting causal and encoder-decoder generators."""
    def __init__(self, *args: Any, generator_family: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("generator_family must be causal_lm or seq2seq_lm")
        self.generator_family = generator_family

    def _authoritative_collator(self) -> Any:
        collator = FinalSeq2SeqGroundedCollator if self.generator_family == "seq2seq_lm" else FinalCausalGroundedCollator
        return collator(
            self.tokenizer,
            self.collator_config,
            teacher_cache=self.teacher_cache,
            reference_cache=self.reference_cache,
            retriever_batch_builder=self.retriever_batch_builder,
        )

    def _authoritative_model(self) -> Any:
        if self.generator_family == "seq2seq_lm":
            return Seq2SeqGroundedGeneratorTrainingModule(base_model=self.base_model, config=self.plan.architecture, retriever_model=self.retriever_model)
        return GroundedGeneratorTrainingModule(base_model=self.base_model, config=self.plan.architecture, retriever_model=self.retriever_model)

    def run(self, *, resume_checkpoint_digest: str | None = None, event_sink: Any | None = None) -> AdvancedTrainingRunResult:
        train_dataset = ManifestBoundAdvancedJsonlDataset(self.train_split.path, expected_sha256=self.train_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.train_split.split_name, record_kind="grounded_generation", expected_record_count=self.train_split.expected_record_count)
        validation_dataset = ManifestBoundAdvancedJsonlDataset(self.validation_split.path, expected_sha256=self.validation_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.validation_split.split_name, record_kind="grounded_generation", expected_record_count=self.validation_split.expected_record_count)
        self._preflight(train_dataset); self._preflight(validation_dataset)
        effective_trainability = _effective_trainability(self.plan.stages, self.trainability)
        path_binding = GroundedGeneratorPathBinding(self.generator_family, self.collator_config)
        input_identity = AdvancedTrainingInputIdentity(
            kind="grounded_generation",
            plan_sha256=self.plan.plan_sha256,
            training_split_sha256=train_dataset.binding.content_sha256,
            validation_split_sha256=validation_dataset.binding.content_sha256,
            tokenizer_sha256=self.plan.tokenizer_sha256,
            execution_config_sha256=dataclass_sha256(self.execution, label="advanced-execution-config"),
            collator_config_sha256=dataclass_sha256(path_binding, label="grounded-generator-path-binding"),
            trainability_sha256=trainability_sha256(effective_trainability),
            teacher_cache_sha256=provider_identity_sha256(self.teacher_cache, label="teacher cache"),
            reference_cache_sha256=provider_identity_sha256(self.reference_cache, label="reference cache"),
            retriever_supervision_sha256=provider_identity_sha256(self.retriever_batch_builder, label="retriever supervision"),
        )
        model = self._authoritative_model()
        trainer_config = grounded_plan_to_trainer_config(self.plan, device=self.execution.device, precision=self.execution.precision, gradient_accumulation_steps=self.execution.gradient_accumulation_steps, max_grad_norm=self.execution.max_grad_norm, seed=self.execution.seed, deterministic_algorithms=self.execution.deterministic_algorithms, ddp=self.execution.ddp, weight_decay=self.execution.weight_decay, scheduler=self.execution.scheduler, warmup_steps=self.execution.warmup_steps)
        trainer_config = _trainer_with_evaluation(trainer_config, self.execution, input_identity)
        runtimes, validation_steps = [], []
        for stage_index, stage in enumerate(self.plan.stages):
            sampler = ResumableDeterministicSampler(len(train_dataset), seed=self.execution.seed + stage_index, shuffle=True)
            collator = self._authoritative_collator()
            step = AuthoritativeGroundedGenerationStep(GroundedStepConfig(stage.objective))
            validation_steps.append(AuthoritativeGroundedGenerationStep(GroundedStepConfig(stage.objective)))
            runtimes.append(StageRuntime(
                dataloader=_loader(train_dataset, sampler, collator, batch_size=self.execution.train_batch_size, execution=self.execution),
                step=_TrainabilityStep(step, effective_trainability[stage.name]),
                sampler=sampler,
                collator=collator,
            ))
        validation_loader = _loader(validation_dataset, None, self._authoritative_collator(), batch_size=self.execution.validation_batch_size, execution=self.execution)
        evaluator = GroundedValidationEvaluator(validation_loader, validation_steps, ValidationLimits(self.execution.validation_maximum_batches))
        engine = TorchTrainingEngine(model, trainer_config, CheckpointManager(self.checkpoint_root))
        summary = engine.fit(runtimes, resume_checkpoint_digest=resume_checkpoint_digest, evaluator=evaluator, event_sink=event_sink)
        return AdvancedTrainingRunResult(summary, self.plan.plan_sha256, input_identity.input_sha256, train_dataset.binding.content_sha256, validation_dataset.binding.content_sha256, str(Path(self.checkpoint_root).expanduser().resolve()))


class AuthoritativeDynamicRagPolicyTrainingRunner(DynamicRagPolicyTrainingRunner):
    """Final dynamic runner with variable-length hidden-state alignment and strict off-policy loss."""
    def run(self, *, resume_checkpoint_digest: str | None = None, event_sink: Any | None = None) -> AdvancedTrainingRunResult:
        train_dataset = ManifestBoundAdvancedJsonlDataset(self.train_split.path, expected_sha256=self.train_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.train_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=self.train_split.expected_record_count)
        validation_dataset = ManifestBoundAdvancedJsonlDataset(self.validation_split.path, expected_sha256=self.validation_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.validation_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=self.validation_split.expected_record_count)
        self._preflight(train_dataset); self._preflight(validation_dataset)
        effective_trainability = _effective_trainability(self.plan.stages, self.trainability)
        input_identity = AdvancedTrainingInputIdentity(
            kind="dynamic_rag_policy",
            plan_sha256=self.plan.plan_sha256,
            training_split_sha256=train_dataset.binding.content_sha256,
            validation_split_sha256=validation_dataset.binding.content_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            execution_config_sha256=dataclass_sha256(self.execution, label="advanced-execution-config"),
            collator_config_sha256=dataclass_sha256(self.collator_config, label="dynamic-collator-config"),
            trainability_sha256=trainability_sha256(effective_trainability),
            hidden_state_cache_sha256=provider_identity_sha256(self.hidden_state_cache, label="hidden-state cache"),
        )
        model = DynamicRagPolicyModel(self.plan.architecture)
        trainer_config = dynamic_plan_to_trainer_config(self.plan, device=self.execution.device, precision=self.execution.precision, gradient_accumulation_steps=self.execution.gradient_accumulation_steps, max_grad_norm=self.execution.max_grad_norm, seed=self.execution.seed, deterministic_algorithms=self.execution.deterministic_algorithms, ddp=self.execution.ddp, weight_decay=self.execution.weight_decay, scheduler=self.execution.scheduler, warmup_steps=self.execution.warmup_steps)
        trainer_config = _trainer_with_evaluation(trainer_config, self.execution, input_identity)
        runtimes, validation_steps = [], []
        for stage_index, stage in enumerate(self.plan.stages):
            sampler = ResumableDeterministicSampler(len(train_dataset), seed=self.execution.seed + stage_index, shuffle=True)
            collator = FinalDynamicRagEpisodeCollator(self.tokenizer, self.plan.architecture, self.collator_config, hidden_state_cache=self.hidden_state_cache)
            step = StrictDynamicRetrievalPolicyStep(DynamicPolicyStepConfig(stage.objective), actions=self.plan.architecture.actions)
            validation_steps.append(StrictDynamicRetrievalPolicyStep(DynamicPolicyStepConfig(stage.objective), actions=self.plan.architecture.actions))
            runtimes.append(StageRuntime(
                dataloader=_loader(train_dataset, sampler, collator, batch_size=self.execution.train_batch_size, execution=self.execution),
                step=_TrainabilityStep(step, effective_trainability[stage.name]),
                sampler=sampler,
                collator=collator,
            ))
        validation_collator = FinalDynamicRagEpisodeCollator(self.tokenizer, self.plan.architecture, self.collator_config, hidden_state_cache=self.hidden_state_cache)
        validation_loader = _loader(validation_dataset, None, validation_collator, batch_size=self.execution.validation_batch_size, execution=self.execution)
        evaluator = DynamicPolicyValidationEvaluator(validation_loader, validation_steps, ValidationLimits(self.execution.validation_maximum_batches))
        engine = TorchTrainingEngine(model, trainer_config, CheckpointManager(self.checkpoint_root))
        summary = engine.fit(runtimes, resume_checkpoint_digest=resume_checkpoint_digest, evaluator=evaluator, event_sink=event_sink)
        return AdvancedTrainingRunResult(summary, self.plan.plan_sha256, input_identity.input_sha256, train_dataset.binding.content_sha256, validation_dataset.binding.content_sha256, str(Path(self.checkpoint_root).expanduser().resolve()))


__all__ = ["AuthoritativeDynamicRagPolicyTrainingRunner", "AuthoritativeGroundedGeneratorTrainingRunner", "GroundedGeneratorPathBinding"]
