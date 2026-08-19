"""Authoritative ready-to-train composition for advanced RAG.

Configuration-driven and direct-library training use the same strict path/tokenizer/cache
authority, exact input identities, multi-evidence/contested grounding supervision, and
legal-action-masked dynamic policy objectives while preserving older research primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from training.advanced_checkpoint_authority import AdvancedCheckpointManager
from training.advanced_rag_action_legality import LegalActionDynamicRagEpisodeCollator, LegalActionDynamicRetrievalPolicyStep
from training.advanced_rag_authoritative_data import ManifestBoundAuthoritativeJsonlDataset
from training.advanced_rag_evaluators import DynamicPolicyValidationEvaluator, GroundedValidationEvaluator, ValidationLimits
from training.advanced_rag_final_objectives import AuthoritativeGroundedGenerationStep
from training.advanced_rag_identity import AdvancedTrainingInputIdentity, dataclass_sha256, provider_identity_sha256, trainability_sha256
from training.advanced_rag_models import DynamicRagPolicyModel, GroundedGeneratorTrainingModule
from training.advanced_rag_multi_evidence import MultiEvidenceCausalGroundedCollator, MultiEvidenceSeq2SeqGroundedCollator
from training.advanced_rag_runner import AdvancedTrainingRunResult, DynamicRagPolicyTrainingRunner, GroundedGeneratorTrainingRunner, _TrainabilityStep, _effective_trainability, _loader, _trainer_with_evaluation
from training.advanced_rag_steps import DynamicPolicyStepConfig, GroundedStepConfig, dynamic_plan_to_trainer_config, grounded_plan_to_trainer_config
from training.advanced_tokenizer_contract import assert_advanced_training_tokenizer
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


def _assert_cache_binding(
    cache: Any | None,
    *,
    label: str,
    expected_kind: str,
    dataset_manifest_sha256: str,
    tokenizer_sha256: str,
    source_commit: str,
    producer_sha256: str | None = None,
) -> None:
    if cache is None:
        return
    identity = getattr(cache, "identity", None)
    if identity is None:
        raise ValueError(f"{label} must expose immutable cache identity")
    if getattr(identity, "cache_kind", None) != expected_kind:
        raise ValueError(f"{label} must use cache_kind={expected_kind}")
    if getattr(identity, "dataset_manifest_sha256", None) != dataset_manifest_sha256:
        raise ValueError(f"{label} dataset manifest differs from training plan")
    if getattr(identity, "tokenizer_sha256", None) != tokenizer_sha256:
        raise ValueError(f"{label} tokenizer identity differs from training plan")
    if getattr(identity, "source_commit", None) != source_commit:
        raise ValueError(f"{label} source commit differs from training plan")
    if producer_sha256 is not None and getattr(identity, "producer_sha256", None) != producer_sha256:
        raise ValueError(f"{label} producer identity differs from training plan")
    # Force the strongest exact content contract now; malformed/orphan/mutated strict caches
    # therefore fail before any optimizer or dataloader work begins.
    provider_identity_sha256(cache, label=label)


class AuthoritativeGroundedGeneratorTrainingRunner(GroundedGeneratorTrainingRunner):
    """Final grounded runner supporting causal/seq2seq and contested evidence stances."""
    def __init__(self, *args: Any, generator_family: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("generator_family must be causal_lm or seq2seq_lm")
        self.generator_family = generator_family
        assert_advanced_training_tokenizer(self.tokenizer)
        _assert_cache_binding(
            self.teacher_cache,
            label="teacher cache",
            expected_kind="teacher_logits",
            dataset_manifest_sha256=self.plan.dataset_manifest_sha256,
            tokenizer_sha256=self.plan.tokenizer_sha256,
            source_commit=self.plan.source_commit,
            producer_sha256=self.plan.teacher_model_sha256,
        )
        _assert_cache_binding(
            self.reference_cache,
            label="reference cache",
            expected_kind="reference_policy_log_probs",
            dataset_manifest_sha256=self.plan.dataset_manifest_sha256,
            tokenizer_sha256=self.plan.tokenizer_sha256,
            source_commit=self.plan.source_commit,
        )
        if self.retriever_batch_builder is not None:
            if getattr(self.retriever_batch_builder, "tokenizer_sha256", None) != self.plan.tokenizer_sha256:
                raise ValueError("retriever supervision tokenizer identity differs from training plan")
            _assert_cache_binding(
                getattr(self.retriever_batch_builder, "utility_cache", None),
                label="retriever utility cache",
                expected_kind="document_lm_utility",
                dataset_manifest_sha256=self.plan.dataset_manifest_sha256,
                tokenizer_sha256=self.plan.tokenizer_sha256,
                source_commit=self.plan.source_commit,
            )

    def _authoritative_collator(self) -> Any:
        collator = MultiEvidenceSeq2SeqGroundedCollator if self.generator_family == "seq2seq_lm" else MultiEvidenceCausalGroundedCollator
        return collator(self.tokenizer, self.collator_config, teacher_cache=self.teacher_cache, reference_cache=self.reference_cache, retriever_batch_builder=self.retriever_batch_builder)

    def _authoritative_model(self) -> Any:
        if self.generator_family == "seq2seq_lm":
            return Seq2SeqGroundedGeneratorTrainingModule(base_model=self.base_model, config=self.plan.architecture, retriever_model=self.retriever_model)
        return GroundedGeneratorTrainingModule(base_model=self.base_model, config=self.plan.architecture, retriever_model=self.retriever_model)

    def run(self, *, resume_checkpoint_digest: str | None = None, event_sink: Any | None = None) -> AdvancedTrainingRunResult:
        train_dataset = ManifestBoundAuthoritativeJsonlDataset(self.train_split.path, expected_sha256=self.train_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.train_split.split_name, record_kind="grounded_generation", expected_record_count=self.train_split.expected_record_count)
        validation_dataset = ManifestBoundAuthoritativeJsonlDataset(self.validation_split.path, expected_sha256=self.validation_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.validation_split.split_name, record_kind="grounded_generation", expected_record_count=self.validation_split.expected_record_count)
        self._preflight(train_dataset); self._preflight(validation_dataset)
        effective_trainability = _effective_trainability(self.plan.stages, self.trainability)
        path_binding = GroundedGeneratorPathBinding(self.generator_family, self.collator_config)
        input_identity = AdvancedTrainingInputIdentity(
            kind="grounded_generation", plan_sha256=self.plan.plan_sha256,
            training_split_sha256=train_dataset.binding.content_sha256, validation_split_sha256=validation_dataset.binding.content_sha256,
            tokenizer_sha256=self.plan.tokenizer_sha256, execution_config_sha256=dataclass_sha256(self.execution, label="advanced-execution-config"),
            collator_config_sha256=dataclass_sha256(path_binding, label="grounded-generator-path-binding"), trainability_sha256=trainability_sha256(effective_trainability),
            teacher_cache_sha256=provider_identity_sha256(self.teacher_cache, label="teacher cache"), reference_cache_sha256=provider_identity_sha256(self.reference_cache, label="reference cache"),
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
            runtimes.append(StageRuntime(dataloader=_loader(train_dataset, sampler, collator, batch_size=self.execution.train_batch_size, execution=self.execution), step=_TrainabilityStep(step, effective_trainability[stage.name]), sampler=sampler, collator=collator))
        validation_loader = _loader(validation_dataset, None, self._authoritative_collator(), batch_size=self.execution.validation_batch_size, execution=self.execution)
        evaluator = GroundedValidationEvaluator(validation_loader, validation_steps, ValidationLimits(self.execution.validation_maximum_batches))
        manager = AdvancedCheckpointManager(self.checkpoint_root)
        engine = TorchTrainingEngine(model, trainer_config, manager)
        summary = engine.fit(runtimes, resume_checkpoint_digest=resume_checkpoint_digest, evaluator=evaluator, event_sink=event_sink)
        return AdvancedTrainingRunResult(summary, self.plan.plan_sha256, input_identity.input_sha256, train_dataset.binding.content_sha256, validation_dataset.binding.content_sha256, str(manager.root))


class AuthoritativeDynamicRagPolicyTrainingRunner(DynamicRagPolicyTrainingRunner):
    """Final dynamic runner with exact legal-action masking and strict off-policy loss."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert_advanced_training_tokenizer(self.tokenizer)
        _assert_cache_binding(
            self.hidden_state_cache,
            label="hidden-state cache",
            expected_kind="generator_hidden_states",
            dataset_manifest_sha256=self.plan.dataset_manifest_sha256,
            tokenizer_sha256=self.tokenizer_sha256,
            source_commit=self.plan.source_commit,
            producer_sha256=self.plan.base_generator_sha256,
        )

    def run(self, *, resume_checkpoint_digest: str | None = None, event_sink: Any | None = None) -> AdvancedTrainingRunResult:
        train_dataset = ManifestBoundAuthoritativeJsonlDataset(self.train_split.path, expected_sha256=self.train_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.train_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=self.train_split.expected_record_count)
        validation_dataset = ManifestBoundAuthoritativeJsonlDataset(self.validation_split.path, expected_sha256=self.validation_split.content_sha256, dataset_manifest_sha256=self.plan.dataset_manifest_sha256, split_name=self.validation_split.split_name, record_kind="dynamic_rag_episode", expected_record_count=self.validation_split.expected_record_count)
        self._preflight(train_dataset); self._preflight(validation_dataset)
        effective_trainability = _effective_trainability(self.plan.stages, self.trainability)
        input_identity = AdvancedTrainingInputIdentity(
            kind="dynamic_rag_policy", plan_sha256=self.plan.plan_sha256,
            training_split_sha256=train_dataset.binding.content_sha256, validation_split_sha256=validation_dataset.binding.content_sha256,
            tokenizer_sha256=self.tokenizer_sha256, execution_config_sha256=dataclass_sha256(self.execution, label="advanced-execution-config"),
            collator_config_sha256=dataclass_sha256(self.collator_config, label="dynamic-collator-config"), trainability_sha256=trainability_sha256(effective_trainability),
            hidden_state_cache_sha256=provider_identity_sha256(self.hidden_state_cache, label="hidden-state cache"),
        )
        model = DynamicRagPolicyModel(self.plan.architecture)
        trainer_config = dynamic_plan_to_trainer_config(self.plan, device=self.execution.device, precision=self.execution.precision, gradient_accumulation_steps=self.execution.gradient_accumulation_steps, max_grad_norm=self.execution.max_grad_norm, seed=self.execution.seed, deterministic_algorithms=self.execution.deterministic_algorithms, ddp=self.execution.ddp, weight_decay=self.execution.weight_decay, scheduler=self.execution.scheduler, warmup_steps=self.execution.warmup_steps)
        trainer_config = _trainer_with_evaluation(trainer_config, self.execution, input_identity)
        runtimes, validation_steps = [], []
        for stage_index, stage in enumerate(self.plan.stages):
            sampler = ResumableDeterministicSampler(len(train_dataset), seed=self.execution.seed + stage_index, shuffle=True)
            collator = LegalActionDynamicRagEpisodeCollator(self.tokenizer, self.plan.architecture, self.collator_config, hidden_state_cache=self.hidden_state_cache)
            step = LegalActionDynamicRetrievalPolicyStep(DynamicPolicyStepConfig(stage.objective), actions=self.plan.architecture.actions)
            validation_steps.append(LegalActionDynamicRetrievalPolicyStep(DynamicPolicyStepConfig(stage.objective), actions=self.plan.architecture.actions))
            runtimes.append(StageRuntime(dataloader=_loader(train_dataset, sampler, collator, batch_size=self.execution.train_batch_size, execution=self.execution), step=_TrainabilityStep(step, effective_trainability[stage.name]), sampler=sampler, collator=collator))
        validation_collator = LegalActionDynamicRagEpisodeCollator(self.tokenizer, self.plan.architecture, self.collator_config, hidden_state_cache=self.hidden_state_cache)
        validation_loader = _loader(validation_dataset, None, validation_collator, batch_size=self.execution.validation_batch_size, execution=self.execution)
        evaluator = DynamicPolicyValidationEvaluator(validation_loader, validation_steps, ValidationLimits(self.execution.validation_maximum_batches))
        manager = AdvancedCheckpointManager(self.checkpoint_root)
        engine = TorchTrainingEngine(model, trainer_config, manager)
        summary = engine.fit(runtimes, resume_checkpoint_digest=resume_checkpoint_digest, evaluator=evaluator, event_sink=event_sink)
        return AdvancedTrainingRunResult(summary, self.plan.plan_sha256, input_identity.input_sha256, train_dataset.binding.content_sha256, validation_dataset.binding.content_sha256, str(manager.root))


__all__ = ["AuthoritativeDynamicRagPolicyTrainingRunner", "AuthoritativeGroundedGeneratorTrainingRunner", "GroundedGeneratorPathBinding"]
