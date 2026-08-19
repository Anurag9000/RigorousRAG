"""Reconstruct and verify exact advanced-RAG training identities from run configs.

This module is intentionally source-only: it does not load model weights, datasets into
accelerators, or execute training. It rebuilds the same immutable input identity and generic
trainer configuration used by the authoritative runners, then verifies a content-addressed
checkpoint before export/promotion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from training.advanced_rag_authoritative_runner import GroundedGeneratorPathBinding
from training.advanced_rag_config import DynamicConfiguredRun, GroundedConfiguredRun, TensorCacheSpec
from training.advanced_rag_identity import AdvancedTrainingInputIdentity, dataclass_sha256, provider_identity_sha256, trainability_sha256
from training.advanced_rag_runner import _effective_trainability, _trainer_with_evaluation
from training.advanced_rag_steps import dynamic_plan_to_trainer_config, grounded_plan_to_trainer_config
from training.checkpointing import CheckpointManager


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _cache_identity(spec: TensorCacheSpec | None, *, label: str, expected_kind: str) -> str | None:
    if spec is None:
        return None
    if not isinstance(spec, TensorCacheSpec):
        raise ValueError(f"{label} must be TensorCacheSpec")
    if spec.identity.cache_kind != expected_kind:
        raise ValueError(f"{label} must use cache_kind={expected_kind}")
    cache = spec.build()
    # provider_identity_sha256 deliberately prefers the exact sealed content contract over
    # the weaker cache-configuration identity, matching the authoritative training runner.
    return provider_identity_sha256(cache, label=label)


def _retriever_supervision_sha256(config: GroundedConfiguredRun) -> str | None:
    if config.retriever_utility_cache is None:
        return None
    utility_cache_sha = _cache_identity(
        config.retriever_utility_cache,
        label="retriever utility cache",
        expected_kind="document_lm_utility",
    )
    if utility_cache_sha is None:
        raise RuntimeError("retriever utility cache identity unexpectedly missing")
    return _digest(
        {
            "schema": "rigorousrag-cached-document-utility-builder/v1",
            "tokenizer_sha256": config.plan.tokenizer_sha256,
            "utility_cache_sha256": utility_cache_sha,
            "config_sha256": config.retriever_coupling.config_sha256,
        }
    )


def grounded_training_input_identity(config: GroundedConfiguredRun) -> AdvancedTrainingInputIdentity:
    if not isinstance(config, GroundedConfiguredRun):
        raise ValueError("config must be GroundedConfiguredRun")
    trainability = _effective_trainability(config.plan.stages, config.trainability)
    path_binding = GroundedGeneratorPathBinding(config.base_model.artifact_kind, config.collator)
    return AdvancedTrainingInputIdentity(
        kind="grounded_generation",
        plan_sha256=config.plan.plan_sha256,
        training_split_sha256=config.train_split.content_sha256,
        validation_split_sha256=config.validation_split.content_sha256,
        tokenizer_sha256=config.plan.tokenizer_sha256,
        execution_config_sha256=dataclass_sha256(config.execution, label="advanced-execution-config"),
        collator_config_sha256=dataclass_sha256(path_binding, label="grounded-generator-path-binding"),
        trainability_sha256=trainability_sha256(trainability),
        teacher_cache_sha256=_cache_identity(config.teacher_cache, label="teacher cache", expected_kind="teacher_logits"),
        reference_cache_sha256=_cache_identity(config.reference_cache, label="reference cache", expected_kind="reference_policy_log_probs"),
        retriever_supervision_sha256=_retriever_supervision_sha256(config),
    )


def dynamic_training_input_identity(config: DynamicConfiguredRun) -> AdvancedTrainingInputIdentity:
    if not isinstance(config, DynamicConfiguredRun):
        raise ValueError("config must be DynamicConfiguredRun")
    trainability = _effective_trainability(config.plan.stages, config.trainability)
    return AdvancedTrainingInputIdentity(
        kind="dynamic_rag_policy",
        plan_sha256=config.plan.plan_sha256,
        training_split_sha256=config.train_split.content_sha256,
        validation_split_sha256=config.validation_split.content_sha256,
        tokenizer_sha256=config.tokenizer.expected_sha256,
        execution_config_sha256=dataclass_sha256(config.execution, label="advanced-execution-config"),
        collator_config_sha256=dataclass_sha256(config.collator, label="dynamic-collator-config"),
        trainability_sha256=trainability_sha256(trainability),
        hidden_state_cache_sha256=_cache_identity(config.hidden_state_cache, label="hidden-state cache", expected_kind="generator_hidden_states"),
    )


def _grounded_trainer(config: GroundedConfiguredRun, identity: AdvancedTrainingInputIdentity) -> Any:
    trainer = grounded_plan_to_trainer_config(
        config.plan,
        device=config.execution.device,
        precision=config.execution.precision,
        gradient_accumulation_steps=config.execution.gradient_accumulation_steps,
        max_grad_norm=config.execution.max_grad_norm,
        seed=config.execution.seed,
        deterministic_algorithms=config.execution.deterministic_algorithms,
        ddp=config.execution.ddp,
        weight_decay=config.execution.weight_decay,
        scheduler=config.execution.scheduler,
        warmup_steps=config.execution.warmup_steps,
    )
    return _trainer_with_evaluation(trainer, config.execution, identity)


def _dynamic_trainer(config: DynamicConfiguredRun, identity: AdvancedTrainingInputIdentity) -> Any:
    trainer = dynamic_plan_to_trainer_config(
        config.plan,
        device=config.execution.device,
        precision=config.execution.precision,
        gradient_accumulation_steps=config.execution.gradient_accumulation_steps,
        max_grad_norm=config.execution.max_grad_norm,
        seed=config.execution.seed,
        deterministic_algorithms=config.execution.deterministic_algorithms,
        ddp=config.execution.ddp,
        weight_decay=config.execution.weight_decay,
        scheduler=config.execution.scheduler,
        warmup_steps=config.execution.warmup_steps,
    )
    return _trainer_with_evaluation(trainer, config.execution, identity)


@dataclass(frozen=True)
class VerifiedAdvancedCheckpointBinding:
    kind: str
    checkpoint_digest: str
    plan_sha256: str
    training_input_sha256: str
    training_config_sha256: str
    bound_run_id: str
    source_commit: str
    dataset_manifest_sha256: str
    model_architecture: str
    generator_family: str
    tokenizer_sha256: str
    retriever_positive_label_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced checkpoint kind")
        if self.generator_family not in {"causal_lm", "seq2seq_lm"}:
            raise ValueError("advanced checkpoint binding requires causal_lm or seq2seq_lm generator_family")
        for name in ("checkpoint_digest", "plan_sha256", "training_input_sha256", "training_config_sha256", "dataset_manifest_sha256", "tokenizer_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        commit = str(self.source_commit).strip().lower()
        if len(commit) not in {40, 64} or any(ch not in "0123456789abcdef" for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)
        if not isinstance(self.bound_run_id, str) or not self.bound_run_id.strip():
            raise ValueError("bound_run_id is required")
        if not isinstance(self.model_architecture, str) or not self.model_architecture.strip():
            raise ValueError("model_architecture is required")
        if self.kind == "grounded_generation":
            if self.retriever_positive_label_index is not None and (
                isinstance(self.retriever_positive_label_index, bool)
                or not isinstance(self.retriever_positive_label_index, int)
                or self.retriever_positive_label_index < 0
            ):
                raise ValueError("retriever_positive_label_index must be non-negative or None")
        elif self.retriever_positive_label_index is not None:
            raise ValueError("dynamic checkpoint binding may not carry grounded retriever adapter data")


def verify_checkpoint_against_run_config(
    checkpoint_manager: CheckpointManager,
    checkpoint_digest: str,
    config: GroundedConfiguredRun | DynamicConfiguredRun,
) -> VerifiedAdvancedCheckpointBinding:
    """Fail closed unless a checkpoint exactly matches the supplied immutable run config."""
    if not isinstance(checkpoint_manager, CheckpointManager):
        raise ValueError("checkpoint_manager must be CheckpointManager")
    _, manifest = checkpoint_manager.verify(checkpoint_digest)

    if isinstance(config, GroundedConfiguredRun):
        identity = grounded_training_input_identity(config)
        trainer = _grounded_trainer(config, identity)
        expected_architecture = f"grounded_generation:{config.plan.plan_sha256}"
        generator_family = config.base_model.artifact_kind
        tokenizer_sha256 = config.plan.tokenizer_sha256
        retriever_positive_label_index = config.retriever_coupling.positive_label_index if config.retriever_model is not None else None
        kind = "grounded_generation"
        plan_sha256 = config.plan.plan_sha256
        dataset_sha = config.plan.dataset_manifest_sha256
        source_commit = config.plan.source_commit
    elif isinstance(config, DynamicConfiguredRun):
        identity = dynamic_training_input_identity(config)
        trainer = _dynamic_trainer(config, identity)
        expected_architecture = f"dynamic_retrieval_policy:{config.plan.plan_sha256}"
        generator_family = config.generator.artifact_kind
        tokenizer_sha256 = config.tokenizer.expected_sha256
        retriever_positive_label_index = None
        kind = "dynamic_rag_policy"
        plan_sha256 = config.plan.plan_sha256
        dataset_sha = config.plan.dataset_manifest_sha256
        source_commit = config.plan.source_commit
    else:
        raise TypeError("unsupported configured run type")

    failures: list[str] = []
    if manifest.run_id != trainer.run_id:
        failures.append("run_id")
    if manifest.training_config_digest != trainer.digest:
        failures.append("training_config_digest")
    if manifest.source_commit != source_commit:
        failures.append("source_commit")
    if manifest.dataset_manifest_digest != dataset_sha:
        failures.append("dataset_manifest_digest")
    if manifest.model_architecture != expected_architecture:
        failures.append("model_architecture")
    if failures:
        raise ValueError(f"checkpoint differs from configured training identity: {','.join(failures)}")

    return VerifiedAdvancedCheckpointBinding(
        kind=kind,
        checkpoint_digest=manifest.digest,
        plan_sha256=plan_sha256,
        training_input_sha256=identity.input_sha256,
        training_config_sha256=trainer.digest,
        bound_run_id=trainer.run_id,
        source_commit=source_commit,
        dataset_manifest_sha256=dataset_sha,
        model_architecture=expected_architecture,
        generator_family=generator_family,
        tokenizer_sha256=tokenizer_sha256,
        retriever_positive_label_index=retriever_positive_label_index,
    )


__all__ = [
    "VerifiedAdvancedCheckpointBinding",
    "dynamic_training_input_identity",
    "grounded_training_input_identity",
    "verify_checkpoint_against_run_config",
]
