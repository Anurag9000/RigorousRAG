"""Emit advanced-RAG training configs from restart-verified canonical data bundles."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from training.advanced_rag_config import TensorCacheSpec
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_recipes import AdvancedRecipeReceipt, write_dynamic_training_recipe, write_grounded_training_recipe
from training.advanced_rag_runner import LocalTrainingSplit, ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.canonical_training_data_bundle import CanonicalTrainingDataBundle
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig
from training.grounded_supervision_pipeline import RetrieverCouplingConfig
from training.local_artifact_loading import LocalArtifactTreeBinding


def _split(bundle: CanonicalTrainingDataBundle, name: str) -> LocalTrainingSplit:
    matches = [item for item in bundle.splits if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"canonical bundle has no unique split {name!r}")
    item = matches[0]
    return LocalTrainingSplit(item.path, item.sha256, item.name, item.record_count)


def _cache(bundle: CanonicalTrainingDataBundle, role: str) -> TensorCacheSpec | None:
    matches = [item for item in bundle.caches if item.role == role]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"canonical bundle has duplicate cache role {role!r}")
    item = matches[0]
    cache = item.reopen()
    actual = cache.seal()
    if actual != item.contract_sha256:
        raise ValueError(f"canonical bundle cache role {role!r} changed after restart verification")
    return TensorCacheSpec(root=item.root, identity=item.identity, contract_sha256=actual)


def write_grounded_recipe_from_canonical_bundle(
    bundle: CanonicalTrainingDataBundle,
    *,
    train_split_name: str,
    validation_split_name: str,
    output_path: str | Path,
    run_id: str,
    source_commit: str,
    base_model: LocalArtifactTreeBinding,
    tokenizer: LocalArtifactTreeBinding,
    architecture: GroundedGenerationArchitectureConfig,
    checkpoint_root: str | Path,
    execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: GroundedCollatorConfig = GroundedCollatorConfig(),
    include_preference: bool = True,
    retriever_model: LocalArtifactTreeBinding | None = None,
    retriever_coupling: RetrieverCouplingConfig = RetrieverCouplingConfig(),
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    if not isinstance(bundle, CanonicalTrainingDataBundle) or bundle.kind != "grounded_generation":
        raise ValueError("bundle must be a grounded_generation CanonicalTrainingDataBundle")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    teacher = _cache(bundle, "teacher")
    reference = _cache(bundle, "reference")
    utility = _cache(bundle, "retriever_utility")
    if (retriever_model is None) != (utility is None):
        raise ValueError("retriever_model and canonical retriever_utility cache must be supplied together")
    return write_grounded_training_recipe(
        output_path,
        run_id=run_id,
        source_commit=source_commit,
        dataset_manifest_sha256=bundle.dataset_manifest_sha256,
        base_model=base_model,
        tokenizer=tokenizer,
        architecture=architecture,
        train_split=_split(bundle, train_split_name),
        validation_split=_split(bundle, validation_split_name),
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        include_preference=include_preference,
        teacher_cache=teacher,
        reference_cache=reference,
        retriever_model=retriever_model,
        retriever_utility_cache=utility,
        retriever_coupling=retriever_coupling,
        trainability=trainability,
        resume_checkpoint_digest=resume_checkpoint_digest,
    )


def write_dynamic_recipe_from_canonical_bundle(
    bundle: CanonicalTrainingDataBundle,
    *,
    train_split_name: str,
    validation_split_name: str,
    output_path: str | Path,
    run_id: str,
    source_commit: str,
    generator: LocalArtifactTreeBinding,
    tokenizer: LocalArtifactTreeBinding,
    retrieval_stack_sha256: str,
    architecture: DynamicPolicyArchitecture,
    budget: DynamicRetrievalBudget,
    checkpoint_root: str | Path,
    execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: DynamicCollatorConfig = DynamicCollatorConfig(),
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    if not isinstance(bundle, CanonicalTrainingDataBundle) or bundle.kind != "dynamic_rag_policy":
        raise ValueError("bundle must be a dynamic_rag_policy CanonicalTrainingDataBundle")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    hidden = _cache(bundle, "hidden_state")
    if hidden is None:
        raise ValueError("canonical dynamic bundle lacks hidden_state cache")
    return write_dynamic_training_recipe(
        output_path,
        run_id=run_id,
        source_commit=source_commit,
        dataset_manifest_sha256=bundle.dataset_manifest_sha256,
        generator=generator,
        tokenizer=tokenizer,
        retrieval_stack_sha256=retrieval_stack_sha256,
        architecture=architecture,
        budget=budget,
        train_split=_split(bundle, train_split_name),
        validation_split=_split(bundle, validation_split_name),
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        include_need_selection=True,
        hidden_state_cache=hidden,
        trainability=trainability,
        resume_checkpoint_digest=resume_checkpoint_digest,
    )


__all__ = ["write_dynamic_recipe_from_canonical_bundle", "write_grounded_recipe_from_canonical_bundle"]
