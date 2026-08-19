"""Bridge verified governed dataset publications directly into advanced-RAG recipes.

These helpers eliminate the final manual construction of ``LocalTrainingSplit`` and accidental
mixing of a split from one manifest with the manifest SHA of another.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from training.advanced_rag_config import TensorCacheSpec
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_recipes import AdvancedRecipeReceipt, write_dynamic_training_recipe, write_grounded_training_recipe
from training.advanced_rag_runner import LocalTrainingSplit, ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig
from training.grounded_supervision_pipeline import RetrieverCouplingConfig
from training.governed_grounded_io import VerifiedGovernedGroundedDataset
from training.local_artifact_loading import LocalArtifactTreeBinding


def _grounded_split(dataset: VerifiedGovernedGroundedDataset, name: str) -> LocalTrainingSplit:
    matches = [item for item in dataset.receipt.splits if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"grounded governed dataset has no unique split {name!r}")
    item = matches[0]
    return LocalTrainingSplit(item.output_path, item.output_sha256, item.name, item.record_count)


def _dynamic_split(dataset: VerifiedDynamicDatasetPublication, name: str) -> LocalTrainingSplit:
    matches = [item for item in dataset.receipt.splits if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"dynamic governed dataset has no unique split {name!r}")
    item = matches[0]
    return LocalTrainingSplit(item.path, item.sha256, item.name, item.record_count)


def write_grounded_recipe_from_governed_dataset(
    dataset: VerifiedGovernedGroundedDataset,
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
    teacher_cache: TensorCacheSpec | None = None,
    reference_cache: TensorCacheSpec | None = None,
    retriever_model: LocalArtifactTreeBinding | None = None,
    retriever_utility_cache: TensorCacheSpec | None = None,
    retriever_coupling: RetrieverCouplingConfig = RetrieverCouplingConfig(),
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    if not isinstance(dataset, VerifiedGovernedGroundedDataset):
        raise ValueError("dataset must be VerifiedGovernedGroundedDataset")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    return write_grounded_training_recipe(
        output_path,
        run_id=run_id,
        source_commit=source_commit,
        dataset_manifest_sha256=dataset.manifest.manifest_digest,
        base_model=base_model,
        tokenizer=tokenizer,
        architecture=architecture,
        train_split=_grounded_split(dataset, train_split_name),
        validation_split=_grounded_split(dataset, validation_split_name),
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        include_preference=include_preference,
        teacher_cache=teacher_cache,
        reference_cache=reference_cache,
        retriever_model=retriever_model,
        retriever_utility_cache=retriever_utility_cache,
        retriever_coupling=retriever_coupling,
        trainability=trainability,
        resume_checkpoint_digest=resume_checkpoint_digest,
    )


def write_dynamic_recipe_from_governed_dataset(
    dataset: VerifiedDynamicDatasetPublication,
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
    include_need_selection: bool = True,
    hidden_state_cache: TensorCacheSpec | None = None,
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    if not isinstance(dataset, VerifiedDynamicDatasetPublication):
        raise ValueError("dataset must be VerifiedDynamicDatasetPublication")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    return write_dynamic_training_recipe(
        output_path,
        run_id=run_id,
        source_commit=source_commit,
        dataset_manifest_sha256=dataset.manifest.manifest_digest,
        generator=generator,
        tokenizer=tokenizer,
        retrieval_stack_sha256=retrieval_stack_sha256,
        architecture=architecture,
        budget=budget,
        train_split=_dynamic_split(dataset, train_split_name),
        validation_split=_dynamic_split(dataset, validation_split_name),
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        include_need_selection=include_need_selection,
        hidden_state_cache=hidden_state_cache,
        trainability=trainability,
        resume_checkpoint_digest=resume_checkpoint_digest,
    )


__all__ = ["write_dynamic_recipe_from_governed_dataset", "write_grounded_recipe_from_governed_dataset"]
