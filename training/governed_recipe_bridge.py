"""Bridge verified governed dataset publications directly into advanced-RAG recipes.

These helpers eliminate manual ``LocalTrainingSplit`` construction and prevent mixing split
bytes, manifests, or supervision caches from different runs. Canonical grounded/dynamic
bridges re-check sealed cache contracts and embed those exact contracts in emitted training
configurations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from training.advanced_rag_config import TensorCacheSpec
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_recipes import AdvancedRecipeReceipt, write_dynamic_training_recipe, write_grounded_training_recipe
from training.advanced_rag_runner import LocalTrainingSplit, ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.dynamic_canonical_training_data_pipeline import CanonicalDynamicTrainingDataResult
from training.dynamic_dataset_io import VerifiedDynamicDatasetPublication
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalBudget
from training.grounded_canonical_training_data_pipeline import CanonicalGroundedTrainingDataResult
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


def _canonical_grounded_split(dataset: CanonicalGroundedTrainingDataResult, name: str) -> LocalTrainingSplit:
    matches = [item for item in dataset.splits if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"canonical grounded dataset has no unique split {name!r}")
    item = matches[0]
    return LocalTrainingSplit(item.path, item.sha256, item.name, item.record_count)


def _dynamic_split(dataset: VerifiedDynamicDatasetPublication, name: str) -> LocalTrainingSplit:
    matches = [item for item in dataset.receipt.splits if item.name == name]
    if len(matches) != 1:
        raise ValueError(f"dynamic governed dataset has no unique split {name!r}")
    item = matches[0]
    return LocalTrainingSplit(item.path, item.sha256, item.name, item.record_count)


def _cache_spec(cache: object | None, expected_contract: str | None, label: str) -> TensorCacheSpec | None:
    if cache is None:
        if expected_contract is not None:
            raise ValueError(f"{label} receipt exists without a live cache")
        return None
    identity = getattr(cache, "identity", None)
    root = getattr(cache, "root", None)
    seal = getattr(cache, "seal", None)
    if identity is None or root is None or not callable(seal):
        raise ValueError(f"{label} is not an authoritative sealable cache")
    contract = seal()
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(f"{label} sealed contract differs from canonical receipt")
    return TensorCacheSpec(root=str(root), identity=identity, contract_sha256=contract)


def write_grounded_recipe_from_governed_dataset(
    dataset: VerifiedGovernedGroundedDataset, *, train_split_name: str, validation_split_name: str, output_path: str | Path,
    run_id: str, source_commit: str, base_model: LocalArtifactTreeBinding, tokenizer: LocalArtifactTreeBinding,
    architecture: GroundedGenerationArchitectureConfig, checkpoint_root: str | Path, execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: GroundedCollatorConfig = GroundedCollatorConfig(), include_preference: bool = True, teacher_cache: TensorCacheSpec | None = None,
    reference_cache: TensorCacheSpec | None = None, retriever_model: LocalArtifactTreeBinding | None = None, retriever_utility_cache: TensorCacheSpec | None = None,
    retriever_coupling: RetrieverCouplingConfig = RetrieverCouplingConfig(), trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    if not isinstance(dataset, VerifiedGovernedGroundedDataset):
        raise ValueError("dataset must be VerifiedGovernedGroundedDataset")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    return write_grounded_training_recipe(
        output_path, run_id=run_id, source_commit=source_commit, dataset_manifest_sha256=dataset.manifest.manifest_digest,
        base_model=base_model, tokenizer=tokenizer, architecture=architecture, train_split=_grounded_split(dataset, train_split_name),
        validation_split=_grounded_split(dataset, validation_split_name), checkpoint_root=checkpoint_root, execution=execution,
        collator=collator, include_preference=include_preference, teacher_cache=teacher_cache, reference_cache=reference_cache,
        retriever_model=retriever_model, retriever_utility_cache=retriever_utility_cache, retriever_coupling=retriever_coupling,
        trainability=trainability, resume_checkpoint_digest=resume_checkpoint_digest,
    )


def write_grounded_recipe_from_canonical_training_data(
    result: CanonicalGroundedTrainingDataResult, *, train_split_name: str, validation_split_name: str, output_path: str | Path,
    run_id: str, source_commit: str, base_model: LocalArtifactTreeBinding, tokenizer: LocalArtifactTreeBinding,
    architecture: GroundedGenerationArchitectureConfig, checkpoint_root: str | Path, execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: GroundedCollatorConfig = GroundedCollatorConfig(), include_preference: bool = True, retriever_model: LocalArtifactTreeBinding | None = None,
    retriever_coupling: RetrieverCouplingConfig = RetrieverCouplingConfig(), trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    """Emit a grounded recipe directly from the final-manifest-bound canonical result."""
    if not isinstance(result, CanonicalGroundedTrainingDataResult):
        raise ValueError("result must be CanonicalGroundedTrainingDataResult")
    if result.manifest.manifest_digest != result.receipt.dataset_manifest_sha256:
        raise ValueError("canonical grounded manifest differs from receipt")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    by_kind = {item.kind: item for item in result.receipt.caches}
    teacher = _cache_spec(result.teacher_cache, by_kind.get("teacher_logits").contract_sha256 if "teacher_logits" in by_kind else None, "teacher cache")
    reference = _cache_spec(result.reference_cache, by_kind.get("reference_policy_log_probs").contract_sha256 if "reference_policy_log_probs" in by_kind else None, "reference cache")
    utility = _cache_spec(result.retriever_utility_cache, by_kind.get("document_lm_utility").contract_sha256 if "document_lm_utility" in by_kind else None, "document utility cache")
    if (retriever_model is None) != (utility is None):
        raise ValueError("canonical document-utility cache and retriever model must be supplied together")
    return write_grounded_training_recipe(
        output_path, run_id=run_id, source_commit=source_commit, dataset_manifest_sha256=result.manifest.manifest_digest,
        base_model=base_model, tokenizer=tokenizer, architecture=architecture, train_split=_canonical_grounded_split(result, train_split_name),
        validation_split=_canonical_grounded_split(result, validation_split_name), checkpoint_root=checkpoint_root, execution=execution,
        collator=collator, include_preference=include_preference, teacher_cache=teacher, reference_cache=reference,
        retriever_model=retriever_model, retriever_utility_cache=utility, retriever_coupling=retriever_coupling,
        trainability=trainability, resume_checkpoint_digest=resume_checkpoint_digest,
    )


def write_dynamic_recipe_from_governed_dataset(
    dataset: VerifiedDynamicDatasetPublication, *, train_split_name: str, validation_split_name: str, output_path: str | Path,
    run_id: str, source_commit: str, generator: LocalArtifactTreeBinding, tokenizer: LocalArtifactTreeBinding, retrieval_stack_sha256: str,
    architecture: DynamicPolicyArchitecture, budget: DynamicRetrievalBudget, checkpoint_root: str | Path, execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: DynamicCollatorConfig = DynamicCollatorConfig(), include_need_selection: bool = True, hidden_state_cache: TensorCacheSpec | None = None,
    trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None, resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    if not isinstance(dataset, VerifiedDynamicDatasetPublication):
        raise ValueError("dataset must be VerifiedDynamicDatasetPublication")
    if train_split_name == validation_split_name:
        raise ValueError("training and validation split names must differ")
    return write_dynamic_training_recipe(
        output_path, run_id=run_id, source_commit=source_commit, dataset_manifest_sha256=dataset.manifest.manifest_digest,
        generator=generator, tokenizer=tokenizer, retrieval_stack_sha256=retrieval_stack_sha256, architecture=architecture,
        budget=budget, train_split=_dynamic_split(dataset, train_split_name), validation_split=_dynamic_split(dataset, validation_split_name),
        checkpoint_root=checkpoint_root, execution=execution, collator=collator, include_need_selection=include_need_selection,
        hidden_state_cache=hidden_state_cache, trainability=trainability, resume_checkpoint_digest=resume_checkpoint_digest,
    )


def write_dynamic_recipe_from_canonical_training_data(
    result: CanonicalDynamicTrainingDataResult, *, train_split_name: str, validation_split_name: str, output_path: str | Path,
    run_id: str, source_commit: str, generator: LocalArtifactTreeBinding, tokenizer: LocalArtifactTreeBinding, retrieval_stack_sha256: str,
    architecture: DynamicPolicyArchitecture, budget: DynamicRetrievalBudget, checkpoint_root: str | Path, execution: TrainingExecutionConfig = TrainingExecutionConfig(),
    collator: DynamicCollatorConfig = DynamicCollatorConfig(), trainability: Mapping[str, ParameterTrainabilityPolicy] | None = None,
    resume_checkpoint_digest: str | None = None,
) -> AdvancedRecipeReceipt:
    """Emit a need-selection recipe directly from the final two-phase canonical result."""
    if not isinstance(result, CanonicalDynamicTrainingDataResult):
        raise ValueError("result must be CanonicalDynamicTrainingDataResult")
    if result.dataset.manifest.manifest_digest != result.receipt.dataset_manifest_sha256:
        raise ValueError("canonical dynamic result dataset manifest differs from top-level receipt")
    if result.hidden_cache.identity.digest != result.receipt.hidden_cache_identity_sha256:
        raise ValueError("canonical dynamic result cache identity differs from top-level receipt")
    actual_contract = result.hidden_cache.seal()
    if actual_contract != result.receipt.hidden_cache_contract_sha256 or actual_contract != result.hidden_cache_receipt.cache_contract_sha256:
        raise ValueError("canonical dynamic result sealed hidden-cache contract differs from receipts")
    cache_spec = TensorCacheSpec(root=str(result.hidden_cache.root), identity=result.hidden_cache.identity, contract_sha256=actual_contract)
    return write_dynamic_recipe_from_governed_dataset(
        result.dataset, train_split_name=train_split_name, validation_split_name=validation_split_name, output_path=output_path,
        run_id=run_id, source_commit=source_commit, generator=generator, tokenizer=tokenizer,
        retrieval_stack_sha256=retrieval_stack_sha256, architecture=architecture, budget=budget, checkpoint_root=checkpoint_root,
        execution=execution, collator=collator, include_need_selection=True, hidden_state_cache=cache_spec,
        trainability=trainability, resume_checkpoint_digest=resume_checkpoint_digest,
    )


__all__ = [
    "write_dynamic_recipe_from_canonical_training_data",
    "write_dynamic_recipe_from_governed_dataset",
    "write_grounded_recipe_from_canonical_training_data",
    "write_grounded_recipe_from_governed_dataset",
]
