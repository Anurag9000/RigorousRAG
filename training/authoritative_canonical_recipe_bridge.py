"""Recipe bridge that admits only restart-verified canonical training-data v2 bundles.

Historical bundle v1 readers remain available. New authoritative recipes should enter here: the
neutral bundle is reopened, its outer canonical receipt is independently verified by the relevant
Grounded/Dynamic v2 authority (including retained lineage artifacts), bundle identities are
cross-checked against that authority, and only then is the existing deterministic recipe emitter
invoked. No model is loaded and no training executes here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

from training.authoritative_dynamic_canonical_training_data import (
    verify_authoritative_dynamic_canonical_training_data,
)
from training.authoritative_grounded_canonical_training_data import (
    verify_authoritative_grounded_canonical_training_data,
)
from training.canonical_bundle_recipe_bridge import (
    write_dynamic_recipe_from_canonical_bundle,
    write_grounded_recipe_from_canonical_bundle,
)
from training.canonical_training_data_bundle import (
    CanonicalTrainingDataBundle,
    read_canonical_training_data_bundle,
)
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_recipes import AdvancedRecipeReceipt
from training.advanced_rag_runner import ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.dynamic_retrieval_policy import DynamicPolicyArchitecture, DynamicRetrievalBudget
from training.grounded_generation import GroundedGenerationArchitectureConfig
from training.grounded_supervision_pipeline import RetrieverCouplingConfig
from training.local_artifact_loading import LocalArtifactTreeBinding


def _outer_canonical_receipt_path(bundle: CanonicalTrainingDataBundle) -> Path:
    """Resolve the outer v2 canonical receipt without overloading dataset_receipt_path.

    Grounded v2 has no nested final-dataset publication receipt, so its neutral historical field
    remains the outer canonical receipt. Dynamic v2 has a nested ``published`` dataset authority;
    its dataset_receipt_path points there and the outer canonical receipt is the grandparent's
    canonical root child.
    """
    selected = Path(bundle.dataset_receipt_path)
    if bundle.kind == "grounded_generation":
        return selected
    if bundle.kind == "dynamic_rag_policy":
        if selected.name != "publication_receipt.json" or selected.parent.name != "published":
            raise ValueError("dynamic canonical bundle dataset receipt is not the canonical published receipt")
        return selected.parent.parent / "canonical_receipt.json"
    raise ValueError("unsupported canonical training bundle kind")


def read_authoritative_canonical_training_bundle(path: str | Path) -> CanonicalTrainingDataBundle:
    bundle = read_canonical_training_data_bundle(path)
    canonical_receipt_path = _outer_canonical_receipt_path(bundle)
    if bundle.kind == "grounded_generation":
        verified = verify_authoritative_grounded_canonical_training_data(canonical_receipt_path)
        manifest_sha = verified.manifest.manifest_digest
        canonical_sha = verified.receipt.receipt_sha256
        expected_splits = {
            item.name: (item.sha256, item.record_count)
            for item in verified.receipt.splits
        }
    elif bundle.kind == "dynamic_rag_policy":
        verified = verify_authoritative_dynamic_canonical_training_data(canonical_receipt_path)
        manifest_sha = verified.dataset.manifest.manifest_digest
        canonical_sha = verified.receipt.receipt_sha256
        expected_splits = {
            item.name: (item.sha256, item.record_count)
            for item in verified.dataset.receipt.splits
        }
    else:  # pragma: no cover
        raise ValueError("unsupported canonical training bundle kind")
    if manifest_sha != bundle.dataset_manifest_sha256:
        raise ValueError("canonical v2 authority manifest differs from bundle")
    if bundle.canonical_receipt.get("receipt_sha256") != canonical_sha:
        raise ValueError("canonical v2 receipt identity differs from bundle")
    actual_splits = {item.name: (item.sha256, item.record_count) for item in bundle.splits}
    if actual_splits != expected_splits:
        raise ValueError("canonical v2 split universe differs from bundle")
    return bundle


def write_grounded_recipe_from_authoritative_bundle(
    bundle_path: str | Path,
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
    bundle = read_authoritative_canonical_training_bundle(bundle_path)
    if bundle.kind != "grounded_generation":
        raise ValueError("authoritative bundle is not grounded_generation")
    return write_grounded_recipe_from_canonical_bundle(
        bundle,
        train_split_name=train_split_name,
        validation_split_name=validation_split_name,
        output_path=output_path,
        run_id=run_id,
        source_commit=source_commit,
        base_model=base_model,
        tokenizer=tokenizer,
        architecture=architecture,
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        include_preference=include_preference,
        retriever_model=retriever_model,
        retriever_coupling=retriever_coupling,
        trainability=trainability,
        resume_checkpoint_digest=resume_checkpoint_digest,
    )


def write_dynamic_recipe_from_authoritative_bundle(
    bundle_path: str | Path,
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
    bundle = read_authoritative_canonical_training_bundle(bundle_path)
    if bundle.kind != "dynamic_rag_policy":
        raise ValueError("authoritative bundle is not dynamic_rag_policy")
    return write_dynamic_recipe_from_canonical_bundle(
        bundle,
        train_split_name=train_split_name,
        validation_split_name=validation_split_name,
        output_path=output_path,
        run_id=run_id,
        source_commit=source_commit,
        generator=generator,
        tokenizer=tokenizer,
        retrieval_stack_sha256=retrieval_stack_sha256,
        architecture=architecture,
        budget=budget,
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        trainability=trainability,
        resume_checkpoint_digest=resume_checkpoint_digest,
    )


__all__ = [
    "read_authoritative_canonical_training_bundle",
    "write_dynamic_recipe_from_authoritative_bundle",
    "write_grounded_recipe_from_authoritative_bundle",
]
