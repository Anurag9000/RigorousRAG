"""Strict config-only recipe emission from restart-verified canonical-v2 training bundles.

The canonical bundle already seals the dataset/cache/source lineage.  This operator therefore
never accepts an independent ``source_commit``: it derives the commit from the verified outer
canonical receipt and uses that exact identity when emitting the authoritative training config.
No model is loaded and no training executes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_data import DynamicCollatorConfig, GroundedCollatorConfig
from training.advanced_rag_runner import ParameterTrainabilityPolicy, TrainingExecutionConfig
from training.authoritative_canonical_recipe_bridge import (
    read_authoritative_canonical_training_bundle,
    write_dynamic_recipe_from_authoritative_bundle,
    write_grounded_recipe_from_authoritative_bundle,
)
from training.dynamic_retrieval_policy import (
    DynamicPolicyArchitecture,
    DynamicRetrievalAction,
    DynamicRetrievalBudget,
)
from training.grounded_generation import GroundedGenerationArchitectureConfig, ReflectionAction
from training.grounded_supervision_pipeline import RetrieverCouplingConfig
from training.local_artifact_loading import LocalArtifactTreeBinding

_MAX_CONFIG_BYTES = 16 * 1024 * 1024


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="canonical recipe config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("canonical recipe config exceeds byte safety bound")
    try:
        value = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("canonical recipe config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("canonical recipe config must contain an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], allowed: set[str], label: str, *, required: set[str] | None = None) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    selected_required = allowed if required is None else required
    missing = selected_required - set(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")


def _artifact(raw: Any, label: str, kinds: set[str]) -> LocalArtifactTreeBinding:
    value = _mapping(raw, label)
    _closed(value, {"path", "sha256", "kind"}, label)
    if value["kind"] not in kinds:
        raise ValueError(f"{label}.kind must be one of {sorted(kinds)}")
    binding = LocalArtifactTreeBinding(value["path"], value["sha256"], value["kind"])
    binding.verify()
    return binding


def _dataclass_kwargs(raw: Any, cls: Any, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    value = dict(_mapping(raw, label))
    allowed = {field.name for field in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    return value


def _execution(raw: Any) -> TrainingExecutionConfig:
    return TrainingExecutionConfig(**_dataclass_kwargs(raw, TrainingExecutionConfig, "execution"))


def _trainability(raw: Any) -> Mapping[str, ParameterTrainabilityPolicy] | None:
    if raw is None:
        return None
    value = _mapping(raw, "trainability")
    result: dict[str, ParameterTrainabilityPolicy] = {}
    for stage, prefixes in value.items():
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("trainability stage names must be non-empty strings")
        if not isinstance(prefixes, list) or any(not isinstance(item, str) for item in prefixes):
            raise ValueError(f"trainability[{stage!r}] must be an array of parameter prefixes")
        result[stage] = ParameterTrainabilityPolicy(tuple(prefixes))
    return result


def _grounded_architecture(raw: Any) -> GroundedGenerationArchitectureConfig:
    value = dict(_mapping(raw, "architecture"))
    allowed = {"hidden_size", "attribution_size", "reflection_actions"}
    _closed(value, allowed, "architecture", required={"hidden_size"})
    if "reflection_actions" in value:
        actions = value["reflection_actions"]
        if not isinstance(actions, list):
            raise ValueError("architecture.reflection_actions must be an array")
        value["reflection_actions"] = tuple(ReflectionAction(item) for item in actions)
    return GroundedGenerationArchitectureConfig(**value)


def _dynamic_architecture(raw: Any) -> DynamicPolicyArchitecture:
    value = dict(_mapping(raw, "architecture"))
    allowed = {"feature_names", "hidden_size", "context_hidden_size", "need_projection_size", "actions"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"architecture contains unsupported fields: {sorted(unknown)}")
    if "feature_names" in value:
        if not isinstance(value["feature_names"], list):
            raise ValueError("architecture.feature_names must be an array")
        value["feature_names"] = tuple(value["feature_names"])
    if "actions" in value:
        if not isinstance(value["actions"], list):
            raise ValueError("architecture.actions must be an array")
        value["actions"] = tuple(DynamicRetrievalAction(item) for item in value["actions"])
    return DynamicPolicyArchitecture(**value)


def _budget(raw: Any) -> DynamicRetrievalBudget:
    value = _mapping(raw, "budget")
    allowed = {field.name for field in fields(DynamicRetrievalBudget)}
    required = {"max_generation_tokens", "max_retrievals", "max_verifications"}
    _closed(value, allowed, "budget", required=required)
    return DynamicRetrievalBudget(**dict(value))


def _canonical_source_commit(bundle: Any) -> str:
    receipt = bundle.canonical_receipt
    if bundle.kind == "grounded_generation":
        value = receipt.get("source_commit")
    else:
        lineage = receipt.get("runtime_lineage")
        if not isinstance(lineage, Mapping):
            raise ValueError("dynamic canonical receipt lacks runtime_lineage")
        value = lineage.get("source_commit")
        hidden_commit = receipt.get("hidden_cache_source_commit")
        if hidden_commit != value:
            raise ValueError("dynamic canonical runtime and hidden-cache source commits differ")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("canonical receipt lacks source_commit")
    return value


def _base_fields(raw: Mapping[str, Any]) -> tuple[Any, str, str, str, str, str | Path, Any, Any]:
    bundle = read_authoritative_canonical_training_bundle(raw["bundle_path"])
    source_commit = _canonical_source_commit(bundle)
    return (
        bundle,
        source_commit,
        raw["train_split_name"],
        raw["validation_split_name"],
        raw["run_id"],
        raw["checkpoint_root"],
        _execution(raw.get("execution")),
        _trainability(raw.get("trainability")),
    )


def run_grounded_canonical_recipe_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "grounded canonical recipe config")
    allowed = {
        "schema", "bundle_path", "output_path", "train_split_name", "validation_split_name", "run_id",
        "base_model", "tokenizer", "architecture", "checkpoint_root", "execution", "collator",
        "include_preference", "retriever_model", "retriever_coupling", "trainability", "resume_checkpoint_digest",
    }
    required = {
        "schema", "bundle_path", "output_path", "train_split_name", "validation_split_name", "run_id",
        "base_model", "tokenizer", "architecture", "checkpoint_root",
    }
    _closed(value, allowed, "grounded canonical recipe config", required=required)
    if value["schema"] != "rigorousrag-authoritative-grounded-canonical-recipe-config/v1":
        raise ValueError("unsupported grounded canonical recipe config schema")
    bundle, source_commit, train_name, validation_name, run_id, checkpoint_root, execution, trainability = _base_fields(value)
    if bundle.kind != "grounded_generation":
        raise ValueError("grounded recipe config requires a grounded canonical bundle")
    base_model = _artifact(value["base_model"], "base_model", {"causal_lm", "seq2seq_lm"})
    tokenizer = _artifact(value["tokenizer"], "tokenizer", {"tokenizer"})
    collator = GroundedCollatorConfig(**_dataclass_kwargs(value.get("collator"), GroundedCollatorConfig, "collator"))
    coupling = RetrieverCouplingConfig(**_dataclass_kwargs(value.get("retriever_coupling"), RetrieverCouplingConfig, "retriever_coupling"))
    retriever = None if value.get("retriever_model") is None else _artifact(value["retriever_model"], "retriever_model", {"sequence_classifier"})
    include_preference = value.get("include_preference", True)
    if not isinstance(include_preference, bool):
        raise ValueError("include_preference must be boolean")
    receipt = write_grounded_recipe_from_authoritative_bundle(
        value["bundle_path"],
        train_split_name=train_name,
        validation_split_name=validation_name,
        output_path=value["output_path"],
        run_id=run_id,
        source_commit=source_commit,
        base_model=base_model,
        tokenizer=tokenizer,
        architecture=_grounded_architecture(value["architecture"]),
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        include_preference=include_preference,
        retriever_model=retriever,
        retriever_coupling=coupling,
        trainability=trainability,
        resume_checkpoint_digest=value.get("resume_checkpoint_digest"),
    )
    return {
        "kind": receipt.kind,
        "run_id": receipt.run_id,
        "config_path": receipt.output_path,
        "config_sha256": receipt.config_sha256,
        "plan_sha256": receipt.plan_sha256,
        "recipe_receipt_sha256": receipt.receipt_sha256,
        "source_commit": source_commit,
        "canonical_bundle_sha256": bundle.bundle_sha256,
    }


def run_dynamic_canonical_recipe_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "dynamic canonical recipe config")
    allowed = {
        "schema", "bundle_path", "output_path", "train_split_name", "validation_split_name", "run_id",
        "generator", "tokenizer", "retrieval_stack_sha256", "architecture", "budget", "checkpoint_root",
        "execution", "collator", "trainability", "resume_checkpoint_digest",
    }
    required = {
        "schema", "bundle_path", "output_path", "train_split_name", "validation_split_name", "run_id",
        "generator", "tokenizer", "retrieval_stack_sha256", "architecture", "budget", "checkpoint_root",
    }
    _closed(value, allowed, "dynamic canonical recipe config", required=required)
    if value["schema"] != "rigorousrag-authoritative-dynamic-canonical-recipe-config/v1":
        raise ValueError("unsupported dynamic canonical recipe config schema")
    bundle, source_commit, train_name, validation_name, run_id, checkpoint_root, execution, trainability = _base_fields(value)
    if bundle.kind != "dynamic_rag_policy":
        raise ValueError("dynamic recipe config requires a dynamic canonical bundle")
    generator = _artifact(value["generator"], "generator", {"causal_lm", "seq2seq_lm"})
    tokenizer = _artifact(value["tokenizer"], "tokenizer", {"tokenizer"})
    collator = DynamicCollatorConfig(**_dataclass_kwargs(value.get("collator"), DynamicCollatorConfig, "collator"))
    receipt = write_dynamic_recipe_from_authoritative_bundle(
        value["bundle_path"],
        train_split_name=train_name,
        validation_split_name=validation_name,
        output_path=value["output_path"],
        run_id=run_id,
        source_commit=source_commit,
        generator=generator,
        tokenizer=tokenizer,
        retrieval_stack_sha256=value["retrieval_stack_sha256"],
        architecture=_dynamic_architecture(value["architecture"]),
        budget=_budget(value["budget"]),
        checkpoint_root=checkpoint_root,
        execution=execution,
        collator=collator,
        trainability=trainability,
        resume_checkpoint_digest=value.get("resume_checkpoint_digest"),
    )
    return {
        "kind": receipt.kind,
        "run_id": receipt.run_id,
        "config_path": receipt.output_path,
        "config_sha256": receipt.config_sha256,
        "plan_sha256": receipt.plan_sha256,
        "recipe_receipt_sha256": receipt.receipt_sha256,
        "source_commit": source_commit,
        "canonical_bundle_sha256": bundle.bundle_sha256,
    }


def run_canonical_recipe_config(path: str | Path) -> Mapping[str, Any]:
    raw = _read(path)
    schema = raw.get("schema")
    if schema == "rigorousrag-authoritative-grounded-canonical-recipe-config/v1":
        return run_grounded_canonical_recipe_config(raw)
    if schema == "rigorousrag-authoritative-dynamic-canonical-recipe-config/v1":
        return run_dynamic_canonical_recipe_config(raw)
    raise ValueError("unsupported authoritative canonical recipe config schema")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit authoritative advanced-RAG training config from a restart-verified canonical-v2 bundle"
    )
    parser.add_argument("config", help="strict authoritative canonical recipe JSON config")
    result = run_canonical_recipe_config(parser.parse_args(argv).config)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "run_canonical_recipe_config",
    "run_dynamic_canonical_recipe_config",
    "run_grounded_canonical_recipe_config",
]
