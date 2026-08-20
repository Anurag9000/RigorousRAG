"""Config-driven local-only canonical training-data materialization for advanced RAG.

This module closes the operator gap between governed local inputs and the authoritative Grounded
and Dynamic canonical-v2 builders.  It deliberately performs no work on import.  Explicit calls
may load *already-admitted local* model/tokenizer trees and may execute them to materialize
supervision caches, but remote model ids, network fallback, dataset download and training are not
supported.

The public ``run_*`` functions consume strict in-memory mappings so the CLI can own JSON parsing.
Every model/tokenizer is a ``LocalArtifactTreeBinding`` and is re-hashed before loading. Dynamic
trajectory corpora are exposed through a lazy Sequence facade instead of becoming Python tuples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.dataset_governance import DatasetCard, LicenseStatus
from training.authoritative_canonical_bundle_bridge import (
    write_authoritative_dynamic_canonical_bundle,
    write_authoritative_grounded_canonical_bundle,
)
from training.authoritative_dynamic_canonical_training_data import (
    build_authoritative_dynamic_canonical_training_data,
)
from training.authoritative_grounded_canonical_training_data import (
    build_authoritative_grounded_canonical_training_data,
)
from training.dynamic_canonical_training_data_pipeline import DynamicRuntimeTrainingLineage
from training.dynamic_dataset_publication import DynamicDatasetGovernance, EpisodeSplitPolicy
from training.advanced_rag_supervision import DynamicRewardConfig
from training.governed_grounded_io import verify_governed_grounded_import
from training.indexed_dynamic_step_sequence import DynamicJsonlShard, IndexedDynamicStepSequence
from training.local_artifact_loading import (
    LocalArtifactTreeBinding,
    load_local_language_model,
    load_local_tokenizer,
)
from training.local_dynamic_hidden_provider import (
    LocalDynamicHiddenStateConfig,
    LocalGeneratorHiddenStateProvider,
)
from training.local_supervision_providers import (
    LocalDocumentUtilityProvider,
    LocalLanguageModelSupervisionConfig,
    LocalSequenceReferenceProvider,
    LocalTeacherLogitProvider,
)
from training.sqlite_dynamic_supervision_sidecars import (
    SqliteCounterfactualActionProvider,
    SqliteInformationNeedAnnotationProvider,
    SqliteLoggedValueProvider,
    SqliteRealizedRetrievalGainProvider,
)


_MAX_SHARDS = 100_000


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _closed(value: Mapping[str, Any], allowed: set[str], label: str, *, required: set[str] | None = None) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    required_fields = allowed if required is None else required
    missing = required_fields - set(value)
    if missing:
        raise ValueError(f"{label} is missing required fields: {sorted(missing)}")


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _positive_int(value: Any, label: str, *, maximum: int = 1_000_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{label} must be an integer in [1,{maximum}]")
    return value


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _artifact(raw: Any, label: str, allowed_kinds: set[str]) -> LocalArtifactTreeBinding:
    value = _mapping(raw, label)
    _closed(value, {"path", "sha256", "kind"}, label)
    kind = value["kind"]
    if kind not in allowed_kinds:
        raise ValueError(f"{label}.kind must be one of {sorted(allowed_kinds)}")
    return LocalArtifactTreeBinding(
        path=value["path"],
        expected_sha256=value["sha256"],
        artifact_kind=kind,
    )


def _lm_supervision_config(raw: Any, *, generator_family: str, label: str) -> LocalLanguageModelSupervisionConfig:
    value = {} if raw is None else dict(_mapping(raw, label))
    allowed = {"max_length", "normalize_document_utility_by_tokens", "evidence_prefix", "answer_prefix"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unsupported fields: {sorted(unknown)}")
    return LocalLanguageModelSupervisionConfig(generator_family=generator_family, **value)


def _grounded_lm_provider(raw: Any, *, role: str, tokenizer: Any, tokenizer_sha256: str) -> Any | None:
    if raw is None:
        return None
    value = _mapping(raw, f"{role} provider")
    _closed(value, {"model", "config"}, f"{role} provider", required={"model"})
    model_binding = _artifact(value["model"], f"{role}.model", {"causal_lm", "seq2seq_lm"})
    model = load_local_language_model(model_binding)
    config = _lm_supervision_config(
        value.get("config"),
        generator_family=model_binding.artifact_kind,
        label=f"{role}.config",
    )
    kwargs = {
        "model_sha256": model_binding.expected_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "config": config,
    }
    if role == "teacher":
        return LocalTeacherLogitProvider(model, tokenizer, **kwargs)
    if role == "reference":
        return LocalSequenceReferenceProvider(model, tokenizer, **kwargs)
    if role == "document_utility":
        return LocalDocumentUtilityProvider(model, tokenizer, **kwargs)
    raise ValueError("unsupported grounded local supervision role")


def run_grounded_canonical_materialization_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "grounded canonical materialization config")
    allowed = {
        "schema",
        "source_receipt_path",
        "output_dir",
        "source_commit",
        "tokenizer",
        "teacher",
        "reference",
        "document_utility",
        "materialization_batch_size",
        "require_promotable_source",
        "bundle_output_path",
    }
    required = {"schema", "source_receipt_path", "output_dir", "source_commit", "tokenizer"}
    _closed(value, allowed, "grounded canonical materialization config", required=required)
    if value["schema"] != "rigorousrag-authoritative-grounded-canonical-materialization-config/v1":
        raise ValueError("unsupported grounded canonical materialization config schema")
    require_promotable = _boolean(value.get("require_promotable_source", False), "require_promotable_source")
    source = verify_governed_grounded_import(
        value["source_receipt_path"],
        require_promotable=require_promotable,
    )
    if (
        source.manifest.loader_name != "training.authoritative_governed_grounded_import"
        or source.manifest.loader_version != "2"
    ):
        raise ValueError("production canonical materialization requires authoritative grounded import v2")

    tokenizer_binding = _artifact(value["tokenizer"], "tokenizer", {"tokenizer"})
    tokenizer = load_local_tokenizer(tokenizer_binding)
    teacher = _grounded_lm_provider(
        value.get("teacher"), role="teacher", tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_binding.expected_sha256,
    )
    reference = _grounded_lm_provider(
        value.get("reference"), role="reference", tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_binding.expected_sha256,
    )
    utility = _grounded_lm_provider(
        value.get("document_utility"), role="document_utility", tokenizer=tokenizer,
        tokenizer_sha256=tokenizer_binding.expected_sha256,
    )
    batch_size = _positive_int(value.get("materialization_batch_size", 8), "materialization_batch_size", maximum=4096)
    verified = build_authoritative_grounded_canonical_training_data(
        source,
        tokenizer_sha256=tokenizer_binding.expected_sha256,
        source_commit=value["source_commit"],
        output_dir=value["output_dir"],
        teacher_provider=teacher,
        reference_provider=reference,
        document_utility_provider=utility,
        materialization_batch_size=batch_size,
    )
    bundle = None
    if value.get("bundle_output_path") is not None:
        bundle = write_authoritative_grounded_canonical_bundle(
            value["bundle_output_path"],
            Path(verified.root) / "canonical_receipt.json",
        )
    return {
        "kind": "grounded_generation",
        "canonical_root": verified.root,
        "canonical_receipt_path": str(Path(verified.root) / "canonical_receipt.json"),
        "canonical_receipt_sha256": verified.receipt.receipt_sha256,
        "dataset_manifest_sha256": verified.manifest.manifest_digest,
        "split_count": len(verified.receipt.splits),
        "record_count": sum(item.record_count for item in verified.receipt.splits),
        "cache_contracts": {item.kind: item.contract_sha256 for item in verified.receipt.caches},
        "bundle_path": None if bundle is None else str(value["bundle_output_path"]),
        "bundle_sha256": None if bundle is None else bundle.bundle_sha256,
    }


def _dataset_card(raw: Any) -> DatasetCard:
    value = _mapping(raw, "governance.card")
    required = {
        "summary", "intended_uses", "forbidden_uses", "populations_or_domains", "languages",
        "pii_notes", "safety_notes", "source_citation", "known_limitations",
    }
    _closed(value, required, "governance.card")
    return DatasetCard(
        summary=value["summary"],
        intended_uses=_strings(value["intended_uses"], "governance.card.intended_uses"),
        forbidden_uses=_strings(value["forbidden_uses"], "governance.card.forbidden_uses"),
        populations_or_domains=_strings(value["populations_or_domains"], "governance.card.populations_or_domains"),
        languages=_strings(value["languages"], "governance.card.languages"),
        pii_notes=value["pii_notes"],
        safety_notes=value["safety_notes"],
        source_citation=value["source_citation"],
        known_limitations=_strings(value["known_limitations"], "governance.card.known_limitations"),
    )


def _dynamic_governance(raw: Any) -> DynamicDatasetGovernance:
    value = _mapping(raw, "governance")
    allowed = {
        "dataset_id", "exact_version", "source_locator", "license_identifier", "license_status",
        "license_evidence", "card", "metadata", "require_promotable",
    }
    required = allowed - {"metadata", "require_promotable"}
    _closed(value, allowed, "governance", required=required)
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("governance.metadata must be an object")
    require_promotable = _boolean(value.get("require_promotable", False), "governance.require_promotable")
    return DynamicDatasetGovernance(
        dataset_id=value["dataset_id"],
        exact_version=value["exact_version"],
        source_locator=value["source_locator"],
        license_identifier=value["license_identifier"],
        license_status=LicenseStatus(value["license_status"]),
        license_evidence=value["license_evidence"],
        card=_dataset_card(value["card"]),
        metadata={str(key): str(item) for key, item in metadata.items()},
        require_promotable=require_promotable,
    )


def _split_policy(raw: Any) -> EpisodeSplitPolicy:
    value = _mapping(raw, "split_policy")
    _closed(value, {"seed", "weights"}, "split_policy")
    if not isinstance(value["weights"], Mapping):
        raise ValueError("split_policy.weights must be an object")
    return EpisodeSplitPolicy(
        seed=value["seed"],
        weights={str(key): item for key, item in value["weights"].items()},
    )


def _runtime_lineage(raw: Any) -> DynamicRuntimeTrainingLineage:
    value = _mapping(raw, "runtime_lineage")
    allowed = {
        "source_dataset_sha256", "source_dataset_manifest_sha256", "runtime_stack_sha256",
        "feature_provider_sha256", "behavior_policy_sha256", "source_commit", "reward_config",
    }
    required = allowed - {"reward_config"}
    _closed(value, allowed, "runtime_lineage", required=required)
    reward_raw = value.get("reward_config", {})
    if not isinstance(reward_raw, Mapping):
        raise ValueError("runtime_lineage.reward_config must be an object")
    reward_allowed = {"discount", "gae_lambda", "retrieval_cost", "verification_cost", "abstention_cost"}
    unknown = set(reward_raw) - reward_allowed
    if unknown:
        raise ValueError(f"runtime_lineage.reward_config contains unsupported fields: {sorted(unknown)}")
    return DynamicRuntimeTrainingLineage(
        source_dataset_sha256=value["source_dataset_sha256"],
        source_dataset_manifest_sha256=value["source_dataset_manifest_sha256"],
        runtime_stack_sha256=value["runtime_stack_sha256"],
        feature_provider_sha256=value["feature_provider_sha256"],
        behavior_policy_sha256=value["behavior_policy_sha256"],
        source_commit=value["source_commit"],
        reward_config=DynamicRewardConfig(**dict(reward_raw)),
    )


def _dynamic_shards(raw: Any, *, runtime_manifest_sha256: str) -> IndexedDynamicStepSequence:
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_SHARDS:
        raise ValueError(f"source_shards must contain 1..{_MAX_SHARDS} entries")
    shards = []
    for index, item in enumerate(raw):
        value = _mapping(item, f"source_shards[{index}]")
        allowed = {"path", "sha256", "dataset_manifest_sha256", "split_name", "expected_record_count"}
        required = allowed - {"expected_record_count"}
        _closed(value, allowed, f"source_shards[{index}]", required=required)
        if value["dataset_manifest_sha256"] != runtime_manifest_sha256:
            raise ValueError(f"source_shards[{index}] manifest identity differs from runtime lineage")
        shards.append(
            DynamicJsonlShard(
                path=value["path"],
                content_sha256=value["sha256"],
                dataset_manifest_sha256=value["dataset_manifest_sha256"],
                split_name=value["split_name"],
                expected_record_count=_optional_positive_int(
                    value.get("expected_record_count"),
                    f"source_shards[{index}].expected_record_count",
                ),
            )
        )
    return IndexedDynamicStepSequence(tuple(shards))


def _optional_sidecar(raw: Any, provider_type: Any, label: str) -> Any | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a sidecar_receipt.json path string")
    return provider_type(raw)


def run_dynamic_canonical_materialization_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "dynamic canonical materialization config")
    allowed = {
        "schema", "source_shards", "output_dir", "generator", "tokenizer", "hidden_state_config",
        "information_need_sidecar_receipt", "realized_gain_sidecar_receipt", "logged_value_sidecar_receipt",
        "counterfactual_sidecar_receipt", "runtime_lineage", "governance", "split_policy",
        "require_need_annotations", "bundle_output_path",
    }
    required = {
        "schema", "source_shards", "output_dir", "generator", "tokenizer",
        "logged_value_sidecar_receipt", "runtime_lineage", "governance", "split_policy",
    }
    _closed(value, allowed, "dynamic canonical materialization config", required=required)
    if value["schema"] != "rigorousrag-authoritative-dynamic-canonical-materialization-config/v1":
        raise ValueError("unsupported dynamic canonical materialization config schema")

    lineage = _runtime_lineage(value["runtime_lineage"])
    steps = _dynamic_shards(value["source_shards"], runtime_manifest_sha256=lineage.source_dataset_manifest_sha256)
    generator_binding = _artifact(value["generator"], "generator", {"causal_lm", "seq2seq_lm"})
    tokenizer_binding = _artifact(value["tokenizer"], "tokenizer", {"tokenizer"})
    tokenizer = load_local_tokenizer(tokenizer_binding)
    generator = load_local_language_model(generator_binding)
    hidden_raw = {} if value.get("hidden_state_config") is None else dict(_mapping(value["hidden_state_config"], "hidden_state_config"))
    hidden_allowed = {"max_length", "pooling", "pad_to_multiple_of"}
    unknown_hidden = set(hidden_raw) - hidden_allowed
    if unknown_hidden:
        raise ValueError(f"hidden_state_config contains unsupported fields: {sorted(unknown_hidden)}")
    hidden_config = LocalDynamicHiddenStateConfig(
        generator_family=generator_binding.artifact_kind,
        **hidden_raw,
    )
    hidden_provider = LocalGeneratorHiddenStateProvider(
        generator,
        tokenizer,
        generator_sha256=generator_binding.expected_sha256,
        tokenizer_sha256=tokenizer_binding.expected_sha256,
        config=hidden_config,
    )

    annotation = _optional_sidecar(
        value.get("information_need_sidecar_receipt"),
        SqliteInformationNeedAnnotationProvider,
        "information_need_sidecar_receipt",
    )
    gain = _optional_sidecar(
        value.get("realized_gain_sidecar_receipt"),
        SqliteRealizedRetrievalGainProvider,
        "realized_gain_sidecar_receipt",
    )
    logged_value_path = value["logged_value_sidecar_receipt"]
    if not isinstance(logged_value_path, str) or not logged_value_path.strip():
        raise ValueError("logged_value_sidecar_receipt must be a sidecar_receipt.json path string")
    logged_value = SqliteLoggedValueProvider(logged_value_path)
    counterfactual = _optional_sidecar(
        value.get("counterfactual_sidecar_receipt"),
        SqliteCounterfactualActionProvider,
        "counterfactual_sidecar_receipt",
    )
    require_need = _boolean(value.get("require_need_annotations", True), "require_need_annotations")
    if require_need and annotation is None:
        raise ValueError("require_need_annotations=true requires information_need_sidecar_receipt")

    governance = _dynamic_governance(value["governance"])
    policy = _split_policy(value["split_policy"])
    verified = build_authoritative_dynamic_canonical_training_data(
        steps,
        hidden_provider=hidden_provider,
        annotation_provider=annotation,
        realized_gain_provider=gain,
        value_provider=logged_value,
        counterfactual_provider=counterfactual,
        runtime_lineage=lineage,
        governance=governance,
        split_policy=policy,
        output_dir=value["output_dir"],
        require_need_annotations=require_need,
    )
    bundle = None
    if value.get("bundle_output_path") is not None:
        bundle = write_authoritative_dynamic_canonical_bundle(
            value["bundle_output_path"],
            Path(verified.root) / "canonical_receipt.json",
        )
    return {
        "kind": "dynamic_rag_policy",
        "canonical_root": verified.root,
        "canonical_receipt_path": str(Path(verified.root) / "canonical_receipt.json"),
        "canonical_receipt_sha256": verified.receipt.receipt_sha256,
        "dataset_manifest_sha256": verified.dataset.manifest.manifest_digest,
        "split_count": len(verified.dataset.receipt.splits),
        "record_count": verified.receipt.materialized_record_count,
        "episode_count": verified.receipt.materialized_episode_count,
        "hidden_cache_contract_sha256": verified.receipt.hidden_cache_contract_sha256,
        "bundle_path": None if bundle is None else str(value["bundle_output_path"]),
        "bundle_sha256": None if bundle is None else bundle.bundle_sha256,
    }


__all__ = [
    "run_dynamic_canonical_materialization_config",
    "run_grounded_canonical_materialization_config",
]
