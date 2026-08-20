"""Production canonical-v2 materialization entry points.

This thin production layer keeps historical materialization helpers readable while enforcing the
repository-wide split ceiling and using worker-local SQLite sidecar providers for dynamic data.
It reuses the exact canonical builders and local artifact/provider implementations; only operator
admission/performance behavior differs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from training.authoritative_canonical_bundle_bridge import write_authoritative_dynamic_canonical_bundle
from training.authoritative_canonical_materialization import (
    _artifact,
    _boolean,
    _closed,
    _dynamic_governance,
    _dynamic_shards,
    _mapping,
    _runtime_lineage,
    _split_policy,
    run_grounded_canonical_materialization_config,
)
from training.authoritative_dynamic_canonical_training_data import build_authoritative_dynamic_canonical_training_data
from training.local_artifact_loading import load_local_language_model, load_local_tokenizer
from training.local_dynamic_hidden_provider import LocalDynamicHiddenStateConfig, LocalGeneratorHiddenStateProvider
from training.production_canonical_limits import (
    assert_production_split_count,
    grounded_source_split_count_from_receipt,
)
from training.worker_local_dynamic_sidecars import (
    WorkerLocalCounterfactualActionProvider,
    WorkerLocalInformationNeedAnnotationProvider,
    WorkerLocalLoggedValueProvider,
    WorkerLocalRealizedRetrievalGainProvider,
)


def _sidecar(raw: Any, provider_type: Any, label: str, *, required: bool = False) -> Any | None:
    if raw is None:
        if required:
            raise ValueError(f"{label} is required")
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a sidecar_receipt.json path string")
    return provider_type(raw)


def _close_sidecars(*providers: Any | None) -> None:
    first_error: Exception | None = None
    for provider in providers:
        if provider is None:
            continue
        closer = getattr(provider, "close", None)
        if not callable(closer):
            continue
        try:
            closer()
        except Exception as exc:  # cleanup must attempt every provider before surfacing failure.
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise RuntimeError("failed to close one or more dynamic supervision sidecars") from first_error


def run_production_grounded_canonical_materialization_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "production grounded canonical materialization config")
    receipt_path = value.get("source_receipt_path")
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        raise ValueError("grounded canonical materialization requires source_receipt_path")
    assert_production_split_count(
        grounded_source_split_count_from_receipt(receipt_path),
        label="grounded source split count",
    )
    result = run_grounded_canonical_materialization_config(value)
    assert_production_split_count(int(result["split_count"]), label="grounded canonical output split count")
    return result


def run_production_dynamic_canonical_materialization_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "production dynamic canonical materialization config")
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
    _closed(value, allowed, "production dynamic canonical materialization config", required=required)
    if value["schema"] != "rigorousrag-authoritative-dynamic-canonical-materialization-config/v1":
        raise ValueError("unsupported dynamic canonical materialization config schema")

    lineage = _runtime_lineage(value["runtime_lineage"])
    steps = _dynamic_shards(
        value["source_shards"],
        runtime_manifest_sha256=lineage.source_dataset_manifest_sha256,
    )
    generator_binding = _artifact(value["generator"], "generator", {"causal_lm", "seq2seq_lm"})
    tokenizer_binding = _artifact(value["tokenizer"], "tokenizer", {"tokenizer"})
    tokenizer = load_local_tokenizer(tokenizer_binding)
    generator = load_local_language_model(generator_binding)

    hidden_raw = {} if value.get("hidden_state_config") is None else dict(_mapping(value["hidden_state_config"], "hidden_state_config"))
    hidden_allowed = {"max_length", "pooling", "pad_to_multiple_of"}
    unknown_hidden = set(hidden_raw) - hidden_allowed
    if unknown_hidden:
        raise ValueError(f"hidden_state_config contains unsupported fields: {sorted(unknown_hidden)}")
    hidden_provider = LocalGeneratorHiddenStateProvider(
        generator,
        tokenizer,
        generator_sha256=generator_binding.expected_sha256,
        tokenizer_sha256=tokenizer_binding.expected_sha256,
        config=LocalDynamicHiddenStateConfig(
            generator_family=generator_binding.artifact_kind,
            **hidden_raw,
        ),
    )

    annotation = _sidecar(
        value.get("information_need_sidecar_receipt"),
        WorkerLocalInformationNeedAnnotationProvider,
        "information_need_sidecar_receipt",
    )
    gain = _sidecar(
        value.get("realized_gain_sidecar_receipt"),
        WorkerLocalRealizedRetrievalGainProvider,
        "realized_gain_sidecar_receipt",
    )
    logged_value = _sidecar(
        value.get("logged_value_sidecar_receipt"),
        WorkerLocalLoggedValueProvider,
        "logged_value_sidecar_receipt",
        required=True,
    )
    counterfactual = _sidecar(
        value.get("counterfactual_sidecar_receipt"),
        WorkerLocalCounterfactualActionProvider,
        "counterfactual_sidecar_receipt",
    )
    try:
        require_need = _boolean(value.get("require_need_annotations", True), "require_need_annotations")
        if require_need and annotation is None:
            raise ValueError("require_need_annotations=true requires information_need_sidecar_receipt")

        governance = _dynamic_governance(value["governance"])
        policy = _split_policy(value["split_policy"])
        assert_production_split_count(len(policy.weights), label="dynamic split-policy count")
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
        assert_production_split_count(len(verified.dataset.receipt.splits), label="dynamic canonical output split count")
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
            "sidecar_lookup_authority": "worker_local_immutable_sqlite/v1",
        }
    finally:
        _close_sidecars(annotation, gain, logged_value, counterfactual)


__all__ = [
    "run_production_dynamic_canonical_materialization_config",
    "run_production_grounded_canonical_materialization_config",
]
