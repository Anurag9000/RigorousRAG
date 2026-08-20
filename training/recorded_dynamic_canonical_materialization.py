"""Canonical dynamic training-data materialization from one sealed runtime-cohort receipt.

This bridge removes manual provenance transcription.  ``source_shards`` and the full
``DynamicRuntimeTrainingLineage`` are derived from ``RecordedDynamicCohortReceipt``; operators
supply only target-supervision artifacts, final dataset governance/splitting and local generator /
tokenizer bindings.  The resulting call enters the same production canonical-v2 materializer.
"""
from __future__ import annotations

from typing import Any, Mapping

from training.authoritative_canonical_materialization import _closed, _mapping
from training.production_canonical_materialization import run_production_dynamic_canonical_materialization_config
from training.recorded_dynamic_cohort_authority import verify_recorded_dynamic_cohort

_SCHEMA = "rigorousrag-authoritative-recorded-dynamic-canonical-materialization-config/v1"


def run_recorded_dynamic_canonical_materialization_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping(raw, "recorded dynamic canonical materialization config")
    allowed = {
        "schema", "recorded_cohort_receipt", "output_dir", "generator", "tokenizer",
        "hidden_state_config", "information_need_sidecar_receipt", "realized_gain_sidecar_receipt",
        "logged_value_sidecar_receipt", "counterfactual_sidecar_receipt", "governance",
        "split_policy", "require_need_annotations", "bundle_output_path",
    }
    required = {
        "schema", "recorded_cohort_receipt", "output_dir", "generator", "tokenizer",
        "logged_value_sidecar_receipt", "governance", "split_policy",
    }
    _closed(value, allowed, "recorded dynamic canonical materialization config", required=required)
    if value["schema"] != _SCHEMA:
        raise ValueError("unsupported recorded dynamic canonical materialization schema")
    cohort_path = value["recorded_cohort_receipt"]
    if not isinstance(cohort_path, str) or not cohort_path.strip():
        raise ValueError("recorded_cohort_receipt must be a non-empty path string")
    cohort = verify_recorded_dynamic_cohort(cohort_path)
    delegated = {
        key: item
        for key, item in value.items()
        if key not in {"schema", "recorded_cohort_receipt"}
    }
    delegated.update({
        "schema": "rigorousrag-authoritative-dynamic-canonical-materialization-config/v1",
        "source_shards": [dict(item) for item in cohort.source_shards],
        "runtime_lineage": dict(cohort.runtime_lineage_payload),
    })
    result = dict(run_production_dynamic_canonical_materialization_config(delegated))
    result.update({
        "recorded_cohort_receipt_sha256": cohort.receipt.receipt_sha256,
        "recorded_cohort_runtime_lineage_sha256": cohort.receipt.runtime_lineage_sha256,
        "recorded_source_dataset_manifest_sha256": cohort.dataset.manifest.manifest_digest,
        "recorded_source_dataset_sha256": cohort.dataset.receipt.source_set_sha256,
    })
    return result


__all__ = ["run_recorded_dynamic_canonical_materialization_config"]
