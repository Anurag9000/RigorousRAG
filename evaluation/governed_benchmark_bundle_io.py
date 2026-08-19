"""Strict read-side verification for persisted governed benchmark bundle receipts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.governed_benchmark_bundle import AuxiliaryBenchmarkArtifact, GovernedBenchmarkBundleReceipt, build_governed_benchmark_bundle
from evaluation.governed_benchmark_io import VerifiedGovernedBenchmark
from evaluation.governed_benchmark_qualification import GovernedBenchmarkLeakageReceipt
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 64 * 1024 * 1024


def read_governed_benchmark_bundle(path: str | Path) -> GovernedBenchmarkBundleReceipt:
    source = safe_advanced_path(path, label="governed benchmark bundle receipt", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("governed benchmark bundle receipt exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except Exception as exc:
        raise ValueError("governed benchmark bundle receipt is not strict JSON") from exc
    required = {"schema", "dataset_manifest_sha256", "query_import_receipt_sha256", "leakage_receipt_sha256", "corpus_receipt_sha256", "corpus_output_sha256", "relevant_id_sha256", "relevant_id_count", "auxiliary_artifacts", "receipt_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-governed-benchmark-bundle-receipt/v1":
        raise ValueError("unsupported governed benchmark bundle receipt schema")
    auxiliary_raw = raw["auxiliary_artifacts"]
    if not isinstance(auxiliary_raw, list):
        raise ValueError("auxiliary_artifacts must be an array")
    auxiliary = []
    for index, item in enumerate(auxiliary_raw):
        if not isinstance(item, Mapping) or set(item) != {"role", "path", "sha256"}:
            raise ValueError(f"auxiliary artifact {index} fields are invalid")
        auxiliary.append(AuxiliaryBenchmarkArtifact(item["role"], item["path"], item["sha256"]))
    return GovernedBenchmarkBundleReceipt(
        dataset_manifest_sha256=raw["dataset_manifest_sha256"], query_import_receipt_sha256=raw["query_import_receipt_sha256"],
        leakage_receipt_sha256=raw["leakage_receipt_sha256"], corpus_receipt_sha256=raw["corpus_receipt_sha256"], corpus_output_sha256=raw["corpus_output_sha256"],
        relevant_id_sha256=raw["relevant_id_sha256"], relevant_id_count=raw["relevant_id_count"], auxiliary_artifacts=tuple(auxiliary), receipt_sha256=raw["receipt_sha256"],
    )


def verify_governed_benchmark_bundle(
    path: str | Path,
    *,
    benchmark: VerifiedGovernedBenchmark,
    leakage_receipt: GovernedBenchmarkLeakageReceipt,
    corpus_receipt_path: str | Path,
) -> GovernedBenchmarkBundleReceipt:
    """Rebuild coverage/auxiliary identity and require exact equality with persisted receipt."""
    persisted = read_governed_benchmark_bundle(path)
    rebuilt = build_governed_benchmark_bundle(
        benchmark,
        leakage_receipt=leakage_receipt,
        corpus_receipt_path=corpus_receipt_path,
        auxiliary_artifacts=persisted.auxiliary_artifacts,
    )
    if rebuilt.receipt_sha256 != persisted.receipt_sha256 or rebuilt != persisted:
        raise ValueError("persisted benchmark bundle differs from re-verified query/corpus/leakage/auxiliary inputs")
    return persisted


__all__ = ["read_governed_benchmark_bundle", "verify_governed_benchmark_bundle"]
