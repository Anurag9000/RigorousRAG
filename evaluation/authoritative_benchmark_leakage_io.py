"""Strict read/recomputation authority for v2 governed benchmark leakage receipts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.authoritative_governed_benchmark_io import (
    VerifiedAuthoritativeGovernedBenchmark,
)
from evaluation.authoritative_governed_benchmark_qualification import (
    qualify_authoritative_governed_benchmark_leakage,
)
from evaluation.governed_benchmark_qualification import (
    GovernedBenchmarkLeakageReceipt,
    LeakageFindingEvidence,
)
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 64 * 1024 * 1024


def read_authoritative_benchmark_leakage_receipt(
    path: str | Path,
) -> GovernedBenchmarkLeakageReceipt:
    source = safe_advanced_path(
        path,
        label="authoritative benchmark leakage receipt",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("benchmark leakage receipt exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("benchmark leakage receipt is not strict JSON") from exc
    required = {
        "schema",
        "dataset_manifest_sha256",
        "import_receipt_sha256",
        "split_key_sha256",
        "blocking_key_kinds",
        "findings",
        "passed",
        "receipt_sha256",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema") != "rigorousrag-governed-benchmark-leakage-receipt/v1"
    ):
        raise ValueError("unsupported benchmark leakage receipt schema")
    split_keys = raw["split_key_sha256"]
    if not isinstance(split_keys, Mapping):
        raise ValueError("split_key_sha256 must be an object")
    normalized_keys: dict[str, dict[str, str]] = {}
    for split, groups in split_keys.items():
        if not isinstance(groups, Mapping):
            raise ValueError("split leakage key groups must be objects")
        normalized_keys[str(split)] = {
            str(kind): str(value) for kind, value in groups.items()
        }
    blocking = raw["blocking_key_kinds"]
    findings_raw = raw["findings"]
    if not isinstance(blocking, list) or not isinstance(findings_raw, list):
        raise ValueError("blocking_key_kinds/findings must be arrays")
    finding_fields = {
        "left_split",
        "right_split",
        "key_kind",
        "severity",
        "overlap_count",
        "overlap_sha256",
        "overlap_sample",
    }
    findings = []
    for item in findings_raw:
        if not isinstance(item, Mapping) or set(item) != finding_fields:
            raise ValueError("benchmark leakage finding fields are invalid")
        if not isinstance(item["overlap_sample"], list):
            raise ValueError("benchmark leakage overlap_sample must be an array")
        findings.append(
            LeakageFindingEvidence(
                left_split=item["left_split"],
                right_split=item["right_split"],
                key_kind=item["key_kind"],
                severity=item["severity"],
                overlap_count=item["overlap_count"],
                overlap_sha256=item["overlap_sha256"],
                overlap_sample=tuple(item["overlap_sample"]),
            )
        )
    return GovernedBenchmarkLeakageReceipt(
        dataset_manifest_sha256=raw["dataset_manifest_sha256"],
        import_receipt_sha256=raw["import_receipt_sha256"],
        split_key_sha256=normalized_keys,
        blocking_key_kinds=tuple(blocking),
        findings=tuple(findings),
        passed=raw["passed"],
        receipt_sha256=raw["receipt_sha256"],
    )


def verify_authoritative_benchmark_leakage_receipt(
    path: str | Path,
    *,
    benchmark: VerifiedAuthoritativeGovernedBenchmark,
    require_pass: bool = True,
) -> GovernedBenchmarkLeakageReceipt:
    if not isinstance(benchmark, VerifiedAuthoritativeGovernedBenchmark):
        raise ValueError("benchmark must be VerifiedAuthoritativeGovernedBenchmark")
    if not isinstance(require_pass, bool):
        raise ValueError("require_pass must be boolean")
    persisted = read_authoritative_benchmark_leakage_receipt(path)
    if (
        persisted.dataset_manifest_sha256 != benchmark.manifest.manifest_digest
        or persisted.import_receipt_sha256 != benchmark.receipt.receipt_sha256
    ):
        raise ValueError("leakage receipt is bound to a different authoritative benchmark")
    recomputed = qualify_authoritative_governed_benchmark_leakage(
        benchmark,
        blocking_key_kinds=persisted.blocking_key_kinds,
        require_pass=require_pass,
    )
    if recomputed.receipt_sha256 != persisted.receipt_sha256:
        raise ValueError("persisted leakage receipt differs from independent recomputation")
    if require_pass and not persisted.passed:
        raise ValueError("benchmark leakage receipt is not passed")
    return persisted


__all__ = [
    "read_authoritative_benchmark_leakage_receipt",
    "verify_authoritative_benchmark_leakage_receipt",
]
