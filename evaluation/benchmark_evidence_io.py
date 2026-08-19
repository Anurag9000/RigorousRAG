"""Strict disk IO for benchmark qualification/result evidence and advanced evaluation runs.

This module is intentionally execution-free. It persists self-verifying result/leakage receipts,
reconstructs them from strict JSON, re-verifies detailed result artifacts, and writes the
``rigorousrag-advanced-evaluation-runs/v1`` envelope consumed by the authoritative advanced-RAG
operator.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.benchmark_run_evidence import BenchmarkResultArtifactReceipt, verify_benchmark_result_artifact
from evaluation.governed_benchmark_qualification import GovernedBenchmarkLeakageReceipt, LeakageFindingEvidence
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 64 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _strict_json(path: str | Path, label: str) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label=label, must_exist=True, require_file=True)
    size = source.stat().st_size
    if size <= 0 or size > _MAX_BYTES:
        raise ValueError(f"{label} exceeds byte safety bound")
    try:
        payload = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _atomic(path: str | Path, payload: Mapping[str, Any], label: str) -> None:
    destination = safe_advanced_path(path, label=label, must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError(f"{label} destination must be a file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_benchmark_result_receipt(path: str | Path, receipt: BenchmarkResultArtifactReceipt) -> None:
    if not isinstance(receipt, BenchmarkResultArtifactReceipt):
        raise ValueError("receipt must be BenchmarkResultArtifactReceipt")
    _atomic(path, {**receipt._unsigned(), "receipt_sha256": receipt.receipt_sha256}, "benchmark result receipt")


def read_benchmark_result_receipt(path: str | Path, *, verify_artifact: bool = True) -> BenchmarkResultArtifactReceipt:
    payload = _strict_json(path, "benchmark result receipt")
    required = {"schema", "benchmark_id", "benchmark_manifest_sha256", "evaluator_contract_sha256", "seed", "repeat_index", "sample_count", "result_artifact_path", "result_artifact_sha256", "metrics_sha256", "receipt_sha256"}
    if set(payload) != required or payload.get("schema") != "rigorousrag-benchmark-result-artifact-receipt/v1":
        raise ValueError("unsupported benchmark result receipt schema")
    receipt = BenchmarkResultArtifactReceipt(
        benchmark_id=payload["benchmark_id"], benchmark_manifest_sha256=payload["benchmark_manifest_sha256"], evaluator_contract_sha256=payload["evaluator_contract_sha256"],
        seed=payload["seed"], repeat_index=payload["repeat_index"], sample_count=payload["sample_count"], result_artifact_path=payload["result_artifact_path"],
        result_artifact_sha256=payload["result_artifact_sha256"], metrics_sha256=payload["metrics_sha256"], receipt_sha256=payload["receipt_sha256"],
    )
    if verify_artifact:
        verify_benchmark_result_artifact(receipt)
    return receipt


def write_benchmark_leakage_receipt(path: str | Path, receipt: GovernedBenchmarkLeakageReceipt) -> None:
    if not isinstance(receipt, GovernedBenchmarkLeakageReceipt):
        raise ValueError("receipt must be GovernedBenchmarkLeakageReceipt")
    _atomic(path, {**receipt._unsigned(), "receipt_sha256": receipt.receipt_sha256}, "benchmark leakage receipt")


def read_benchmark_leakage_receipt(path: str | Path) -> GovernedBenchmarkLeakageReceipt:
    payload = _strict_json(path, "benchmark leakage receipt")
    required = {"schema", "dataset_manifest_sha256", "import_receipt_sha256", "split_key_sha256", "blocking_key_kinds", "findings", "passed", "receipt_sha256"}
    if set(payload) != required or payload.get("schema") != "rigorousrag-governed-benchmark-leakage-receipt/v1":
        raise ValueError("unsupported benchmark leakage receipt schema")
    findings_raw = payload["findings"]
    if not isinstance(findings_raw, list):
        raise ValueError("benchmark leakage findings must be an array")
    finding_fields = {field.name for field in fields(LeakageFindingEvidence)}
    findings = []
    for index, item in enumerate(findings_raw):
        if not isinstance(item, Mapping) or set(item) != finding_fields:
            raise ValueError(f"benchmark leakage finding {index} fields are invalid")
        normalized = dict(item)
        normalized["overlap_sample"] = tuple(normalized["overlap_sample"])
        findings.append(LeakageFindingEvidence(**normalized))
    split_keys = payload["split_key_sha256"]
    if not isinstance(split_keys, Mapping):
        raise ValueError("split_key_sha256 must be an object")
    normalized_split_keys: dict[str, dict[str, str]] = {}
    for split, groups in split_keys.items():
        if not isinstance(groups, Mapping):
            raise ValueError("split_key_sha256 entries must be objects")
        normalized_split_keys[str(split)] = {str(kind): str(value) for kind, value in groups.items()}
    blocking = payload["blocking_key_kinds"]
    if not isinstance(blocking, list):
        raise ValueError("blocking_key_kinds must be an array")
    return GovernedBenchmarkLeakageReceipt(
        dataset_manifest_sha256=payload["dataset_manifest_sha256"], import_receipt_sha256=payload["import_receipt_sha256"], split_key_sha256=normalized_split_keys,
        blocking_key_kinds=tuple(blocking), findings=tuple(findings), passed=bool(payload["passed"]), receipt_sha256=payload["receipt_sha256"],
    )


def advanced_evaluation_run_from_result_receipt(receipt: BenchmarkResultArtifactReceipt) -> AdvancedEvaluationRun:
    """Re-verify detailed result bytes and return the exact AdvancedEvaluationRun."""
    return verify_benchmark_result_artifact(receipt)


def write_advanced_evaluation_runs(path: str | Path, runs: Sequence[AdvancedEvaluationRun]) -> None:
    selected = tuple(runs)
    if not selected or len(selected) > 10_000 or any(not isinstance(run, AdvancedEvaluationRun) for run in selected):
        raise ValueError("runs must be a non-empty bounded AdvancedEvaluationRun sequence")
    if len({run.run_sha256 for run in selected}) != len(selected):
        raise ValueError("advanced evaluation runs contain duplicate identities")
    _atomic(path, {"schema": "rigorousrag-advanced-evaluation-runs/v1", "runs": [asdict(run) for run in selected]}, "advanced evaluation runs")


def read_advanced_evaluation_runs(path: str | Path) -> tuple[AdvancedEvaluationRun, ...]:
    payload = _strict_json(path, "advanced evaluation runs")
    if set(payload) != {"schema", "runs"} or payload.get("schema") != "rigorousrag-advanced-evaluation-runs/v1" or not isinstance(payload.get("runs"), list):
        raise ValueError("unsupported advanced evaluation runs schema")
    allowed = {field.name for field in fields(AdvancedEvaluationRun)}
    runs = []
    for index, item in enumerate(payload["runs"]):
        if not isinstance(item, Mapping) or set(item) != allowed:
            raise ValueError(f"advanced evaluation run {index} fields are invalid")
        runs.append(AdvancedEvaluationRun(**dict(item)))
    if not runs or len({run.run_sha256 for run in runs}) != len(runs):
        raise ValueError("advanced evaluation runs must be non-empty and unique")
    return tuple(runs)


def runs_file_from_result_receipts(path: str | Path, receipts: Sequence[BenchmarkResultArtifactReceipt]) -> tuple[AdvancedEvaluationRun, ...]:
    """Re-verify one or more persisted benchmark results and emit the operator runs file."""
    selected = tuple(receipts)
    if not selected:
        raise ValueError("at least one benchmark result receipt is required")
    runs = tuple(advanced_evaluation_run_from_result_receipt(receipt) for receipt in selected)
    write_advanced_evaluation_runs(path, runs)
    return runs


__all__ = [
    "advanced_evaluation_run_from_result_receipt",
    "read_advanced_evaluation_runs",
    "read_benchmark_leakage_receipt",
    "read_benchmark_result_receipt",
    "runs_file_from_result_receipts",
    "write_advanced_evaluation_runs",
    "write_benchmark_leakage_receipt",
    "write_benchmark_result_receipt",
]
