"""Bind executed benchmark-suite results into advanced-RAG evaluation evidence.

This module does not execute a benchmark.  It consumes an already-produced
:class:`evaluation.benchmark_suite.BenchmarkSuiteResult`, independently recomputes its
aggregate metrics from detailed rows, atomically persists the detailed result artifact and
returns an :class:`evaluation.advanced_rag_receipts.AdvancedEvaluationRun` suitable for the
existing checkpoint-bound evaluation-receipt / promotion pipeline.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from evaluation.advanced_rag_receipts import AdvancedEvaluationRun
from evaluation.benchmark_suite import BenchmarkRow, BenchmarkSuiteResult
from evaluation.dataset_governance import DatasetManifest
from training.advanced_path_authority import safe_advanced_path

_HEX = frozenset("0123456789abcdef")
_MAX_ROWS = 100_000_000
_MAX_METRICS = 10_000
_MAX_TEXT = 64 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _identifier(value: Any, label: str, maximum: int = 1_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def _metric_map(value: Mapping[str, Any], label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or len(value) > _MAX_METRICS:
        raise ValueError(f"{label} must be a bounded mapping")
    return {_identifier(str(key), f"{label} key", 300): _finite(item, f"{label}[{key}]") for key, item in value.items()}


def _normalize_row(row: BenchmarkRow) -> Mapping[str, Any]:
    if not isinstance(row, BenchmarkRow):
        raise ValueError("benchmark result rows must be BenchmarkRow")
    answer = str(row.generated_answer)
    if len(answer) > _MAX_TEXT or "\x00" in answer:
        raise ValueError("generated benchmark answer exceeds safety bound or contains NUL")
    return {
        "example_id": _identifier(row.example_id, "benchmark example_id", 10_000),
        "retrieval_metrics": _metric_map(row.retrieval_metrics, "retrieval metrics"),
        "retrieval_latency_ms": _finite(row.retrieval_latency_ms, "retrieval_latency_ms"),
        "generated_answer": answer,
        "generation_latency_ms": _finite(row.generation_latency_ms, "generation_latency_ms"),
        "generation_metrics": _metric_map(row.generation_metrics, "generation metrics"),
    }


def recompute_benchmark_aggregate(rows: Sequence[BenchmarkRow]) -> Mapping[str, float]:
    """Reproduce ``run_benchmark_suite`` aggregation from detailed rows."""
    selected = tuple(rows)
    if not selected:
        raise ValueError("benchmark evidence requires at least one result row")
    if len(selected) > _MAX_ROWS:
        raise ValueError("benchmark evidence exceeds row safety bound")
    normalized = [_normalize_row(row) for row in selected]
    identifiers = [row["example_id"] for row in normalized]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("benchmark result example identities must be unique")
    aggregate: dict[str, float] = {}
    retrieval_names = sorted({name for row in normalized for name in row["retrieval_metrics"]})
    generation_names = sorted({name for row in normalized for name in row["generation_metrics"]})
    for name in retrieval_names:
        if any(name not in row["retrieval_metrics"] for row in normalized):
            raise ValueError(f"retrieval metric {name!r} is missing from some benchmark rows")
        aggregate[name] = float(fmean(row["retrieval_metrics"][name] for row in normalized))
    for name in generation_names:
        values = [row["generation_metrics"][name] for row in normalized if name in row["generation_metrics"]]
        aggregate[name] = float(fmean(values))
    aggregate["retrieval_latency_ms"] = float(fmean(row["retrieval_latency_ms"] for row in normalized))
    aggregate["generation_latency_ms"] = float(fmean(row["generation_latency_ms"] for row in normalized))
    return aggregate


def _assert_aggregate(result: BenchmarkSuiteResult, recomputed: Mapping[str, float]) -> None:
    supplied = _metric_map(result.aggregate, "benchmark aggregate")
    if set(supplied) != set(recomputed):
        raise ValueError("benchmark aggregate metric names differ from detailed-row recomputation")
    mismatches = [name for name in supplied if not math.isclose(supplied[name], recomputed[name], rel_tol=1e-12, abs_tol=1e-12)]
    if mismatches:
        raise ValueError(f"benchmark aggregate differs from detailed-row recomputation: {mismatches[:20]}")


@dataclass(frozen=True)
class BenchmarkResultArtifactReceipt:
    benchmark_id: str
    benchmark_manifest_sha256: str
    evaluator_contract_sha256: str
    seed: int
    repeat_index: int
    sample_count: int
    result_artifact_path: str
    result_artifact_sha256: str
    metrics_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_id", _identifier(self.benchmark_id, "benchmark_id"))
        for name in ("benchmark_manifest_sha256", "evaluator_contract_sha256", "result_artifact_sha256", "metrics_sha256", "receipt_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        for name in ("seed", "repeat_index", "sample_count"):
            value = getattr(self, name)
            minimum = 1 if name == "sample_count" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be integer >= {minimum}")
        if _digest(self._unsigned()) != self.receipt_sha256:
            raise ValueError("benchmark result artifact receipt digest mismatch")

    def _unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-benchmark-result-artifact-receipt/v1",
            "benchmark_id": self.benchmark_id,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "seed": self.seed,
            "repeat_index": self.repeat_index,
            "sample_count": self.sample_count,
            "result_artifact_path": self.result_artifact_path,
            "result_artifact_sha256": self.result_artifact_sha256,
            "metrics_sha256": self.metrics_sha256,
        }


def materialize_benchmark_run_evidence(
    result: BenchmarkSuiteResult,
    *,
    benchmark_manifest: DatasetManifest,
    evaluator_contract_sha256: str,
    seed: int,
    repeat_index: int,
    output_path: str | Path,
) -> tuple[AdvancedEvaluationRun, BenchmarkResultArtifactReceipt]:
    """Persist one completed benchmark result and create promotion-ready run evidence."""
    if not isinstance(result, BenchmarkSuiteResult):
        raise ValueError("result must be BenchmarkSuiteResult")
    if not isinstance(benchmark_manifest, DatasetManifest):
        raise ValueError("benchmark_manifest must be DatasetManifest")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int) or repeat_index < 0:
        raise ValueError("repeat_index must be a non-negative integer")
    evaluator_sha = _sha(evaluator_contract_sha256, "evaluator_contract_sha256")
    rows = tuple(result.rows)
    recomputed = recompute_benchmark_aggregate(rows)
    _assert_aggregate(result, recomputed)
    normalized_rows = [_normalize_row(row) for row in rows]
    payload = {
        "schema": "rigorousrag-benchmark-result-artifact/v1",
        "benchmark_id": benchmark_manifest.dataset_id,
        "benchmark_manifest_sha256": benchmark_manifest.manifest_digest,
        "evaluator_contract_sha256": evaluator_sha,
        "seed": seed,
        "repeat_index": repeat_index,
        "sample_count": len(rows),
        "metrics": dict(recomputed),
        "rows": normalized_rows,
    }
    destination = safe_advanced_path(output_path, label="benchmark result artifact", must_exist=False)
    if destination.exists() and destination.is_dir():
        raise ValueError("benchmark result artifact output must be a file path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    artifact_sha = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    with destination.open("rb") as handle:
        actual_sha = hashlib.sha256(handle.read()).hexdigest()
    if actual_sha != artifact_sha:
        raise RuntimeError("benchmark result artifact changed during publication")
    run = AdvancedEvaluationRun(
        benchmark_id=benchmark_manifest.dataset_id,
        benchmark_manifest_sha256=benchmark_manifest.manifest_digest,
        evaluator_contract_sha256=evaluator_sha,
        seed=seed,
        repeat_index=repeat_index,
        sample_count=len(rows),
        metrics=recomputed,
        result_artifact_sha256=artifact_sha,
    )
    unsigned = {
        "schema": "rigorousrag-benchmark-result-artifact-receipt/v1",
        "benchmark_id": benchmark_manifest.dataset_id,
        "benchmark_manifest_sha256": benchmark_manifest.manifest_digest,
        "evaluator_contract_sha256": evaluator_sha,
        "seed": seed,
        "repeat_index": repeat_index,
        "sample_count": len(rows),
        "result_artifact_path": str(destination),
        "result_artifact_sha256": artifact_sha,
        "metrics_sha256": _digest(dict(recomputed)),
    }
    receipt = BenchmarkResultArtifactReceipt(
        benchmark_id=benchmark_manifest.dataset_id,
        benchmark_manifest_sha256=benchmark_manifest.manifest_digest,
        evaluator_contract_sha256=evaluator_sha,
        seed=seed,
        repeat_index=repeat_index,
        sample_count=len(rows),
        result_artifact_path=str(destination),
        result_artifact_sha256=artifact_sha,
        metrics_sha256=unsigned["metrics_sha256"],
        receipt_sha256=_digest(unsigned),
    )
    return run, receipt


def verify_benchmark_result_artifact(receipt: BenchmarkResultArtifactReceipt) -> AdvancedEvaluationRun:
    """Re-hash a persisted detailed result artifact and recreate its AdvancedEvaluationRun."""
    if not isinstance(receipt, BenchmarkResultArtifactReceipt):
        raise ValueError("receipt must be BenchmarkResultArtifactReceipt")
    path = safe_advanced_path(receipt.result_artifact_path, label="benchmark result artifact", must_exist=True, require_file=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.result_artifact_sha256:
        raise ValueError("benchmark result artifact bytes differ from receipt")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("benchmark result artifact is not strict JSON") from exc
    required = {"schema", "benchmark_id", "benchmark_manifest_sha256", "evaluator_contract_sha256", "seed", "repeat_index", "sample_count", "metrics", "rows"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload.get("schema") != "rigorousrag-benchmark-result-artifact/v1":
        raise ValueError("unsupported benchmark result artifact schema")
    checks = {
        "benchmark_id": payload["benchmark_id"] == receipt.benchmark_id,
        "benchmark_manifest_sha256": payload["benchmark_manifest_sha256"] == receipt.benchmark_manifest_sha256,
        "evaluator_contract_sha256": payload["evaluator_contract_sha256"] == receipt.evaluator_contract_sha256,
        "seed": payload["seed"] == receipt.seed,
        "repeat_index": payload["repeat_index"] == receipt.repeat_index,
        "sample_count": payload["sample_count"] == receipt.sample_count,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError(f"benchmark result artifact metadata differs from receipt: {failures}")
    rows_raw = payload["rows"]
    if not isinstance(rows_raw, list) or len(rows_raw) != receipt.sample_count:
        raise ValueError("benchmark result artifact rows differ from receipt sample_count")
    rows = []
    expected_row_fields = {"example_id", "retrieval_metrics", "retrieval_latency_ms", "generated_answer", "generation_latency_ms", "generation_metrics"}
    for item in rows_raw:
        if not isinstance(item, Mapping) or set(item) != expected_row_fields:
            raise ValueError("benchmark result artifact row fields are invalid")
        rows.append(BenchmarkRow(
            example_id=item["example_id"],
            retrieval_metrics=_metric_map(item["retrieval_metrics"], "retrieval metrics"),
            retrieval_latency_ms=_finite(item["retrieval_latency_ms"], "retrieval_latency_ms"),
            generated_answer=str(item["generated_answer"]),
            generation_latency_ms=_finite(item["generation_latency_ms"], "generation_latency_ms"),
            generation_metrics=_metric_map(item["generation_metrics"], "generation metrics"),
        ))
    recomputed = recompute_benchmark_aggregate(rows)
    supplied = _metric_map(payload["metrics"], "benchmark metrics")
    if set(supplied) != set(recomputed) or any(not math.isclose(supplied[name], recomputed[name], rel_tol=1e-12, abs_tol=1e-12) for name in supplied):
        raise ValueError("benchmark result artifact aggregate differs from detailed rows")
    if _digest(dict(recomputed)) != receipt.metrics_sha256:
        raise ValueError("benchmark result metrics digest differs from receipt")
    return AdvancedEvaluationRun(
        benchmark_id=receipt.benchmark_id,
        benchmark_manifest_sha256=receipt.benchmark_manifest_sha256,
        evaluator_contract_sha256=receipt.evaluator_contract_sha256,
        seed=receipt.seed,
        repeat_index=receipt.repeat_index,
        sample_count=receipt.sample_count,
        metrics=recomputed,
        result_artifact_sha256=receipt.result_artifact_sha256,
    )


__all__ = ["BenchmarkResultArtifactReceipt", "materialize_benchmark_run_evidence", "recompute_benchmark_aggregate", "verify_benchmark_result_artifact"]
