"""Content-addressed evaluation evidence for advanced RAG artifact promotion.

This module does not run benchmarks. It turns already-produced, governed benchmark results
into immutable receipts bound to the exact checkpoint and training identity. Repeats/seeds,
benchmark manifests, evaluator contracts, sample counts and result-artifact digests are all
part of the receipt; promotion consumes only the deterministic aggregate metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

from training.advanced_rag_artifacts import (
    AdvancedArtifactManifest,
    AdvancedArtifactPromotionReceipt,
    MetricQualificationPolicy,
    qualify_advanced_artifact,
)
from training.advanced_rag_run_binding import VerifiedAdvancedCheckpointBinding

_HEX = frozenset("0123456789abcdef")
_MAX_METRICS = 10000
_MAX_RUNS = 10000
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _text(value: Any, label: str, maximum: int = 500) -> str:
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


def _metrics(value: Mapping[str, Any], label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value or len(value) > _MAX_METRICS:
        raise ValueError(f"{label} must be a bounded non-empty mapping")
    return {_text(str(key), f"{label} key", 300): _finite(metric, f"{label}[{key}]") for key, metric in value.items()}


@dataclass(frozen=True)
class AdvancedEvaluationRun:
    benchmark_id: str
    benchmark_manifest_sha256: str
    evaluator_contract_sha256: str
    seed: int
    repeat_index: int
    sample_count: int
    metrics: Mapping[str, float]
    result_artifact_sha256: str
    slice_metrics_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_id", _text(self.benchmark_id, "benchmark_id"))
        for name in ("benchmark_manifest_sha256", "evaluator_contract_sha256", "result_artifact_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.slice_metrics_sha256 is not None:
            object.__setattr__(self, "slice_metrics_sha256", _sha(self.slice_metrics_sha256, "slice_metrics_sha256"))
        for name in ("seed", "repeat_index", "sample_count"):
            value = getattr(self, name)
            minimum = 1 if name == "sample_count" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        object.__setattr__(self, "metrics", _metrics(self.metrics, "evaluation metrics"))

    @property
    def run_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-advanced-evaluation-run/v1", **asdict(self)})


@dataclass(frozen=True)
class AdvancedEvaluationReceipt:
    kind: str
    checkpoint_digest: str
    plan_sha256: str
    training_input_sha256: str
    training_config_sha256: str
    source_commit: str
    aggregation: str
    runs: tuple[AdvancedEvaluationRun, ...]
    metrics: Mapping[str, float]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced evaluation kind")
        for name in ("checkpoint_digest", "plan_sha256", "training_input_sha256", "training_config_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        commit = str(self.source_commit).strip().lower()
        if len(commit) not in {40, 64} or any(ch not in _HEX for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be mean or median")
        runs = tuple(self.runs)
        if not runs or len(runs) > _MAX_RUNS or any(not isinstance(run, AdvancedEvaluationRun) for run in runs):
            raise ValueError("evaluation receipt requires bounded AdvancedEvaluationRun values")
        if len({run.run_sha256 for run in runs}) != len(runs):
            raise ValueError("evaluation receipt contains duplicate run identities")
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "metrics", _metrics(self.metrics, "aggregate metrics"))
        expected = _digest(self._payload())
        provided = _sha(self.receipt_sha256, "receipt_sha256")
        if expected != provided:
            raise ValueError("advanced evaluation receipt digest mismatch")
        object.__setattr__(self, "receipt_sha256", provided)

    def _payload(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-advanced-evaluation-receipt/v1",
            "kind": self.kind,
            "checkpoint_digest": self.checkpoint_digest,
            "plan_sha256": self.plan_sha256,
            "training_input_sha256": self.training_input_sha256,
            "training_config_sha256": self.training_config_sha256,
            "source_commit": self.source_commit,
            "aggregation": self.aggregation,
            "runs": [{**asdict(run), "run_sha256": run.run_sha256} for run in self.runs],
            "metrics": dict(self.metrics),
        }


def aggregate_evaluation_metrics(runs: Sequence[AdvancedEvaluationRun], *, aggregation: str = "mean") -> Mapping[str, float]:
    selected = tuple(runs)
    if not selected:
        raise ValueError("cannot aggregate an empty evaluation run set")
    if aggregation not in {"mean", "median"}:
        raise ValueError("aggregation must be mean or median")
    keys = set(selected[0].metrics)
    for run in selected[1:]:
        if set(run.metrics) != keys:
            raise ValueError("all repeated evaluation runs must expose the same metric names")
    result: dict[str, float] = {}
    for key in sorted(keys):
        values = [run.metrics[key] for run in selected]
        result[key] = float(fmean(values) if aggregation == "mean" else median(values))
    return result


def build_advanced_evaluation_receipt(
    binding: VerifiedAdvancedCheckpointBinding,
    runs: Sequence[AdvancedEvaluationRun],
    *,
    aggregation: str = "mean",
) -> AdvancedEvaluationReceipt:
    if not isinstance(binding, VerifiedAdvancedCheckpointBinding):
        raise ValueError("binding must be VerifiedAdvancedCheckpointBinding")
    selected_runs = tuple(runs)
    metrics = aggregate_evaluation_metrics(selected_runs, aggregation=aggregation)
    unsigned = {
        "schema": "rigorousrag-advanced-evaluation-receipt/v1",
        "kind": binding.kind,
        "checkpoint_digest": binding.checkpoint_digest,
        "plan_sha256": binding.plan_sha256,
        "training_input_sha256": binding.training_input_sha256,
        "training_config_sha256": binding.training_config_sha256,
        "source_commit": binding.source_commit,
        "aggregation": aggregation,
        "runs": [{**asdict(run), "run_sha256": run.run_sha256} for run in selected_runs],
        "metrics": dict(metrics),
    }
    return AdvancedEvaluationReceipt(
        kind=binding.kind,
        checkpoint_digest=binding.checkpoint_digest,
        plan_sha256=binding.plan_sha256,
        training_input_sha256=binding.training_input_sha256,
        training_config_sha256=binding.training_config_sha256,
        source_commit=binding.source_commit,
        aggregation=aggregation,
        runs=selected_runs,
        metrics=metrics,
        receipt_sha256=_digest(unsigned),
    )


def assert_evaluation_matches_artifact(receipt: AdvancedEvaluationReceipt, manifest: AdvancedArtifactManifest) -> None:
    if not isinstance(receipt, AdvancedEvaluationReceipt) or not isinstance(manifest, AdvancedArtifactManifest):
        raise ValueError("receipt/manifest types are invalid")
    expected_kind = "grounded_generation" if manifest.kind == "grounded_generator" else "dynamic_rag_policy"
    fields = {
        "kind": receipt.kind == expected_kind,
        "checkpoint_digest": receipt.checkpoint_digest == manifest.checkpoint_digest,
        "plan_sha256": receipt.plan_sha256 == manifest.plan_sha256,
        "training_input_sha256": receipt.training_input_sha256 == manifest.training_input_sha256,
        "training_config_sha256": receipt.training_config_sha256 == manifest.training_config_sha256,
        "source_commit": receipt.source_commit == manifest.source_commit,
    }
    failures = [name for name, matched in fields.items() if not matched]
    if failures:
        raise ValueError(f"evaluation receipt differs from artifact lineage: {','.join(failures)}")
    if manifest.evaluation_receipt_sha256 is not None and manifest.evaluation_receipt_sha256 != receipt.receipt_sha256:
        raise ValueError("artifact is bound to a different evaluation receipt")


def qualify_advanced_artifact_with_receipt(
    manifest: AdvancedArtifactManifest,
    receipt: AdvancedEvaluationReceipt,
    policy: MetricQualificationPolicy,
) -> AdvancedArtifactPromotionReceipt:
    """Authoritative promotion path: lineage-bound evaluation receipt only."""
    assert_evaluation_matches_artifact(receipt, manifest)
    return qualify_advanced_artifact(
        manifest,
        evaluation_receipt_sha256=receipt.receipt_sha256,
        metrics=receipt.metrics,
        policy=policy,
    )


def write_advanced_evaluation_receipt(path: str | Path, receipt: AdvancedEvaluationReceipt) -> str:
    if not isinstance(receipt, AdvancedEvaluationReceipt):
        raise ValueError("receipt must be AdvancedEvaluationReceipt")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {**receipt._payload(), "receipt_sha256": receipt.receipt_sha256}
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(payload) + b"\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return receipt.receipt_sha256


def read_advanced_evaluation_receipt(path: str | Path) -> AdvancedEvaluationReceipt:
    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise ValueError("evaluation receipt path may not be a symlink")
    selected = selected.resolve(strict=True)
    if not selected.is_file() or selected.stat().st_size <= 0 or selected.stat().st_size > _MAX_RECEIPT_BYTES:
        raise ValueError("evaluation receipt must be a bounded regular file")
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
    except Exception as exc:
        raise ValueError("evaluation receipt is not strict JSON") from exc
    required = {"schema", "kind", "checkpoint_digest", "plan_sha256", "training_input_sha256", "training_config_sha256", "source_commit", "aggregation", "runs", "metrics", "receipt_sha256"}
    if not isinstance(payload, Mapping) or set(payload) != required or payload["schema"] != "rigorousrag-advanced-evaluation-receipt/v1":
        raise ValueError("unsupported or malformed evaluation receipt")
    runs = []
    for raw in payload["runs"]:
        if not isinstance(raw, Mapping) or "run_sha256" not in raw:
            raise ValueError("evaluation receipt run is malformed")
        values = {key: value for key, value in raw.items() if key != "run_sha256"}
        run = AdvancedEvaluationRun(**values)
        if run.run_sha256 != raw["run_sha256"]:
            raise ValueError("evaluation run digest mismatch")
        runs.append(run)
    return AdvancedEvaluationReceipt(
        kind=payload["kind"], checkpoint_digest=payload["checkpoint_digest"], plan_sha256=payload["plan_sha256"],
        training_input_sha256=payload["training_input_sha256"], training_config_sha256=payload["training_config_sha256"],
        source_commit=payload["source_commit"], aggregation=payload["aggregation"], runs=tuple(runs),
        metrics=payload["metrics"], receipt_sha256=payload["receipt_sha256"],
    )


__all__ = [
    "AdvancedEvaluationReceipt",
    "AdvancedEvaluationRun",
    "aggregate_evaluation_metrics",
    "assert_evaluation_matches_artifact",
    "build_advanced_evaluation_receipt",
    "qualify_advanced_artifact_with_receipt",
    "read_advanced_evaluation_receipt",
    "write_advanced_evaluation_receipt",
]
