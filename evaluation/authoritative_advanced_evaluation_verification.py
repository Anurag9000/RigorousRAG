"""Restart verification for authoritative advanced evaluation evidence.

This verifier is intentionally independent of a live checkpoint manager. It proves every
result receipt and result artifact still exists and verifies, replays every persisted result
row through the same canonical validation used at publication, rebuilds the homogeneous run
cohort and aggregate, re-reads the persisted ``AdvancedEvaluationReceipt``, and checks the
entire evidence envelope. Artifact promotion can then compare that proven lineage directly to
the exported artifact manifest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from evaluation.advanced_rag_receipts import (
    AdvancedEvaluationReceipt,
    aggregate_evaluation_metrics,
    read_advanced_evaluation_receipt,
)
from evaluation.authoritative_advanced_evaluation import (
    AuthoritativeAdvancedEvaluationEvidence,
    read_authoritative_advanced_evaluation_evidence,
)
from evaluation.strict_authoritative_benchmark_result_verification import (
    verify_strict_authoritative_benchmark_result_receipt,
)
from training.advanced_path_authority import safe_advanced_path


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_authoritative_advanced_evaluation_evidence(
    path: str | Path,
) -> tuple[AdvancedEvaluationReceipt, AuthoritativeAdvancedEvaluationEvidence]:
    evidence = read_authoritative_advanced_evaluation_evidence(path)
    evaluation_path = safe_advanced_path(
        evidence.evaluation_receipt_path,
        label="advanced evaluation receipt",
        must_exist=True,
        require_file=True,
    )
    if _file_sha(evaluation_path) != evidence.evaluation_receipt_file_sha256:
        raise ValueError("advanced evaluation receipt bytes changed after evidence publication")

    runs = []
    for item in evidence.runs:
        receipt_path = safe_advanced_path(
            item.result_receipt_path,
            label="authoritative benchmark result receipt",
            must_exist=True,
            require_file=True,
        )
        if _file_sha(receipt_path) != item.result_receipt_file_sha256:
            raise ValueError("benchmark result receipt bytes changed after evaluation publication")
        run, receipt = verify_strict_authoritative_benchmark_result_receipt(receipt_path)
        if receipt.receipt_sha256 != item.result_receipt_sha256:
            raise ValueError("benchmark result receipt identity differs from evaluation evidence")
        if receipt.result_artifact_sha256 != item.result_artifact_sha256:
            raise ValueError("benchmark result artifact differs from evaluation evidence")
        if run.run_sha256 != item.run_sha256:
            raise ValueError("benchmark run identity differs from evaluation evidence")
        runs.append(run)

    persisted = read_advanced_evaluation_receipt(evaluation_path)
    if persisted.receipt_sha256 != evidence.evaluation_receipt_sha256:
        raise ValueError("persisted advanced evaluation receipt differs from evidence")
    if len(persisted.runs) != len(runs):
        raise ValueError("persisted evaluation run count differs from verified result receipts")
    if tuple(run.run_sha256 for run in persisted.runs) != tuple(run.run_sha256 for run in runs):
        raise ValueError("persisted evaluation run order/identity differs from verified result receipts")
    aggregate = aggregate_evaluation_metrics(runs, aggregation=persisted.aggregation)
    if dict(aggregate) != dict(persisted.metrics):
        raise ValueError("persisted advanced evaluation aggregate differs from verified result artifacts")
    first = persisted.runs[0]
    checks = {
        "kind": persisted.kind == evidence.kind,
        "checkpoint_digest": persisted.checkpoint_digest == evidence.checkpoint_digest,
        "plan_sha256": persisted.plan_sha256 == evidence.plan_sha256,
        "training_input_sha256": persisted.training_input_sha256 == evidence.training_input_sha256,
        "training_config_sha256": persisted.training_config_sha256 == evidence.training_config_sha256,
        "source_commit": persisted.source_commit == evidence.source_commit,
        "aggregation": persisted.aggregation == evidence.aggregation,
        "benchmark_id": first.benchmark_id == evidence.benchmark_id,
        "benchmark_manifest_sha256": first.benchmark_manifest_sha256 == evidence.benchmark_manifest_sha256,
        "evaluator_contract_sha256": first.evaluator_contract_sha256 == evidence.evaluator_contract_sha256,
        "sample_count": first.sample_count == evidence.sample_count,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError(
            "authoritative evaluation evidence differs from verified receipt/results: "
            + ",".join(failures)
        )
    return persisted, evidence


__all__ = ["verify_authoritative_advanced_evaluation_evidence"]
