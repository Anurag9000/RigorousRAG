"""Cohort-bound production advanced evaluation evidence (v2).

Production evidence must prove two independent facts: every result artifact verifies, and the
benchmark/evaluator/sample universe those results claim is an approved authoritative cohort.
This module binds both to the exact checkpoint before artifact promotion. Historical v1
advanced-evaluation evidence remains readable for research but is not accepted by the
production verifier after this module is wired in.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.advanced_rag_receipts import (
    AdvancedEvaluationReceipt,
    build_advanced_evaluation_receipt,
    read_advanced_evaluation_receipt,
    write_advanced_evaluation_receipt,
)
from evaluation.authoritative_advanced_evaluation import AuthoritativeEvaluationRunEvidence
from evaluation.authoritative_evaluation_cohort import (
    AuthoritativeEvaluationCohortContract,
    assert_result_receipt_matches_cohort,
    verify_authoritative_evaluation_cohort,
)
from evaluation.strict_authoritative_benchmark_result_verification import (
    verify_strict_authoritative_benchmark_result_receipt,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_run_binding import VerifiedAdvancedCheckpointBinding

_MAX_BYTES = 64 * 1024 * 1024
_MAX_RUNS = 10_000
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(value: Any, label: str) -> str:
    selected = str(value).strip().lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block: break
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("cohort-bound evaluation evidence destination must be a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _assert_binding(binding: VerifiedAdvancedCheckpointBinding, evaluation: AdvancedEvaluationReceipt) -> None:
    if not isinstance(binding, VerifiedAdvancedCheckpointBinding):
        raise ValueError("binding must be VerifiedAdvancedCheckpointBinding")
    checks = {
        "kind": evaluation.kind == binding.kind,
        "checkpoint_digest": evaluation.checkpoint_digest == binding.checkpoint_digest,
        "plan_sha256": evaluation.plan_sha256 == binding.plan_sha256,
        "training_input_sha256": evaluation.training_input_sha256 == binding.training_input_sha256,
        "training_config_sha256": evaluation.training_config_sha256 == binding.training_config_sha256,
        "source_commit": evaluation.source_commit == binding.source_commit,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError("cohort-bound evaluation differs from checkpoint binding: " + ",".join(failures))


@dataclass(frozen=True)
class CohortBoundAdvancedEvaluationEvidence:
    kind: str
    checkpoint_digest: str
    plan_sha256: str
    training_input_sha256: str
    training_config_sha256: str
    source_commit: str
    cohort_contract_path: str
    cohort_contract_file_sha256: str
    cohort_contract_sha256: str
    evaluation_receipt_path: str
    evaluation_receipt_file_sha256: str
    evaluation_receipt_sha256: str
    aggregation: str
    runs: tuple[AuthoritativeEvaluationRunEvidence, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported advanced evaluation kind")
        for name in ("checkpoint_digest", "plan_sha256", "training_input_sha256", "training_config_sha256", "cohort_contract_file_sha256", "cohort_contract_sha256", "evaluation_receipt_file_sha256", "evaluation_receipt_sha256", "evidence_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        commit = str(self.source_commit).strip().lower()
        if len(commit) not in {40, 64} or any(ch not in _HEX for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)
        cohort_path = safe_advanced_path(self.cohort_contract_path, label="authoritative evaluation cohort", must_exist=True, require_file=True)
        evaluation_path = safe_advanced_path(self.evaluation_receipt_path, label="advanced evaluation receipt", must_exist=True, require_file=True)
        object.__setattr__(self, "cohort_contract_path", str(cohort_path)); object.__setattr__(self, "evaluation_receipt_path", str(evaluation_path))
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be mean or median")
        runs = tuple(self.runs)
        if not runs or len(runs) > _MAX_RUNS or any(not isinstance(item, AuthoritativeEvaluationRunEvidence) for item in runs):
            raise ValueError("runs must be bounded authoritative run evidence")
        if len({item.run_sha256 for item in runs}) != len(runs):
            raise ValueError("cohort-bound evaluation repeats a run identity")
        object.__setattr__(self, "runs", runs)
        if _digest(self.unsigned()) != self.evidence_sha256:
            raise ValueError("cohort-bound advanced evaluation evidence digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-cohort-bound-advanced-evaluation-evidence/v2",
            "kind": self.kind,
            "checkpoint_digest": self.checkpoint_digest,
            "plan_sha256": self.plan_sha256,
            "training_input_sha256": self.training_input_sha256,
            "training_config_sha256": self.training_config_sha256,
            "source_commit": self.source_commit,
            "cohort_contract_path": self.cohort_contract_path,
            "cohort_contract_file_sha256": self.cohort_contract_file_sha256,
            "cohort_contract_sha256": self.cohort_contract_sha256,
            "evaluation_receipt_path": self.evaluation_receipt_path,
            "evaluation_receipt_file_sha256": self.evaluation_receipt_file_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "aggregation": self.aggregation,
            "runs": [asdict(item) for item in self.runs],
        }


def _verified_runs(result_receipt_paths: Sequence[str | Path], cohort: AuthoritativeEvaluationCohortContract) -> tuple[tuple[Any, ...], tuple[AuthoritativeEvaluationRunEvidence, ...]]:
    selected = tuple(result_receipt_paths)
    if not selected or len(selected) > _MAX_RUNS:
        raise ValueError(f"result_receipt_paths must contain 1..{_MAX_RUNS} entries")
    runs = []; evidence = []; seen: set[str] = set()
    for raw_path in selected:
        path = safe_advanced_path(raw_path, label="authoritative benchmark result receipt", must_exist=True, require_file=True)
        if str(path) in seen:
            raise ValueError("cohort-bound evaluation repeats a result receipt path")
        seen.add(str(path))
        # Exact benchmark/evaluator/sample-universe proof; this also invokes strict result verification.
        cohort_run = assert_result_receipt_matches_cohort(path, cohort=cohort)
        run, receipt = verify_strict_authoritative_benchmark_result_receipt(path)
        if run.run_sha256 != cohort_run.run_sha256:
            raise RuntimeError("cohort/result verifiers returned different run identities")
        runs.append(run)
        evidence.append(AuthoritativeEvaluationRunEvidence(
            result_receipt_path=str(path), result_receipt_file_sha256=_file_sha(path),
            result_receipt_sha256=receipt.receipt_sha256, result_artifact_sha256=receipt.result_artifact_sha256,
            run_sha256=run.run_sha256,
        ))
    return tuple(runs), tuple(evidence)


def build_cohort_bound_advanced_evaluation_evidence(
    binding: VerifiedAdvancedCheckpointBinding,
    *,
    cohort_contract_path: str | Path,
    result_receipt_paths: Sequence[str | Path],
    aggregation: str,
    evaluation_receipt_path: str | Path,
) -> tuple[AdvancedEvaluationReceipt, CohortBoundAdvancedEvaluationEvidence]:
    cohort_path = safe_advanced_path(cohort_contract_path, label="authoritative evaluation cohort", must_exist=True, require_file=True)
    cohort = verify_authoritative_evaluation_cohort(cohort_path)
    runs, run_evidence = _verified_runs(result_receipt_paths, cohort)
    evaluation = build_advanced_evaluation_receipt(binding, runs, aggregation=aggregation)
    _assert_binding(binding, evaluation)
    first = evaluation.runs[0]
    if first.benchmark_id != cohort.benchmark_id or first.benchmark_manifest_sha256 != cohort.benchmark_manifest_sha256 or first.evaluator_contract_sha256 != cohort.evaluator_contract_sha256 or first.sample_count != cohort.sample_count:
        raise RuntimeError("advanced evaluation cohort identity differs after receipt construction")
    destination = safe_advanced_path(evaluation_receipt_path, label="advanced evaluation receipt destination", must_exist=False)
    if destination.exists():
        raise ValueError("advanced evaluation receipt destination must not already exist")
    write_advanced_evaluation_receipt(destination, evaluation)
    parsed = read_advanced_evaluation_receipt(destination)
    if parsed.receipt_sha256 != evaluation.receipt_sha256:
        raise RuntimeError("advanced evaluation receipt changed during publication")
    unsigned = {
        "schema": "rigorousrag-cohort-bound-advanced-evaluation-evidence/v2",
        "kind": parsed.kind,
        "checkpoint_digest": parsed.checkpoint_digest,
        "plan_sha256": parsed.plan_sha256,
        "training_input_sha256": parsed.training_input_sha256,
        "training_config_sha256": parsed.training_config_sha256,
        "source_commit": parsed.source_commit,
        "cohort_contract_path": str(cohort_path),
        "cohort_contract_file_sha256": _file_sha(cohort_path),
        "cohort_contract_sha256": cohort.contract_sha256,
        "evaluation_receipt_path": str(destination),
        "evaluation_receipt_file_sha256": _file_sha(destination),
        "evaluation_receipt_sha256": parsed.receipt_sha256,
        "aggregation": parsed.aggregation,
        "runs": [asdict(item) for item in run_evidence],
    }
    return parsed, CohortBoundAdvancedEvaluationEvidence(
        kind=parsed.kind, checkpoint_digest=parsed.checkpoint_digest, plan_sha256=parsed.plan_sha256,
        training_input_sha256=parsed.training_input_sha256, training_config_sha256=parsed.training_config_sha256,
        source_commit=parsed.source_commit, cohort_contract_path=str(cohort_path),
        cohort_contract_file_sha256=unsigned["cohort_contract_file_sha256"], cohort_contract_sha256=cohort.contract_sha256,
        evaluation_receipt_path=str(destination), evaluation_receipt_file_sha256=unsigned["evaluation_receipt_file_sha256"],
        evaluation_receipt_sha256=parsed.receipt_sha256, aggregation=parsed.aggregation, runs=run_evidence,
        evidence_sha256=_digest(unsigned),
    )


def write_cohort_bound_advanced_evaluation_evidence(path: str | Path, evidence: CohortBoundAdvancedEvaluationEvidence) -> None:
    if not isinstance(evidence, CohortBoundAdvancedEvaluationEvidence):
        raise ValueError("evidence must be CohortBoundAdvancedEvaluationEvidence")
    destination = safe_advanced_path(path, label="cohort-bound advanced evaluation evidence", must_exist=False)
    if destination.exists():
        raise ValueError("cohort-bound evaluation evidence destination must not already exist")
    _atomic(destination, _canonical({**evidence.unsigned(), "evidence_sha256": evidence.evidence_sha256}) + b"\n")


def read_cohort_bound_advanced_evaluation_evidence(path: str | Path) -> CohortBoundAdvancedEvaluationEvidence:
    source = safe_advanced_path(path, label="cohort-bound advanced evaluation evidence", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("cohort-bound evaluation evidence exceeds byte safety bound")
    try:
        raw = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("cohort-bound evaluation evidence is not strict JSON") from exc
    required = {"schema", "kind", "checkpoint_digest", "plan_sha256", "training_input_sha256", "training_config_sha256", "source_commit", "cohort_contract_path", "cohort_contract_file_sha256", "cohort_contract_sha256", "evaluation_receipt_path", "evaluation_receipt_file_sha256", "evaluation_receipt_sha256", "aggregation", "runs", "evidence_sha256"}
    if not isinstance(raw, Mapping) or set(raw) != required or raw.get("schema") != "rigorousrag-cohort-bound-advanced-evaluation-evidence/v2" or not isinstance(raw.get("runs"), list):
        raise ValueError("unsupported cohort-bound advanced evaluation evidence schema")
    run_fields = {"result_receipt_path", "result_receipt_file_sha256", "result_receipt_sha256", "result_artifact_sha256", "run_sha256"}
    evidence_runs = []
    for item in raw["runs"]:
        if not isinstance(item, Mapping) or set(item) != run_fields:
            raise ValueError("cohort-bound run evidence fields are invalid")
        evidence_runs.append(AuthoritativeEvaluationRunEvidence(**dict(item)))
    return CohortBoundAdvancedEvaluationEvidence(
        kind=raw["kind"], checkpoint_digest=raw["checkpoint_digest"], plan_sha256=raw["plan_sha256"],
        training_input_sha256=raw["training_input_sha256"], training_config_sha256=raw["training_config_sha256"], source_commit=raw["source_commit"],
        cohort_contract_path=raw["cohort_contract_path"], cohort_contract_file_sha256=raw["cohort_contract_file_sha256"], cohort_contract_sha256=raw["cohort_contract_sha256"],
        evaluation_receipt_path=raw["evaluation_receipt_path"], evaluation_receipt_file_sha256=raw["evaluation_receipt_file_sha256"], evaluation_receipt_sha256=raw["evaluation_receipt_sha256"],
        aggregation=raw["aggregation"], runs=tuple(evidence_runs), evidence_sha256=raw["evidence_sha256"],
    )


def verify_cohort_bound_advanced_evaluation_evidence(path: str | Path) -> tuple[AdvancedEvaluationReceipt, CohortBoundAdvancedEvaluationEvidence]:
    evidence = read_cohort_bound_advanced_evaluation_evidence(path)
    cohort_path = Path(evidence.cohort_contract_path)
    evaluation_path = Path(evidence.evaluation_receipt_path)
    if _file_sha(cohort_path) != evidence.cohort_contract_file_sha256:
        raise ValueError("evaluation cohort bytes changed after evidence publication")
    cohort = verify_authoritative_evaluation_cohort(cohort_path)
    if cohort.contract_sha256 != evidence.cohort_contract_sha256:
        raise ValueError("evaluation cohort identity changed after evidence publication")
    if _file_sha(evaluation_path) != evidence.evaluation_receipt_file_sha256:
        raise ValueError("advanced evaluation receipt bytes changed after evidence publication")
    runs = []
    for item in evidence.runs:
        receipt_path = Path(item.result_receipt_path)
        if _file_sha(receipt_path) != item.result_receipt_file_sha256:
            raise ValueError("result receipt bytes changed after evaluation publication")
        cohort_run = assert_result_receipt_matches_cohort(receipt_path, cohort=cohort)
        run, receipt = verify_strict_authoritative_benchmark_result_receipt(receipt_path)
        if run.run_sha256 != cohort_run.run_sha256 or receipt.receipt_sha256 != item.result_receipt_sha256 or receipt.result_artifact_sha256 != item.result_artifact_sha256 or run.run_sha256 != item.run_sha256:
            raise ValueError("verified result differs from cohort-bound evaluation evidence")
        runs.append(run)
    persisted = read_advanced_evaluation_receipt(evaluation_path)
    if persisted.receipt_sha256 != evidence.evaluation_receipt_sha256 or tuple(run.run_sha256 for run in persisted.runs) != tuple(run.run_sha256 for run in runs):
        raise ValueError("persisted advanced evaluation differs from verified result cohort")
    first = persisted.runs[0]
    checks = {
        "kind": persisted.kind == evidence.kind,
        "checkpoint_digest": persisted.checkpoint_digest == evidence.checkpoint_digest,
        "plan_sha256": persisted.plan_sha256 == evidence.plan_sha256,
        "training_input_sha256": persisted.training_input_sha256 == evidence.training_input_sha256,
        "training_config_sha256": persisted.training_config_sha256 == evidence.training_config_sha256,
        "source_commit": persisted.source_commit == evidence.source_commit,
        "aggregation": persisted.aggregation == evidence.aggregation,
        "benchmark_id": first.benchmark_id == cohort.benchmark_id,
        "benchmark_manifest_sha256": first.benchmark_manifest_sha256 == cohort.benchmark_manifest_sha256,
        "evaluator_contract_sha256": first.evaluator_contract_sha256 == cohort.evaluator_contract_sha256,
        "sample_count": first.sample_count == cohort.sample_count,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError("cohort-bound evaluation reconstruction differs: " + ",".join(failures))
    return persisted, evidence


__all__ = [
    "CohortBoundAdvancedEvaluationEvidence",
    "build_cohort_bound_advanced_evaluation_evidence",
    "read_cohort_bound_advanced_evaluation_evidence",
    "verify_cohort_bound_advanced_evaluation_evidence",
    "write_cohort_bound_advanced_evaluation_evidence",
]
