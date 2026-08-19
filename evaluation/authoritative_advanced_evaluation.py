"""Promotion-grade advanced evaluation evidence built only from verified v2 result artifacts.

``AdvancedEvaluationReceipt`` is the compact checkpoint-bound statistical receipt.  This
module supplies the stronger production evidence envelope: every embedded run is reconstructed
from a byte-verified authoritative benchmark-result receipt, the result receipt file itself is
hashed, and the resulting homogeneous cohort is rebound to the exact advanced checkpoint.

Raw ``AdvancedEvaluationRun`` JSON remains useful for research/reporting, but production
promotion can require this authority and therefore prove the detailed result artifacts exist.
No benchmark, model, download, training or inference executes on import.
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
from evaluation.authoritative_benchmark_run_evidence import (
    AuthoritativeBenchmarkResultReceipt,
    verify_authoritative_benchmark_result_receipt,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_run_binding import VerifiedAdvancedCheckpointBinding

_MAX_BYTES = 64 * 1024 * 1024
_MAX_RUNS = 10_000
_HEX = frozenset("0123456789abcdef")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_dir():
        raise ValueError("authoritative evaluation evidence destination must be a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class AuthoritativeEvaluationRunEvidence:
    result_receipt_path: str
    result_receipt_file_sha256: str
    result_receipt_sha256: str
    result_artifact_sha256: str
    run_sha256: str

    def __post_init__(self) -> None:
        source = safe_advanced_path(
            self.result_receipt_path,
            label="authoritative result receipt",
            must_exist=True,
            require_file=True,
        )
        object.__setattr__(self, "result_receipt_path", str(source))
        for name in (
            "result_receipt_file_sha256",
            "result_receipt_sha256",
            "result_artifact_sha256",
            "run_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))


@dataclass(frozen=True)
class AuthoritativeAdvancedEvaluationEvidence:
    kind: str
    checkpoint_digest: str
    plan_sha256: str
    training_input_sha256: str
    training_config_sha256: str
    source_commit: str
    evaluation_receipt_path: str
    evaluation_receipt_file_sha256: str
    evaluation_receipt_sha256: str
    aggregation: str
    benchmark_id: str
    benchmark_manifest_sha256: str
    evaluator_contract_sha256: str
    sample_count: int
    runs: tuple[AuthoritativeEvaluationRunEvidence, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"grounded_generation", "dynamic_rag_policy"}:
            raise ValueError("unsupported authoritative evaluation kind")
        for name in (
            "checkpoint_digest",
            "plan_sha256",
            "training_input_sha256",
            "training_config_sha256",
            "evaluation_receipt_file_sha256",
            "evaluation_receipt_sha256",
            "benchmark_manifest_sha256",
            "evaluator_contract_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        commit = str(self.source_commit).strip().lower()
        if len(commit) not in {40, 64} or any(ch not in _HEX for ch in commit):
            raise ValueError("source_commit must be a full Git object id")
        object.__setattr__(self, "source_commit", commit)
        receipt_path = safe_advanced_path(
            self.evaluation_receipt_path,
            label="advanced evaluation receipt",
            must_exist=True,
            require_file=True,
        )
        object.__setattr__(self, "evaluation_receipt_path", str(receipt_path))
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be mean or median")
        if not isinstance(self.benchmark_id, str) or not self.benchmark_id.strip():
            raise ValueError("benchmark_id must be non-empty")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        runs = tuple(self.runs)
        if not runs or len(runs) > _MAX_RUNS or any(
            not isinstance(item, AuthoritativeEvaluationRunEvidence) for item in runs
        ):
            raise ValueError("runs must be a bounded non-empty authoritative run sequence")
        if len({item.run_sha256 for item in runs}) != len(runs):
            raise ValueError("authoritative evaluation evidence repeats a run identity")
        object.__setattr__(self, "runs", runs)
        if _digest(self.unsigned()) != self.evidence_sha256:
            raise ValueError("authoritative advanced evaluation evidence digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-authoritative-advanced-evaluation-evidence/v1",
            "kind": self.kind,
            "checkpoint_digest": self.checkpoint_digest,
            "plan_sha256": self.plan_sha256,
            "training_input_sha256": self.training_input_sha256,
            "training_config_sha256": self.training_config_sha256,
            "source_commit": self.source_commit,
            "evaluation_receipt_path": self.evaluation_receipt_path,
            "evaluation_receipt_file_sha256": self.evaluation_receipt_file_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "aggregation": self.aggregation,
            "benchmark_id": self.benchmark_id,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
            "sample_count": self.sample_count,
            "runs": [asdict(item) for item in self.runs],
        }


def _assert_binding(
    binding: VerifiedAdvancedCheckpointBinding,
    evaluation: AdvancedEvaluationReceipt,
) -> None:
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
        raise ValueError(
            "authoritative evaluation differs from checkpoint binding: "
            + ",".join(failures)
        )


def _verified_runs(
    result_receipt_paths: Sequence[str | Path],
) -> tuple[tuple[Any, ...], tuple[AuthoritativeEvaluationRunEvidence, ...]]:
    selected = tuple(result_receipt_paths)
    if not selected or len(selected) > _MAX_RUNS:
        raise ValueError(f"result_receipt_paths must contain 1..{_MAX_RUNS} entries")
    runs = []
    evidence = []
    seen_paths: set[str] = set()
    for raw_path in selected:
        path = safe_advanced_path(
            raw_path,
            label="authoritative benchmark result receipt",
            must_exist=True,
            require_file=True,
        )
        if str(path) in seen_paths:
            raise ValueError("authoritative evaluation repeats a result receipt path")
        seen_paths.add(str(path))
        run, receipt = verify_authoritative_benchmark_result_receipt(path)
        if not isinstance(receipt, AuthoritativeBenchmarkResultReceipt):
            raise RuntimeError("result verifier returned an unexpected receipt type")
        runs.append(run)
        evidence.append(
            AuthoritativeEvaluationRunEvidence(
                result_receipt_path=str(path),
                result_receipt_file_sha256=_file_sha(path),
                result_receipt_sha256=receipt.receipt_sha256,
                result_artifact_sha256=receipt.result_artifact_sha256,
                run_sha256=run.run_sha256,
            )
        )
    return tuple(runs), tuple(evidence)


def build_authoritative_advanced_evaluation_evidence(
    binding: VerifiedAdvancedCheckpointBinding,
    *,
    result_receipt_paths: Sequence[str | Path],
    aggregation: str,
    evaluation_receipt_path: str | Path,
) -> tuple[AdvancedEvaluationReceipt, AuthoritativeAdvancedEvaluationEvidence]:
    runs, run_evidence = _verified_runs(result_receipt_paths)
    evaluation = build_advanced_evaluation_receipt(
        binding,
        runs,
        aggregation=aggregation,
    )
    _assert_binding(binding, evaluation)
    destination = safe_advanced_path(
        evaluation_receipt_path,
        label="advanced evaluation receipt destination",
        must_exist=False,
    )
    if destination.exists():
        raise ValueError("authoritative evaluation receipt destination must not already exist")
    write_advanced_evaluation_receipt(destination, evaluation)
    parsed = read_advanced_evaluation_receipt(destination)
    if parsed.receipt_sha256 != evaluation.receipt_sha256:
        raise RuntimeError("advanced evaluation receipt changed during publication")
    first = parsed.runs[0]
    unsigned = {
        "schema": "rigorousrag-authoritative-advanced-evaluation-evidence/v1",
        "kind": parsed.kind,
        "checkpoint_digest": parsed.checkpoint_digest,
        "plan_sha256": parsed.plan_sha256,
        "training_input_sha256": parsed.training_input_sha256,
        "training_config_sha256": parsed.training_config_sha256,
        "source_commit": parsed.source_commit,
        "evaluation_receipt_path": str(destination),
        "evaluation_receipt_file_sha256": _file_sha(destination),
        "evaluation_receipt_sha256": parsed.receipt_sha256,
        "aggregation": parsed.aggregation,
        "benchmark_id": first.benchmark_id,
        "benchmark_manifest_sha256": first.benchmark_manifest_sha256,
        "evaluator_contract_sha256": first.evaluator_contract_sha256,
        "sample_count": first.sample_count,
        "runs": [asdict(item) for item in run_evidence],
    }
    evidence = AuthoritativeAdvancedEvaluationEvidence(
        **{key: value for key, value in unsigned.items() if key != "schema"},
        evidence_sha256=_digest(unsigned),
    )
    return parsed, evidence


def write_authoritative_advanced_evaluation_evidence(
    path: str | Path,
    evidence: AuthoritativeAdvancedEvaluationEvidence,
) -> None:
    if not isinstance(evidence, AuthoritativeAdvancedEvaluationEvidence):
        raise ValueError("evidence must be AuthoritativeAdvancedEvaluationEvidence")
    destination = safe_advanced_path(
        path,
        label="authoritative advanced evaluation evidence",
        must_exist=False,
    )
    if destination.exists():
        raise ValueError("authoritative advanced evaluation evidence destination must not already exist")
    _atomic(
        destination,
        _canonical({**evidence.unsigned(), "evidence_sha256": evidence.evidence_sha256})
        + b"\n",
    )


def read_authoritative_advanced_evaluation_evidence(
    path: str | Path,
) -> AuthoritativeAdvancedEvaluationEvidence:
    source = safe_advanced_path(
        path,
        label="authoritative advanced evaluation evidence",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("authoritative advanced evaluation evidence exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("authoritative advanced evaluation evidence is not strict JSON") from exc
    required = {
        "schema",
        "kind",
        "checkpoint_digest",
        "plan_sha256",
        "training_input_sha256",
        "training_config_sha256",
        "source_commit",
        "evaluation_receipt_path",
        "evaluation_receipt_file_sha256",
        "evaluation_receipt_sha256",
        "aggregation",
        "benchmark_id",
        "benchmark_manifest_sha256",
        "evaluator_contract_sha256",
        "sample_count",
        "runs",
        "evidence_sha256",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema") != "rigorousrag-authoritative-advanced-evaluation-evidence/v1"
    ):
        raise ValueError("unsupported authoritative advanced evaluation evidence schema")
    runs_raw = raw["runs"]
    if not isinstance(runs_raw, list):
        raise ValueError("authoritative evaluation runs must be an array")
    run_fields = {
        "result_receipt_path",
        "result_receipt_file_sha256",
        "result_receipt_sha256",
        "result_artifact_sha256",
        "run_sha256",
    }
    run_evidence = []
    for item in runs_raw:
        if not isinstance(item, Mapping) or set(item) != run_fields:
            raise ValueError("authoritative evaluation run evidence fields are invalid")
        run_evidence.append(AuthoritativeEvaluationRunEvidence(**dict(item)))
    return AuthoritativeAdvancedEvaluationEvidence(
        **{
            key: value
            for key, value in raw.items()
            if key not in {"schema", "runs"}
        },
        runs=tuple(run_evidence),
    )


def reconstruct_authoritative_advanced_evaluation(
    path: str | Path,
    *,
    binding: VerifiedAdvancedCheckpointBinding,
) -> tuple[AdvancedEvaluationReceipt, AuthoritativeAdvancedEvaluationEvidence]:
    evidence = read_authoritative_advanced_evaluation_evidence(path)
    if _file_sha(Path(evidence.evaluation_receipt_path)) != evidence.evaluation_receipt_file_sha256:
        raise ValueError("advanced evaluation receipt bytes changed after evidence publication")
    paths = []
    for item in evidence.runs:
        receipt_path = Path(item.result_receipt_path)
        if _file_sha(receipt_path) != item.result_receipt_file_sha256:
            raise ValueError("benchmark result receipt bytes changed after evaluation publication")
        run, receipt = verify_authoritative_benchmark_result_receipt(receipt_path)
        if (
            receipt.receipt_sha256 != item.result_receipt_sha256
            or receipt.result_artifact_sha256 != item.result_artifact_sha256
            or run.run_sha256 != item.run_sha256
        ):
            raise ValueError("verified benchmark result differs from authoritative evaluation evidence")
        paths.append(receipt_path)
    runs, rebuilt_run_evidence = _verified_runs(paths)
    rebuilt = build_advanced_evaluation_receipt(
        binding,
        runs,
        aggregation=evidence.aggregation,
    )
    persisted = read_advanced_evaluation_receipt(evidence.evaluation_receipt_path)
    _assert_binding(binding, persisted)
    if rebuilt.receipt_sha256 != persisted.receipt_sha256:
        raise ValueError("reconstructed advanced evaluation differs from persisted receipt")
    first = rebuilt.runs[0]
    checks = {
        "evaluation_receipt_sha256": rebuilt.receipt_sha256 == evidence.evaluation_receipt_sha256,
        "benchmark_id": first.benchmark_id == evidence.benchmark_id,
        "benchmark_manifest_sha256": first.benchmark_manifest_sha256 == evidence.benchmark_manifest_sha256,
        "evaluator_contract_sha256": first.evaluator_contract_sha256 == evidence.evaluator_contract_sha256,
        "sample_count": first.sample_count == evidence.sample_count,
        "run_evidence": rebuilt_run_evidence == evidence.runs,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError(
            "reconstructed authoritative evaluation differs from evidence: "
            + ",".join(failures)
        )
    return persisted, evidence


__all__ = [
    "AuthoritativeAdvancedEvaluationEvidence",
    "AuthoritativeEvaluationRunEvidence",
    "build_authoritative_advanced_evaluation_evidence",
    "read_authoritative_advanced_evaluation_evidence",
    "reconstruct_authoritative_advanced_evaluation",
    "write_authoritative_advanced_evaluation_evidence",
]
