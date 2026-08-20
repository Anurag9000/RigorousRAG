"""Production advanced-evaluation evidence with persisted evaluator provenance.

The v3 envelope binds checkpoint/cohort/result evidence to an
``EvaluatorBoundEvaluationCohort``. Production construction and restart verification also
require the evaluator semantics implemented by the result authority: one row per authorized
sample, row+aggregate metrics, and arithmetic-mean aggregation over the exact cohort.
Historical v1/v2 evidence remains available through its original compatibility modules.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.advanced_rag_receipts import AdvancedEvaluationReceipt
from evaluation.cohort_bound_advanced_evaluation import (
    CohortBoundAdvancedEvaluationEvidence,
    build_cohort_bound_advanced_evaluation_evidence,
    verify_cohort_bound_advanced_evaluation_evidence,
    write_cohort_bound_advanced_evaluation_evidence,
)
from evaluation.evaluator_bound_evaluation_cohort import (
    verify_evaluator_bound_evaluation_cohort,
    verify_result_against_evaluator_bound_cohort,
)
from evaluation.strict_production_evaluator_contract import (
    assert_strict_production_evaluator_contract,
)
from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_run_binding import VerifiedAdvancedCheckpointBinding

_MAX_BYTES = 32 * 1024 * 1024
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
    if path.exists():
        raise ValueError("evaluator-bound advanced evaluation destination must not already exist")
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


def _strict_aggregation(value: Any) -> str:
    if value != "mean":
        raise ValueError(
            "production evaluator-bound evidence requires aggregation='mean' to match "
            "arithmetic_mean_over_exact_cohort"
        )
    return "mean"


@dataclass(frozen=True)
class EvaluatorBoundAdvancedEvaluationEvidence:
    cohort_evidence_path: str
    cohort_evidence_file_sha256: str
    cohort_evidence_sha256: str
    evaluator_bound_cohort_path: str
    evaluator_bound_cohort_file_sha256: str
    evaluator_bound_cohort_sha256: str
    evaluation_receipt_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        cohort_evidence = safe_advanced_path(
            self.cohort_evidence_path,
            label="cohort-bound advanced evaluation evidence",
            must_exist=True,
            require_file=True,
        )
        evaluator_cohort = safe_advanced_path(
            self.evaluator_bound_cohort_path,
            label="evaluator-bound evaluation cohort",
            must_exist=True,
            require_file=True,
        )
        object.__setattr__(self, "cohort_evidence_path", str(cohort_evidence))
        object.__setattr__(self, "evaluator_bound_cohort_path", str(evaluator_cohort))
        for name in (
            "cohort_evidence_file_sha256",
            "cohort_evidence_sha256",
            "evaluator_bound_cohort_file_sha256",
            "evaluator_bound_cohort_sha256",
            "evaluation_receipt_sha256",
            "evidence_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if _file_sha(cohort_evidence) != self.cohort_evidence_file_sha256:
            raise ValueError("cohort evaluation evidence bytes differ from evaluator-bound envelope")
        if _file_sha(evaluator_cohort) != self.evaluator_bound_cohort_file_sha256:
            raise ValueError("evaluator-bound cohort bytes differ from evaluation envelope")
        if _digest(self.unsigned()) != self.evidence_sha256:
            raise ValueError("evaluator-bound advanced evaluation evidence digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-evaluator-bound-advanced-evaluation-evidence/v3",
            "cohort_evidence_path": self.cohort_evidence_path,
            "cohort_evidence_file_sha256": self.cohort_evidence_file_sha256,
            "cohort_evidence_sha256": self.cohort_evidence_sha256,
            "evaluator_bound_cohort_path": self.evaluator_bound_cohort_path,
            "evaluator_bound_cohort_file_sha256": self.evaluator_bound_cohort_file_sha256,
            "evaluator_bound_cohort_sha256": self.evaluator_bound_cohort_sha256,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
        }


def build_evaluator_bound_advanced_evaluation_evidence(
    binding: VerifiedAdvancedCheckpointBinding,
    *,
    evaluator_bound_cohort_path: str | Path,
    result_receipt_paths: Sequence[str | Path],
    aggregation: str,
    evaluation_receipt_path: str | Path,
    cohort_evidence_path: str | Path,
) -> tuple[AdvancedEvaluationReceipt, EvaluatorBoundAdvancedEvaluationEvidence]:
    aggregation = _strict_aggregation(aggregation)
    evaluator_cohort_path = safe_advanced_path(
        evaluator_bound_cohort_path,
        label="evaluator-bound evaluation cohort",
        must_exist=True,
        require_file=True,
    )
    evaluator_binding, cohort, evaluator = verify_evaluator_bound_evaluation_cohort(
        evaluator_cohort_path
    )
    assert_strict_production_evaluator_contract(evaluator)
    selected_results = tuple(result_receipt_paths)
    if not selected_results:
        raise ValueError("production evaluation evidence requires at least one result receipt")
    for result_path in selected_results:
        verify_result_against_evaluator_bound_cohort(
            result_path,
            evaluator_bound_cohort_path=evaluator_cohort_path,
        )
    evaluation, cohort_evidence = build_cohort_bound_advanced_evaluation_evidence(
        binding,
        cohort_contract_path=evaluator_binding.cohort_contract_path,
        result_receipt_paths=selected_results,
        aggregation=aggregation,
        evaluation_receipt_path=evaluation_receipt_path,
    )
    if evaluation.aggregation != "mean":
        raise RuntimeError("production evaluation receipt escaped strict mean aggregation")
    cohort_output = safe_advanced_path(
        cohort_evidence_path,
        label="cohort-bound advanced evaluation evidence destination",
        must_exist=False,
    )
    if cohort_output.exists():
        raise ValueError("cohort-bound advanced evaluation evidence destination must not already exist")
    write_cohort_bound_advanced_evaluation_evidence(cohort_output, cohort_evidence)
    verified_evaluation, verified_cohort_evidence = verify_cohort_bound_advanced_evaluation_evidence(
        cohort_output
    )
    if (
        verified_evaluation.receipt_sha256 != evaluation.receipt_sha256
        or verified_cohort_evidence.evidence_sha256 != cohort_evidence.evidence_sha256
        or verified_cohort_evidence.cohort_contract_sha256 != cohort.contract_sha256
        or verified_evaluation.aggregation != "mean"
    ):
        raise RuntimeError("cohort-bound evaluation changed during evaluator-bound publication")
    unsigned = {
        "schema": "rigorousrag-evaluator-bound-advanced-evaluation-evidence/v3",
        "cohort_evidence_path": str(cohort_output),
        "cohort_evidence_file_sha256": _file_sha(cohort_output),
        "cohort_evidence_sha256": cohort_evidence.evidence_sha256,
        "evaluator_bound_cohort_path": str(evaluator_cohort_path),
        "evaluator_bound_cohort_file_sha256": _file_sha(evaluator_cohort_path),
        "evaluator_bound_cohort_sha256": evaluator_binding.contract_sha256,
        "evaluation_receipt_sha256": evaluation.receipt_sha256,
    }
    return evaluation, EvaluatorBoundAdvancedEvaluationEvidence(
        cohort_evidence_path=str(cohort_output),
        cohort_evidence_file_sha256=unsigned["cohort_evidence_file_sha256"],
        cohort_evidence_sha256=cohort_evidence.evidence_sha256,
        evaluator_bound_cohort_path=str(evaluator_cohort_path),
        evaluator_bound_cohort_file_sha256=unsigned["evaluator_bound_cohort_file_sha256"],
        evaluator_bound_cohort_sha256=evaluator_binding.contract_sha256,
        evaluation_receipt_sha256=evaluation.receipt_sha256,
        evidence_sha256=_digest(unsigned),
    )


def write_evaluator_bound_advanced_evaluation_evidence(
    path: str | Path,
    evidence: EvaluatorBoundAdvancedEvaluationEvidence,
) -> None:
    if not isinstance(evidence, EvaluatorBoundAdvancedEvaluationEvidence):
        raise ValueError("evidence must be EvaluatorBoundAdvancedEvaluationEvidence")
    destination = safe_advanced_path(
        path,
        label="evaluator-bound advanced evaluation evidence",
        must_exist=False,
    )
    _atomic(
        destination,
        _canonical({**evidence.unsigned(), "evidence_sha256": evidence.evidence_sha256})
        + b"\n",
    )


def read_evaluator_bound_advanced_evaluation_evidence(
    path: str | Path,
) -> EvaluatorBoundAdvancedEvaluationEvidence:
    source = safe_advanced_path(
        path,
        label="evaluator-bound advanced evaluation evidence",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("evaluator-bound advanced evaluation evidence exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("evaluator-bound advanced evaluation evidence is not strict JSON") from exc
    required = {
        "schema",
        "cohort_evidence_path",
        "cohort_evidence_file_sha256",
        "cohort_evidence_sha256",
        "evaluator_bound_cohort_path",
        "evaluator_bound_cohort_file_sha256",
        "evaluator_bound_cohort_sha256",
        "evaluation_receipt_sha256",
        "evidence_sha256",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema")
        != "rigorousrag-evaluator-bound-advanced-evaluation-evidence/v3"
    ):
        raise ValueError("unsupported evaluator-bound advanced evaluation schema")
    return EvaluatorBoundAdvancedEvaluationEvidence(
        **{key: value for key, value in raw.items() if key != "schema"}
    )


def verify_evaluator_bound_advanced_evaluation_evidence(
    path: str | Path,
) -> tuple[AdvancedEvaluationReceipt, EvaluatorBoundAdvancedEvaluationEvidence]:
    evidence = read_evaluator_bound_advanced_evaluation_evidence(path)
    if _file_sha(Path(evidence.cohort_evidence_path)) != evidence.cohort_evidence_file_sha256:
        raise ValueError("cohort evaluation evidence bytes changed after publication")
    if _file_sha(Path(evidence.evaluator_bound_cohort_path)) != evidence.evaluator_bound_cohort_file_sha256:
        raise ValueError("evaluator-bound cohort bytes changed after evaluation publication")
    evaluator_binding, _, evaluator = verify_evaluator_bound_evaluation_cohort(
        evidence.evaluator_bound_cohort_path
    )
    assert_strict_production_evaluator_contract(evaluator)
    if evaluator_binding.contract_sha256 != evidence.evaluator_bound_cohort_sha256:
        raise ValueError("evaluator-bound cohort identity changed after evaluation publication")
    evaluation, cohort_evidence = verify_cohort_bound_advanced_evaluation_evidence(
        evidence.cohort_evidence_path
    )
    if evaluation.aggregation != "mean":
        raise ValueError("production evaluator-bound evidence requires mean aggregation")
    if (
        cohort_evidence.evidence_sha256 != evidence.cohort_evidence_sha256
        or evaluation.receipt_sha256 != evidence.evaluation_receipt_sha256
        or cohort_evidence.cohort_contract_sha256
        != evaluator_binding.cohort_contract_sha256
    ):
        raise ValueError("cohort/evaluator evaluation identities differ")
    for item in cohort_evidence.runs:
        verify_result_against_evaluator_bound_cohort(
            item.result_receipt_path,
            evaluator_bound_cohort_path=evidence.evaluator_bound_cohort_path,
        )
    return evaluation, evidence


__all__ = [
    "EvaluatorBoundAdvancedEvaluationEvidence",
    "build_evaluator_bound_advanced_evaluation_evidence",
    "read_evaluator_bound_advanced_evaluation_evidence",
    "verify_evaluator_bound_advanced_evaluation_evidence",
    "write_evaluator_bound_advanced_evaluation_evidence",
]
