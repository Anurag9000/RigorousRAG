"""Evaluator-bound production cohort authority.

The base ``AuthoritativeEvaluationCohortContract`` binds the evaluator contract SHA. This
wrapper additionally binds the independently reconstructable evaluator-contract receipt file,
so restart-time promotion can prove evaluator source/config/metric semantics without relying on
an operator to re-supply the correct receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.authoritative_evaluation_cohort import (
    AuthoritativeEvaluationCohortContract,
    assert_result_receipt_matches_cohort,
    verify_authoritative_evaluation_cohort,
)
from evaluation.authoritative_evaluator_contract import (
    AuthoritativeEvaluatorContract,
    verify_authoritative_evaluator_contract,
)
from evaluation.strict_authoritative_benchmark_result_verification import (
    verify_strict_authoritative_benchmark_result_receipt,
)
from evaluation.strict_result_metric_schema import verify_homogeneous_result_metric_schema
from training.advanced_path_authority import safe_advanced_path

_MAX_BYTES = 16 * 1024 * 1024
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
        raise ValueError("evaluator-bound cohort destination must not already exist")
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
class EvaluatorBoundEvaluationCohort:
    cohort_contract_path: str
    cohort_contract_file_sha256: str
    cohort_contract_sha256: str
    evaluator_contract_receipt_path: str
    evaluator_contract_receipt_file_sha256: str
    evaluator_contract_sha256: str
    contract_sha256: str

    def __post_init__(self) -> None:
        cohort = safe_advanced_path(
            self.cohort_contract_path,
            label="authoritative evaluation cohort",
            must_exist=True,
            require_file=True,
        )
        evaluator = safe_advanced_path(
            self.evaluator_contract_receipt_path,
            label="authoritative evaluator contract receipt",
            must_exist=True,
            require_file=True,
        )
        object.__setattr__(self, "cohort_contract_path", str(cohort))
        object.__setattr__(self, "evaluator_contract_receipt_path", str(evaluator))
        for name in (
            "cohort_contract_file_sha256",
            "cohort_contract_sha256",
            "evaluator_contract_receipt_file_sha256",
            "evaluator_contract_sha256",
            "contract_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if _file_sha(cohort) != self.cohort_contract_file_sha256:
            raise ValueError("evaluation cohort bytes differ from evaluator-bound receipt")
        if _file_sha(evaluator) != self.evaluator_contract_receipt_file_sha256:
            raise ValueError("evaluator receipt bytes differ from evaluator-bound receipt")
        if _digest(self.unsigned()) != self.contract_sha256:
            raise ValueError("evaluator-bound cohort digest mismatch")

    def unsigned(self) -> Mapping[str, Any]:
        return {
            "schema": "rigorousrag-evaluator-bound-evaluation-cohort/v1",
            "cohort_contract_path": self.cohort_contract_path,
            "cohort_contract_file_sha256": self.cohort_contract_file_sha256,
            "cohort_contract_sha256": self.cohort_contract_sha256,
            "evaluator_contract_receipt_path": self.evaluator_contract_receipt_path,
            "evaluator_contract_receipt_file_sha256": self.evaluator_contract_receipt_file_sha256,
            "evaluator_contract_sha256": self.evaluator_contract_sha256,
        }


def build_evaluator_bound_evaluation_cohort(
    cohort_contract_path: str | Path,
    *,
    evaluator_contract_receipt_path: str | Path,
) -> EvaluatorBoundEvaluationCohort:
    cohort_path = safe_advanced_path(
        cohort_contract_path,
        label="authoritative evaluation cohort",
        must_exist=True,
        require_file=True,
    )
    evaluator_path = safe_advanced_path(
        evaluator_contract_receipt_path,
        label="authoritative evaluator contract receipt",
        must_exist=True,
        require_file=True,
    )
    cohort = verify_authoritative_evaluation_cohort(cohort_path)
    evaluator = verify_authoritative_evaluator_contract(evaluator_path)
    if evaluator.contract_sha256 != cohort.base_evaluator_contract_sha256:
        raise ValueError("evaluator receipt differs from evaluation cohort base contract")
    unsigned = {
        "schema": "rigorousrag-evaluator-bound-evaluation-cohort/v1",
        "cohort_contract_path": str(cohort_path),
        "cohort_contract_file_sha256": _file_sha(cohort_path),
        "cohort_contract_sha256": cohort.contract_sha256,
        "evaluator_contract_receipt_path": str(evaluator_path),
        "evaluator_contract_receipt_file_sha256": _file_sha(evaluator_path),
        "evaluator_contract_sha256": evaluator.contract_sha256,
    }
    return EvaluatorBoundEvaluationCohort(
        cohort_contract_path=str(cohort_path),
        cohort_contract_file_sha256=unsigned["cohort_contract_file_sha256"],
        cohort_contract_sha256=cohort.contract_sha256,
        evaluator_contract_receipt_path=str(evaluator_path),
        evaluator_contract_receipt_file_sha256=unsigned[
            "evaluator_contract_receipt_file_sha256"
        ],
        evaluator_contract_sha256=evaluator.contract_sha256,
        contract_sha256=_digest(unsigned),
    )


def write_evaluator_bound_evaluation_cohort(
    path: str | Path,
    binding: EvaluatorBoundEvaluationCohort,
) -> None:
    if not isinstance(binding, EvaluatorBoundEvaluationCohort):
        raise ValueError("binding must be EvaluatorBoundEvaluationCohort")
    destination = safe_advanced_path(
        path,
        label="evaluator-bound evaluation cohort",
        must_exist=False,
    )
    _atomic(
        destination,
        _canonical({**binding.unsigned(), "contract_sha256": binding.contract_sha256})
        + b"\n",
    )


def read_evaluator_bound_evaluation_cohort(
    path: str | Path,
) -> EvaluatorBoundEvaluationCohort:
    source = safe_advanced_path(
        path,
        label="evaluator-bound evaluation cohort",
        must_exist=True,
        require_file=True,
    )
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_BYTES:
        raise ValueError("evaluator-bound cohort exceeds byte safety bound")
    try:
        raw = json.loads(
            source.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except Exception as exc:
        raise ValueError("evaluator-bound cohort is not strict JSON") from exc
    required = {
        "schema",
        "cohort_contract_path",
        "cohort_contract_file_sha256",
        "cohort_contract_sha256",
        "evaluator_contract_receipt_path",
        "evaluator_contract_receipt_file_sha256",
        "evaluator_contract_sha256",
        "contract_sha256",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != required
        or raw.get("schema") != "rigorousrag-evaluator-bound-evaluation-cohort/v1"
    ):
        raise ValueError("unsupported evaluator-bound cohort schema")
    return EvaluatorBoundEvaluationCohort(
        **{key: value for key, value in raw.items() if key != "schema"}
    )


def verify_evaluator_bound_evaluation_cohort(
    path: str | Path,
) -> tuple[
    EvaluatorBoundEvaluationCohort,
    AuthoritativeEvaluationCohortContract,
    AuthoritativeEvaluatorContract,
]:
    binding = read_evaluator_bound_evaluation_cohort(path)
    cohort = verify_authoritative_evaluation_cohort(binding.cohort_contract_path)
    evaluator = verify_authoritative_evaluator_contract(
        binding.evaluator_contract_receipt_path
    )
    checks = {
        "cohort_file": _file_sha(Path(binding.cohort_contract_path))
        == binding.cohort_contract_file_sha256,
        "cohort_contract": cohort.contract_sha256 == binding.cohort_contract_sha256,
        "evaluator_file": _file_sha(Path(binding.evaluator_contract_receipt_path))
        == binding.evaluator_contract_receipt_file_sha256,
        "evaluator_contract": evaluator.contract_sha256
        == binding.evaluator_contract_sha256,
        "cohort_evaluator": cohort.base_evaluator_contract_sha256
        == evaluator.contract_sha256,
    }
    failures = [name for name, matched in checks.items() if not matched]
    if failures:
        raise ValueError(
            "evaluator-bound cohort reconstruction differs: " + ",".join(failures)
        )
    return binding, cohort, evaluator


def verify_result_against_evaluator_bound_cohort(
    result_receipt_path: str | Path,
    *,
    evaluator_bound_cohort_path: str | Path,
) -> Any:
    _, cohort, evaluator = verify_evaluator_bound_evaluation_cohort(
        evaluator_bound_cohort_path
    )
    cohort_run = assert_result_receipt_matches_cohort(
        result_receipt_path,
        cohort=cohort,
    )
    run, receipt = verify_strict_authoritative_benchmark_result_receipt(
        result_receipt_path
    )
    if run.run_sha256 != cohort_run.run_sha256:
        raise RuntimeError("cohort and strict result verifiers disagree on run identity")
    verify_homogeneous_result_metric_schema(receipt)
    declared = {metric.name for metric in evaluator.metrics}
    observed = set(run.metrics)
    if observed != declared:
        raise ValueError(
            "result aggregate metric schema differs from evaluator receipt; "
            f"missing={sorted(declared-observed)[:50]} "
            f"extra={sorted(observed-declared)[:50]}"
        )
    return run


__all__ = [
    "EvaluatorBoundEvaluationCohort",
    "build_evaluator_bound_evaluation_cohort",
    "read_evaluator_bound_evaluation_cohort",
    "verify_evaluator_bound_evaluation_cohort",
    "verify_result_against_evaluator_bound_cohort",
    "write_evaluator_bound_evaluation_cohort",
]
