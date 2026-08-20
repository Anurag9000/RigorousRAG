"""Production verifier for authoritative advanced evaluation evidence.

Historical v1/v2 evidence remains readable through its original modules for research
reproducibility. Production promotion imports this module, which deliberately accepts only the
v3 evaluator-bound envelope: evaluator receipt/config/metric schema + authoritative benchmark
cohort/sample universe + strict v2 result artifacts.
"""
from __future__ import annotations

from pathlib import Path

from evaluation.advanced_rag_receipts import AdvancedEvaluationReceipt
from evaluation.evaluator_bound_advanced_evaluation import (
    EvaluatorBoundAdvancedEvaluationEvidence,
    verify_evaluator_bound_advanced_evaluation_evidence,
)


def verify_authoritative_advanced_evaluation_evidence(
    path: str | Path,
) -> tuple[AdvancedEvaluationReceipt, EvaluatorBoundAdvancedEvaluationEvidence]:
    return verify_evaluator_bound_advanced_evaluation_evidence(path)


__all__ = ["verify_authoritative_advanced_evaluation_evidence"]
