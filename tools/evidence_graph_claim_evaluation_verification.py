"""Strict reconstruction verification for text-free claim evaluation reports."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tools.evidence_graph_claim_contracts import _sha256
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimEvaluationReport,
    _finite_metric,
)


def verify_scientific_claim_evaluation_report(
    report: ScientificClaimEvaluationReport,
    *,
    minimum_span_iou: float = 0.5,
    minimum_claim_token_f1: float = 0.5,
) -> ScientificClaimEvaluationReport:
    if not isinstance(report, ScientificClaimEvaluationReport):
        raise ValueError("report must be ScientificClaimEvaluationReport.")
    span_threshold = _finite_metric(minimum_span_iou, "minimum_span_iou")
    text_threshold = _finite_metric(
        minimum_claim_token_f1, "minimum_claim_token_f1"
    )
    stable: dict[str, Any] = {
        "scope": "rigorousrag-scientific-claim-evaluation-report-v1",
        "owner_id": report.owner_id,
        "doc_id": report.doc_id,
        "generation": report.generation,
        "content_sha256": report.content_sha256,
        "profile_fingerprint": report.profile_fingerprint,
        "gold_count": report.gold_count,
        "proposal_count": report.proposal_count,
        "matched_count": report.matched_count,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "exact_evidence_accuracy": report.exact_evidence_accuracy,
        "exact_locator_accuracy": report.exact_locator_accuracy,
        "mean_span_iou": report.mean_span_iou,
        "mean_claim_token_f1": report.mean_claim_token_f1,
        "claim_type_accuracy": report.claim_type_accuracy,
        "modality_accuracy": report.modality_accuracy,
        "confidence_brier_score": report.confidence_brier_score,
        "unmatched_gold_ids": report.unmatched_gold_ids,
        "unmatched_proposal_ids": report.unmatched_proposal_ids,
        "matches": [asdict(value) for value in report.matches],
        "minimum_span_iou": span_threshold,
        "minimum_claim_token_f1": text_threshold,
    }
    if report.report_digest != _sha256(stable):
        raise ValueError("claim evaluation report digest is invalid.")
    return report


__all__ = ["verify_scientific_claim_evaluation_report"]
