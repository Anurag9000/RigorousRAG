"""Text-free benchmark cases and aggregate suites for governed claim extractors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from tools.evidence_graph_claim_contracts import (
    ScientificClaimProposal,
    _digest,
    _identifier,
    _integer,
    _sha256,
)
from tools.evidence_graph_claim_evaluation import (
    ScientificClaimEvaluationReport,
    _harmonic,
    _ratio,
)
from tools.evidence_graph_claim_evaluation_verification import (
    verify_scientific_claim_evaluation_report,
)
from tools.evidence_graph_claim_extractor_registry import (
    ScientificClaimExtractorRecord,
)
from tools.security import normalize_owner_id

_MAX_CASES = 100_000


@dataclass(frozen=True)
class ScientificClaimExtractorBenchmarkCase:
    case_id: str
    dataset_digest: str
    owner_id: str
    extractor_name: str
    extractor_version: str
    extractor_record_digest: str
    evaluation_report_digest: str
    gold_count: int
    proposal_count: int
    matched_count: int
    precision: float
    recall: float
    f1: float
    exact_evidence_accuracy: float
    exact_locator_accuracy: float
    mean_span_iou: float
    mean_claim_token_f1: float
    claim_type_accuracy: float
    modality_accuracy: float
    confidence_brier_score: float
    case_digest: str
    contains_claim_text: bool = False
    contains_evidence_text: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id", 500))
        object.__setattr__(self, "dataset_digest", _digest(self.dataset_digest, "dataset_digest"))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "extractor_name",
            _identifier(self.extractor_name, "extractor_name", 200),
        )
        object.__setattr__(
            self,
            "extractor_version",
            _identifier(self.extractor_version, "extractor_version", 200),
        )
        for name in (
            "extractor_record_digest",
            "evaluation_report_digest",
            "case_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("gold_count", "proposal_count", "matched_count"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 0, 100_000_000),
            )
        if self.matched_count > min(self.gold_count, self.proposal_count):
            raise ValueError("matched_count exceeds benchmark case counts.")
        for name in (
            "precision",
            "recall",
            "f1",
            "exact_evidence_accuracy",
            "exact_locator_accuracy",
            "mean_span_iou",
            "mean_claim_token_f1",
            "claim_type_accuracy",
            "modality_accuracy",
            "confidence_brier_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            object.__setattr__(self, name, value)
        if self.contains_claim_text is not False or self.contains_evidence_text is not False:
            raise ValueError("benchmark case text-safety flags must remain false.")
        if self.schema_version != 1:
            raise ValueError("benchmark case schema is unsupported.")
        stable = {
            "scope": "rigorousrag-scientific-claim-extractor-benchmark-case-v1",
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"case_digest", "contains_claim_text", "contains_evidence_text"}
            },
        }
        if self.case_digest != _sha256(stable):
            raise ValueError("case_digest differs from benchmark case.")


@dataclass(frozen=True)
class ScientificClaimExtractorBenchmarkSuite:
    benchmark_id: str
    owner_id: str
    extractor_name: str
    extractor_version: str
    extractor_record_digest: str
    case_count: int
    gold_count: int
    proposal_count: int
    matched_count: int
    precision: float
    recall: float
    f1: float
    exact_evidence_accuracy: float
    exact_locator_accuracy: float
    mean_span_iou: float
    mean_claim_token_f1: float
    claim_type_accuracy: float
    modality_accuracy: float
    confidence_brier_score: float
    cases: tuple[ScientificClaimExtractorBenchmarkCase, ...]
    suite_digest: str
    contains_claim_text: bool = False
    contains_evidence_text: bool = False
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "benchmark_id",
            _identifier(self.benchmark_id, "benchmark_id", 500),
        )
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self,
            "extractor_name",
            _identifier(self.extractor_name, "extractor_name", 200),
        )
        object.__setattr__(
            self,
            "extractor_version",
            _identifier(self.extractor_version, "extractor_version", 200),
        )
        object.__setattr__(
            self,
            "extractor_record_digest",
            _digest(self.extractor_record_digest, "extractor_record_digest"),
        )
        for name in ("case_count", "gold_count", "proposal_count", "matched_count"):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), name, 0, 100_000_000),
            )
        if self.case_count != len(self.cases):
            raise ValueError("case_count differs from benchmark cases.")
        if not isinstance(self.cases, tuple) or not 1 <= len(self.cases) <= _MAX_CASES:
            raise ValueError("cases must be a bounded non-empty tuple.")
        if any(
            not isinstance(value, ScientificClaimExtractorBenchmarkCase)
            for value in self.cases
        ):
            raise ValueError("cases contains an unsupported value.")
        if len({value.case_id for value in self.cases}) != len(self.cases):
            raise ValueError("benchmark suite contains duplicate case IDs.")
        if len({value.dataset_digest for value in self.cases}) != len(self.cases):
            raise ValueError("benchmark suite contains duplicate dataset digests.")
        for value in self.cases:
            if (
                value.owner_id != self.owner_id
                or value.extractor_name != self.extractor_name
                or value.extractor_version != self.extractor_version
                or value.extractor_record_digest != self.extractor_record_digest
            ):
                raise ValueError("benchmark case escaped extractor suite scope.")
        for name in (
            "precision",
            "recall",
            "f1",
            "exact_evidence_accuracy",
            "exact_locator_accuracy",
            "mean_span_iou",
            "mean_claim_token_f1",
            "claim_type_accuracy",
            "modality_accuracy",
            "confidence_brier_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "suite_digest", _digest(self.suite_digest, "suite_digest"))
        if self.contains_claim_text is not False or self.contains_evidence_text is not False:
            raise ValueError("benchmark suite text-safety flags must remain false.")
        if self.schema_version != 1:
            raise ValueError("benchmark suite schema is unsupported.")
        stable = {
            "scope": "rigorousrag-scientific-claim-extractor-benchmark-suite-v1",
            **{
                key: value
                for key, value in asdict(self).items()
                if key not in {"suite_digest", "contains_claim_text", "contains_evidence_text"}
            },
        }
        if self.suite_digest != _sha256(stable):
            raise ValueError("suite_digest differs from benchmark suite.")


def build_scientific_claim_extractor_benchmark_case(
    *,
    case_id: str,
    dataset_digest: str,
    evaluation_report: ScientificClaimEvaluationReport,
    proposals: Iterable[ScientificClaimProposal],
    extractor_record: ScientificClaimExtractorRecord,
    minimum_span_iou: float = 0.5,
    minimum_claim_token_f1: float = 0.5,
) -> ScientificClaimExtractorBenchmarkCase:
    if not isinstance(extractor_record, ScientificClaimExtractorRecord):
        raise ValueError("extractor_record must be ScientificClaimExtractorRecord.")
    verify_scientific_claim_evaluation_report(
        evaluation_report,
        minimum_span_iou=minimum_span_iou,
        minimum_claim_token_f1=minimum_claim_token_f1,
    )
    if evaluation_report.owner_id != extractor_record.owner_id:
        raise PermissionError("evaluation owner differs from extractor owner.")
    if isinstance(proposals, (str, bytes, bytearray)):
        raise ValueError("proposals must be an iterable.")
    values = tuple(proposals)
    if len(values) != evaluation_report.proposal_count:
        raise ValueError("proposal count differs from evaluation report.")
    if any(not isinstance(value, ScientificClaimProposal) for value in values):
        raise ValueError("every proposal must be ScientificClaimProposal.")
    proposal_ids = {
        value.proposal_id
        for value in values
    }
    report_ids = {
        value.proposal_id
        for value in evaluation_report.matches
    } | set(evaluation_report.unmatched_proposal_ids)
    if proposal_ids != report_ids:
        raise ValueError("proposal identities differ from evaluation report.")
    for proposal in values:
        metadata_digest = proposal.metadata.get("extractor_registry_record_digest")
        if metadata_digest != extractor_record.record_digest:
            raise PermissionError("proposal lacks the exact extractor registry digest.")
        if (
            proposal.owner_id != extractor_record.owner_id
            or proposal.extractor_name != extractor_record.extractor_name
            or proposal.extractor_version != extractor_record.extractor_version
        ):
            raise PermissionError("proposal differs from extractor record scope.")
    values_for_digest = {
        "scope": "rigorousrag-scientific-claim-extractor-benchmark-case-v1",
        "case_id": _identifier(case_id, "case_id", 500),
        "dataset_digest": _digest(dataset_digest, "dataset_digest"),
        "owner_id": extractor_record.owner_id,
        "extractor_name": extractor_record.extractor_name,
        "extractor_version": extractor_record.extractor_version,
        "extractor_record_digest": extractor_record.record_digest,
        "evaluation_report_digest": evaluation_report.report_digest,
        "gold_count": evaluation_report.gold_count,
        "proposal_count": evaluation_report.proposal_count,
        "matched_count": evaluation_report.matched_count,
        "precision": evaluation_report.precision,
        "recall": evaluation_report.recall,
        "f1": evaluation_report.f1,
        "exact_evidence_accuracy": evaluation_report.exact_evidence_accuracy,
        "exact_locator_accuracy": evaluation_report.exact_locator_accuracy,
        "mean_span_iou": evaluation_report.mean_span_iou,
        "mean_claim_token_f1": evaluation_report.mean_claim_token_f1,
        "claim_type_accuracy": evaluation_report.claim_type_accuracy,
        "modality_accuracy": evaluation_report.modality_accuracy,
        "confidence_brier_score": evaluation_report.confidence_brier_score,
        "schema_version": 1,
    }
    return ScientificClaimExtractorBenchmarkCase(
        **{key: value for key, value in values_for_digest.items() if key != "scope"},
        case_digest=_sha256(values_for_digest),
    )


def aggregate_scientific_claim_extractor_benchmark(
    *,
    benchmark_id: str,
    cases: Iterable[ScientificClaimExtractorBenchmarkCase],
) -> ScientificClaimExtractorBenchmarkSuite:
    if isinstance(cases, (str, bytes, bytearray)):
        raise ValueError("cases must be an iterable.")
    values = tuple(cases)
    if not 1 <= len(values) <= _MAX_CASES:
        raise ValueError("cases must contain a bounded non-empty collection.")
    if any(not isinstance(value, ScientificClaimExtractorBenchmarkCase) for value in values):
        raise ValueError("every case must be ScientificClaimExtractorBenchmarkCase.")
    ordered = tuple(sorted(values, key=lambda value: value.case_id))
    first = ordered[0]
    for value in ordered:
        if (
            value.owner_id != first.owner_id
            or value.extractor_name != first.extractor_name
            or value.extractor_version != first.extractor_version
            or value.extractor_record_digest != first.extractor_record_digest
        ):
            raise PermissionError("benchmark cases differ in extractor scope.")
    gold_count = sum(value.gold_count for value in ordered)
    proposal_count = sum(value.proposal_count for value in ordered)
    matched_count = sum(value.matched_count for value in ordered)
    precision = _ratio(matched_count, proposal_count)
    recall = _ratio(matched_count, gold_count)

    def weighted(name: str, denominator: int, weight_name: str) -> float:
        if denominator == 0:
            return 0.0
        return sum(
            float(getattr(value, name)) * int(getattr(value, weight_name))
            for value in ordered
        ) / denominator

    report_values = {
        "benchmark_id": _identifier(benchmark_id, "benchmark_id", 500),
        "owner_id": first.owner_id,
        "extractor_name": first.extractor_name,
        "extractor_version": first.extractor_version,
        "extractor_record_digest": first.extractor_record_digest,
        "case_count": len(ordered),
        "gold_count": gold_count,
        "proposal_count": proposal_count,
        "matched_count": matched_count,
        "precision": precision,
        "recall": recall,
        "f1": _harmonic(precision, recall),
        "exact_evidence_accuracy": weighted(
            "exact_evidence_accuracy", matched_count, "matched_count"
        ),
        "exact_locator_accuracy": weighted(
            "exact_locator_accuracy", matched_count, "matched_count"
        ),
        "mean_span_iou": weighted("mean_span_iou", matched_count, "matched_count"),
        "mean_claim_token_f1": weighted(
            "mean_claim_token_f1", matched_count, "matched_count"
        ),
        "claim_type_accuracy": weighted(
            "claim_type_accuracy", matched_count, "matched_count"
        ),
        "modality_accuracy": weighted(
            "modality_accuracy", matched_count, "matched_count"
        ),
        "confidence_brier_score": weighted(
            "confidence_brier_score", proposal_count, "proposal_count"
        ),
        "cases": ordered,
        "schema_version": 1,
    }
    stable = {
        "scope": "rigorousrag-scientific-claim-extractor-benchmark-suite-v1",
        **{
            **report_values,
            "cases": [asdict(value) for value in ordered],
        },
    }
    return ScientificClaimExtractorBenchmarkSuite(
        **report_values,
        suite_digest=_sha256(stable),
    )


__all__ = [
    "ScientificClaimExtractorBenchmarkCase",
    "ScientificClaimExtractorBenchmarkSuite",
    "aggregate_scientific_claim_extractor_benchmark",
    "build_scientific_claim_extractor_benchmark_case",
]
