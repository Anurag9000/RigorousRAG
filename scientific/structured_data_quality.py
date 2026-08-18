"""Claim-evidence-specific quality gates for structured table/chart authority."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from evaluation.structured_data_support import ChartQuantityEvidence, TableQuantityEvidence
from scientific.chart_structure import StructuredChart
from scientific.document_structure import StructuredDocument


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be SHA-256")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be SHA-256")
    return selected


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be in [0, 1]")
    selected = float(value)
    if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return selected


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return selected


@dataclass(frozen=True)
class StructuredDataAuthorityPolicy:
    min_chart_extraction_confidence: float = 0.80
    min_axis_confidence: float = 0.70
    min_point_confidence: float = 0.80
    min_table_cell_confidence: float = 0.80
    max_relative_interval_width: float = 1.0
    require_explicit_unit: bool = False
    missing_confidence_requires_review: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_chart_extraction_confidence",
            "min_axis_confidence",
            "min_point_confidence",
            "min_table_cell_confidence",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        object.__setattr__(self, "max_relative_interval_width", _nonnegative(self.max_relative_interval_width, "max_relative_interval_width"))
        if not isinstance(self.require_explicit_unit, bool) or not isinstance(self.missing_confidence_requires_review, bool):
            raise ValueError("structured-data authority booleans are invalid")

    @property
    def policy_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-structured-data-authority-policy/v1", **asdict(self)})


@dataclass(frozen=True)
class StructuredDataAuthorityDecision:
    evidence_sha256: str
    evidence_kind: str
    policy_sha256: str
    action: str
    reason_codes: tuple[str, ...]
    confidence_floor: float | None
    relative_interval_width: float
    decision_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_sha256", _sha(self.evidence_sha256, "evidence_sha256"))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, "policy_sha256"))
        if self.evidence_kind not in {"table_quantity", "chart_quantity", "chart_trend"}:
            raise ValueError("evidence_kind is invalid")
        if self.action not in {"authoritative", "review_required", "blocked"}:
            raise ValueError("action is invalid")
        reasons = tuple(sorted(set(self.reason_codes)))
        if self.action == "authoritative" and reasons:
            raise ValueError("authoritative decision may not contain failure reasons")
        if self.action != "authoritative" and not reasons:
            raise ValueError("non-authoritative decision requires reason codes")
        object.__setattr__(self, "reason_codes", reasons)
        if self.confidence_floor is not None:
            object.__setattr__(self, "confidence_floor", _probability(self.confidence_floor, "confidence_floor"))
        object.__setattr__(self, "relative_interval_width", _nonnegative(self.relative_interval_width, "relative_interval_width"))
        expected = _digest(self._payload())
        provided = _sha(self.decision_sha256, "decision_sha256")
        if expected != provided:
            raise ValueError("decision_sha256 does not match structured-data authority decision")
        object.__setattr__(self, "decision_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-structured-data-authority-decision/v1",
            "evidence_sha256": self.evidence_sha256,
            "evidence_kind": self.evidence_kind,
            "policy_sha256": self.policy_sha256,
            "action": self.action,
            "reason_codes": self.reason_codes,
            "confidence_floor": self.confidence_floor,
            "relative_interval_width": self.relative_interval_width,
        }


def _relative_width(value: float, lower: float, upper: float) -> float:
    width = max(0.0, upper - lower)
    denominator = max(abs(value), 1e-12)
    return width / denominator


def _decision(*, evidence_sha256: str, kind: str, policy: StructuredDataAuthorityPolicy, reasons: list[str], confidences: Sequence[float], relative_width: float) -> StructuredDataAuthorityDecision:
    blocking = {"lineage_mismatch", "evidence_source_missing"}
    action = "blocked" if blocking & set(reasons) else "review_required" if reasons else "authoritative"
    confidence_floor = min(confidences) if confidences else None
    payload = {
        "schema": "rigorousrag-structured-data-authority-decision/v1",
        "evidence_sha256": evidence_sha256,
        "evidence_kind": kind,
        "policy_sha256": policy.policy_sha256,
        "action": action,
        "reason_codes": tuple(sorted(set(reasons))),
        "confidence_floor": confidence_floor,
        "relative_interval_width": relative_width,
    }
    return StructuredDataAuthorityDecision(**payload, decision_sha256=_digest(payload))


def evaluate_table_quantity_authority(
    document: StructuredDocument,
    evidence: TableQuantityEvidence,
    *,
    policy: StructuredDataAuthorityPolicy = StructuredDataAuthorityPolicy(),
) -> StructuredDataAuthorityDecision:
    if not isinstance(document, StructuredDocument) or not isinstance(evidence, TableQuantityEvidence):
        raise ValueError("table authority inputs have invalid types")
    reasons: list[str] = []
    confidences: list[float] = []
    if document.document_id != evidence.document_id or document.generation_id != evidence.generation_id:
        reasons.append("lineage_mismatch")
    table = next((item for item in document.tables if item.table_region_id == evidence.table_region_id), None)
    cell = None if table is None else next((item for item in table.cells if item.cell_id == evidence.cell_id), None)
    if cell is None:
        reasons.append("evidence_source_missing")
    else:
        if cell.row_start != evidence.row_start or cell.column_start != evidence.column_start:
            reasons.append("lineage_mismatch")
        cell_sha = hashlib.sha256(cell.text.encode("utf-8")).hexdigest()
        if cell_sha != evidence.cell_text_sha256:
            reasons.append("lineage_mismatch")
        if cell.confidence is None:
            if policy.missing_confidence_requires_review:
                reasons.append("cell_confidence_missing")
        else:
            confidences.append(cell.confidence)
            if cell.confidence < policy.min_table_cell_confidence:
                reasons.append("cell_confidence_below_threshold")
    if policy.require_explicit_unit and evidence.quantity.unit is None:
        reasons.append("unit_missing")
    relative_width = _relative_width(evidence.quantity.value, float(evidence.quantity.lower), float(evidence.quantity.upper))
    if relative_width > policy.max_relative_interval_width:
        reasons.append("uncertainty_interval_too_wide")
    return _decision(evidence_sha256=evidence.evidence_sha256, kind="table_quantity", policy=policy, reasons=reasons, confidences=confidences, relative_width=relative_width)


def evaluate_chart_quantity_authority(
    chart: StructuredChart,
    evidence: ChartQuantityEvidence,
    *,
    policy: StructuredDataAuthorityPolicy = StructuredDataAuthorityPolicy(),
) -> StructuredDataAuthorityDecision:
    if not isinstance(chart, StructuredChart) or not isinstance(evidence, ChartQuantityEvidence):
        raise ValueError("chart authority inputs have invalid types")
    reasons: list[str] = []
    confidences: list[float] = []
    if chart.anchor.document_id != evidence.document_id or chart.anchor.generation_id != evidence.generation_id or chart.chart_region_id != evidence.chart_region_id or chart.chart_sha256 != evidence.chart_sha256:
        reasons.append("lineage_mismatch")
    if chart.extraction_confidence is None:
        if policy.missing_confidence_requires_review:
            reasons.append("chart_extraction_confidence_missing")
    else:
        confidences.append(chart.extraction_confidence)
        if chart.extraction_confidence < policy.min_chart_extraction_confidence:
            reasons.append("chart_extraction_confidence_below_threshold")
    try:
        series = chart.series_by_id(evidence.series_id)
        point = series.points[evidence.point_index]
        if point.point_id != evidence.point_id:
            reasons.append("lineage_mismatch")
        axis = chart.axis(series.y_axis_id)
    except (KeyError, IndexError):
        point = None
        axis = None
        reasons.append("evidence_source_missing")
    if point is not None:
        if point.confidence is None:
            if policy.missing_confidence_requires_review:
                reasons.append("point_confidence_missing")
        else:
            confidences.append(point.confidence)
            if point.confidence < policy.min_point_confidence:
                reasons.append("point_confidence_below_threshold")
    if axis is not None:
        if axis.confidence is None:
            if policy.missing_confidence_requires_review:
                reasons.append("axis_confidence_missing")
        else:
            confidences.append(axis.confidence)
            if axis.confidence < policy.min_axis_confidence:
                reasons.append("axis_confidence_below_threshold")
        if axis.unit != evidence.quantity.unit:
            reasons.append("lineage_mismatch")
    if policy.require_explicit_unit and evidence.quantity.unit is None:
        reasons.append("unit_missing")
    relative_width = _relative_width(evidence.quantity.value, float(evidence.quantity.lower), float(evidence.quantity.upper))
    if relative_width > policy.max_relative_interval_width:
        reasons.append("uncertainty_interval_too_wide")
    return _decision(evidence_sha256=evidence.evidence_sha256, kind="chart_quantity", policy=policy, reasons=reasons, confidences=confidences, relative_width=relative_width)


def evaluate_chart_trend_authority(
    chart: StructuredChart,
    *,
    series_id: str,
    policy: StructuredDataAuthorityPolicy = StructuredDataAuthorityPolicy(),
) -> StructuredDataAuthorityDecision:
    if not isinstance(chart, StructuredChart):
        raise ValueError("chart must be StructuredChart")
    reasons: list[str] = []
    confidences: list[float] = []
    if chart.extraction_confidence is None:
        if policy.missing_confidence_requires_review:
            reasons.append("chart_extraction_confidence_missing")
    else:
        confidences.append(chart.extraction_confidence)
        if chart.extraction_confidence < policy.min_chart_extraction_confidence:
            reasons.append("chart_extraction_confidence_below_threshold")
    try:
        series = chart.series_by_id(series_id)
        axis = chart.axis(series.y_axis_id)
    except KeyError:
        series = None
        axis = None
        reasons.append("evidence_source_missing")
    if axis is not None:
        if axis.confidence is None:
            if policy.missing_confidence_requires_review:
                reasons.append("axis_confidence_missing")
        else:
            confidences.append(axis.confidence)
            if axis.confidence < policy.min_axis_confidence:
                reasons.append("axis_confidence_below_threshold")
        if policy.require_explicit_unit and axis.unit is None:
            reasons.append("unit_missing")
    if series is not None:
        for point in series.points:
            if point.confidence is None:
                if policy.missing_confidence_requires_review:
                    reasons.append("point_confidence_missing")
                    break
            else:
                confidences.append(point.confidence)
                if point.confidence < policy.min_point_confidence:
                    reasons.append("point_confidence_below_threshold")
                    break
    evidence_sha256 = _digest({"schema": "rigorousrag-chart-trend-evidence/v1", "chart_sha256": chart.chart_sha256, "series_id": series_id})
    return _decision(evidence_sha256=evidence_sha256, kind="chart_trend", policy=policy, reasons=reasons, confidences=confidences, relative_width=0.0)


__all__ = [
    "StructuredDataAuthorityDecision",
    "StructuredDataAuthorityPolicy",
    "evaluate_chart_quantity_authority",
    "evaluate_chart_trend_authority",
    "evaluate_table_quantity_authority",
]
