"""Quality observations and promotion gates for scientific document structure.

The structure IR deliberately allows multiple OCR/layout providers.  This module prevents
that flexibility from becoming silent degradation: callers supply measured observations,
then deterministic gates decide whether a parsed document is eligible for retrieval,
needs review, or must be blocked.  No OCR/layout model is executed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from scientific.document_structure import RegionKind, StructuredDocument


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _unit(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be within [0,1]")
    return result


class QualityDisposition(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    BLOCK = "block"


@dataclass(frozen=True)
class StructureQualityObservation:
    reading_order_coverage: float
    text_region_confidence_mean: float | None
    low_confidence_text_fraction: float | None
    table_structure_coverage: float
    table_nonempty_cell_fraction: float | None
    formula_representation_coverage: float
    figure_caption_coverage: float
    dangling_reference_count: int
    cyclic_reading_order: bool = False
    extraction_error_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "reading_order_coverage",
            "table_structure_coverage",
            "formula_representation_coverage",
            "figure_caption_coverage",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        for name in ("text_region_confidence_mean", "low_confidence_text_fraction", "table_nonempty_cell_fraction"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _unit(value, name))
        for name in ("dangling_reference_count", "extraction_error_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.cyclic_reading_order, bool):
            raise ValueError("cyclic_reading_order must be boolean")


@dataclass(frozen=True)
class StructureQualityPolicy:
    minimum_reading_order_coverage: float = 0.95
    minimum_text_confidence_mean: float = 0.75
    maximum_low_confidence_text_fraction: float = 0.20
    minimum_table_structure_coverage: float = 0.90
    minimum_table_nonempty_cell_fraction: float = 0.80
    minimum_formula_representation_coverage: float = 0.90
    minimum_figure_caption_coverage: float = 0.90
    maximum_dangling_references: int = 0
    maximum_extraction_errors: int = 0
    block_on_cycle: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_reading_order_coverage",
            "minimum_text_confidence_mean",
            "maximum_low_confidence_text_fraction",
            "minimum_table_structure_coverage",
            "minimum_table_nonempty_cell_fraction",
            "minimum_formula_representation_coverage",
            "minimum_figure_caption_coverage",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        for name in ("maximum_dangling_references", "maximum_extraction_errors"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.block_on_cycle, bool):
            raise ValueError("block_on_cycle must be boolean")


@dataclass(frozen=True)
class StructureQualityDecision:
    disposition: QualityDisposition
    reasons: tuple[str, ...]
    observation: StructureQualityObservation


def observe_structure(document: StructuredDocument, *, low_confidence_threshold: float = 0.50) -> StructureQualityObservation:
    """Derive only structural metrics already present in ``StructuredDocument``.

    Provider-specific OCR error rates, semantic correctness and human quality judgments
    are intentionally outside this helper and should be added as separate measured data.
    """

    if not isinstance(document, StructuredDocument):
        raise ValueError("document must be StructuredDocument")
    threshold = _unit(low_confidence_threshold, "low_confidence_threshold")
    regions = document.regions
    region_ids = {region.region_id for region in regions}
    touched = {edge.before_region_id for edge in document.reading_edges} | {
        edge.after_region_id for edge in document.reading_edges
    }
    reading_order_coverage = 1.0 if len(regions) <= 1 else len(touched & region_ids) / len(region_ids)

    text_kinds = {
        RegionKind.TITLE,
        RegionKind.HEADING,
        RegionKind.PARAGRAPH,
        RegionKind.LIST,
        RegionKind.CAPTION,
        RegionKind.FOOTNOTE,
    }
    text_confidences = [
        region.confidence for region in regions if region.kind in text_kinds and region.confidence is not None
    ]
    mean_confidence = None if not text_confidences else sum(text_confidences) / len(text_confidences)
    low_fraction = None if not text_confidences else sum(value < threshold for value in text_confidences) / len(text_confidences)

    table_regions = [region for region in regions if region.kind == RegionKind.TABLE]
    structured_table_ids = {table.table_region_id for table in document.tables}
    table_structure_coverage = 1.0 if not table_regions else sum(
        region.region_id in structured_table_ids for region in table_regions
    ) / len(table_regions)
    table_cells = [cell for table in document.tables for cell in table.cells]
    table_nonempty = None if not table_cells else sum(bool(cell.text.strip()) for cell in table_cells) / len(table_cells)

    formula_regions = [region for region in regions if region.kind == RegionKind.FORMULA]
    represented_formula_ids = {formula.region_id for formula in document.formulas}
    formula_coverage = 1.0 if not formula_regions else sum(
        region.region_id in represented_formula_ids for region in formula_regions
    ) / len(formula_regions)

    figure_regions = [region for region in regions if region.kind == RegionKind.FIGURE]
    captioned_figure_ids = {figure.figure_region_id for figure in document.figures if figure.caption_region_ids}
    figure_caption_coverage = 1.0 if not figure_regions else sum(
        region.region_id in captioned_figure_ids for region in figure_regions
    ) / len(figure_regions)

    referenced: list[str] = []
    for figure in document.figures:
        referenced.extend(figure.caption_region_ids)
        referenced.extend(figure.cited_from_region_ids)
    referenced.extend(
        region.parent_region_id for region in regions if region.parent_region_id is not None
    )
    dangling = sum(reference not in region_ids for reference in referenced)

    cyclic = False
    try:
        _ = document.reading_order
    except ValueError:
        cyclic = True

    return StructureQualityObservation(
        reading_order_coverage=reading_order_coverage,
        text_region_confidence_mean=mean_confidence,
        low_confidence_text_fraction=low_fraction,
        table_structure_coverage=table_structure_coverage,
        table_nonempty_cell_fraction=table_nonempty,
        formula_representation_coverage=formula_coverage,
        figure_caption_coverage=figure_caption_coverage,
        dangling_reference_count=dangling,
        cyclic_reading_order=cyclic,
        extraction_error_count=0,
    )


def decide_structure_quality(
    observation: StructureQualityObservation,
    *,
    policy: StructureQualityPolicy = StructureQualityPolicy(),
) -> StructureQualityDecision:
    if not isinstance(observation, StructureQualityObservation):
        raise ValueError("observation must be StructureQualityObservation")
    if not isinstance(policy, StructureQualityPolicy):
        raise ValueError("policy must be StructureQualityPolicy")

    review_reasons: list[str] = []
    block_reasons: list[str] = []
    if observation.cyclic_reading_order and policy.block_on_cycle:
        block_reasons.append("reading-order graph is cyclic")
    if observation.dangling_reference_count > policy.maximum_dangling_references:
        block_reasons.append("dangling structural references exceed policy")
    if observation.extraction_error_count > policy.maximum_extraction_errors:
        block_reasons.append("extraction errors exceed policy")

    checks = (
        (observation.reading_order_coverage, policy.minimum_reading_order_coverage, "reading-order coverage"),
        (observation.table_structure_coverage, policy.minimum_table_structure_coverage, "table structure coverage"),
        (observation.formula_representation_coverage, policy.minimum_formula_representation_coverage, "formula representation coverage"),
        (observation.figure_caption_coverage, policy.minimum_figure_caption_coverage, "figure-caption coverage"),
    )
    for observed, required, label in checks:
        if observed < required:
            review_reasons.append(f"{label} {observed:.4f} < {required:.4f}")

    if observation.text_region_confidence_mean is None:
        review_reasons.append("text-region confidence is unavailable")
    elif observation.text_region_confidence_mean < policy.minimum_text_confidence_mean:
        review_reasons.append("mean text-region confidence is below policy")
    if observation.low_confidence_text_fraction is None:
        review_reasons.append("low-confidence text fraction is unavailable")
    elif observation.low_confidence_text_fraction > policy.maximum_low_confidence_text_fraction:
        review_reasons.append("low-confidence text fraction exceeds policy")
    if observation.table_nonempty_cell_fraction is not None and (
        observation.table_nonempty_cell_fraction < policy.minimum_table_nonempty_cell_fraction
    ):
        review_reasons.append("non-empty table-cell fraction is below policy")

    if block_reasons:
        return StructureQualityDecision(QualityDisposition.BLOCK, tuple(block_reasons + review_reasons), observation)
    if review_reasons:
        return StructureQualityDecision(QualityDisposition.REVIEW, tuple(review_reasons), observation)
    return StructureQualityDecision(QualityDisposition.ACCEPT, ("all structure quality gates passed",), observation)


__all__ = [
    "QualityDisposition",
    "StructureQualityDecision",
    "StructureQualityObservation",
    "StructureQualityPolicy",
    "decide_structure_quality",
    "observe_structure",
]
