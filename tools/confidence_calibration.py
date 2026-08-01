"""Dependency-free confidence calibration and abstention analysis."""

from __future__ import annotations

import bisect
import itertools
import math
import operator
from dataclasses import dataclass
from typing import Any, Iterable

_MAX_EXAMPLES = 100_000
_MAX_BINS = 100


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be finite and between 0 and 1.")
    return parsed


@dataclass(frozen=True)
class CalibrationExample:
    confidence: float
    correct: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "confidence",
            _probability(self.confidence, "confidence"),
        )
        if not isinstance(self.correct, bool):
            raise ValueError("correct must be a boolean.")


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float
    gap: float


@dataclass(frozen=True)
class CalibrationReport:
    example_count: int
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_gap: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class IsotonicBlock:
    lower: float
    upper: float
    calibrated_probability: float
    count: int


@dataclass(frozen=True)
class IsotonicCalibrator:
    blocks: tuple[IsotonicBlock, ...]

    def calibrate(self, confidence: float) -> float:
        value = _probability(confidence, "confidence")
        if not self.blocks:
            return value
        upper_bounds = [block.upper for block in self.blocks]
        index = min(bisect.bisect_left(upper_bounds, value), len(self.blocks) - 1)
        return self.blocks[index].calibrated_probability


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    selected: int


def _examples(values: Iterable[CalibrationExample]) -> list[CalibrationExample]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("examples must be an iterable of CalibrationExample values.")
    try:
        rows = list(itertools.islice(iter(values), _MAX_EXAMPLES + 1))
    except Exception as exc:
        raise ValueError("examples are not safely iterable.") from exc
    if len(rows) > _MAX_EXAMPLES:
        raise ValueError("calibration example limit exceeded.")
    if any(not isinstance(row, CalibrationExample) for row in rows):
        raise ValueError("every example must be a CalibrationExample.")
    return rows


def reliability_report(
    values: Iterable[CalibrationExample],
    *,
    bin_count: int = 10,
) -> CalibrationReport:
    rows = _examples(values)
    bins = _integer(bin_count, "bin_count", 1, _MAX_BINS)
    if not rows:
        return CalibrationReport(0, 0.0, 0.0, 0.0, ())
    buckets: list[list[CalibrationExample]] = [[] for _ in range(bins)]
    for row in rows:
        index = min(int(row.confidence * bins), bins - 1)
        buckets[index].append(row)
    rendered: list[CalibrationBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean = sum(row.confidence for row in bucket) / len(bucket)
        accuracy = sum(1.0 for row in bucket if row.correct) / len(bucket)
        gap = abs(mean - accuracy)
        weighted_gap += gap * len(bucket) / len(rows)
        maximum_gap = max(maximum_gap, gap)
        rendered.append(
            CalibrationBin(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=len(bucket),
                mean_confidence=round(mean, 9),
                accuracy=round(accuracy, 9),
                gap=round(gap, 9),
            )
        )
    brier = sum(
        (row.confidence - (1.0 if row.correct else 0.0)) ** 2
        for row in rows
    ) / len(rows)
    return CalibrationReport(
        example_count=len(rows),
        brier_score=round(brier, 9),
        expected_calibration_error=round(weighted_gap, 9),
        maximum_calibration_gap=round(maximum_gap, 9),
        bins=tuple(rendered),
    )


def fit_isotonic_calibrator(
    values: Iterable[CalibrationExample],
) -> IsotonicCalibrator:
    rows = sorted(_examples(values), key=lambda row: row.confidence)
    if not rows:
        return IsotonicCalibrator(())
    grouped: list[dict[str, float | int]] = []
    for confidence, group in itertools.groupby(rows, key=lambda row: row.confidence):
        values_at_confidence = list(group)
        grouped.append(
            {
                "lower": confidence,
                "upper": confidence,
                "count": len(values_at_confidence),
                "positive": sum(1 for row in values_at_confidence if row.correct),
            }
        )
    stack: list[dict[str, float | int]] = []
    for block in grouped:
        stack.append(block)
        while len(stack) >= 2:
            left = stack[-2]
            right = stack[-1]
            left_rate = float(left["positive"]) / int(left["count"])
            right_rate = float(right["positive"]) / int(right["count"])
            if left_rate <= right_rate:
                break
            stack[-2:] = [
                {
                    "lower": left["lower"],
                    "upper": right["upper"],
                    "count": int(left["count"]) + int(right["count"]),
                    "positive": int(left["positive"]) + int(right["positive"]),
                }
            ]
    return IsotonicCalibrator(
        tuple(
            IsotonicBlock(
                lower=float(block["lower"]),
                upper=float(block["upper"]),
                calibrated_probability=(
                    int(block["positive"]) / int(block["count"])
                ),
                count=int(block["count"]),
            )
            for block in stack
        )
    )


def risk_coverage_curve(
    values: Iterable[CalibrationExample],
) -> tuple[RiskCoveragePoint, ...]:
    rows = sorted(
        _examples(values),
        key=lambda row: row.confidence,
        reverse=True,
    )
    if not rows:
        return ()
    points: list[RiskCoveragePoint] = []
    correct = 0
    for index, row in enumerate(rows, start=1):
        correct += int(row.correct)
        next_confidence = rows[index].confidence if index < len(rows) else -1.0
        if next_confidence == row.confidence:
            continue
        points.append(
            RiskCoveragePoint(
                threshold=row.confidence,
                coverage=round(index / len(rows), 9),
                risk=round(1.0 - correct / index, 9),
                selected=index,
            )
        )
    return tuple(points)


def select_abstention_threshold(
    values: Iterable[CalibrationExample],
    *,
    maximum_risk: float,
    minimum_coverage: float = 0.0,
) -> RiskCoveragePoint | None:
    risk_limit = _probability(maximum_risk, "maximum_risk")
    coverage_floor = _probability(minimum_coverage, "minimum_coverage")
    eligible = [
        point
        for point in risk_coverage_curve(values)
        if point.risk <= risk_limit and point.coverage >= coverage_floor
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda point: (point.coverage, -point.risk, -point.threshold))


__all__ = [
    "CalibrationBin",
    "CalibrationExample",
    "CalibrationReport",
    "IsotonicBlock",
    "IsotonicCalibrator",
    "RiskCoveragePoint",
    "fit_isotonic_calibrator",
    "reliability_report",
    "risk_coverage_curve",
    "select_abstention_threshold",
]
