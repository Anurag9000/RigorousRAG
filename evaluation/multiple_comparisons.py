"""Multiple-comparison and non-inferiority gates for experiment promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AdjustedHypothesis:
    name: str
    raw_p_value: float
    adjusted_p_value: float
    rejected: bool


def _validated(p_values: Mapping[str, float]) -> list[tuple[str, float]]:
    rows = []
    for raw_name, raw_value in p_values.items():
        name = str(raw_name).strip()
        value = float(raw_value)
        if not name:
            raise ValueError("hypothesis names must be non-empty.")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("p-values must be finite and in [0, 1].")
        rows.append((name, value))
    if len({name for name, _ in rows}) != len(rows):
        raise ValueError("hypothesis names must be unique.")
    return rows


def holm_adjust(p_values: Mapping[str, float], *, alpha: float = 0.05) -> tuple[AdjustedHypothesis, ...]:
    """Holm step-down family-wise-error adjustment."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    ordered = sorted(_validated(p_values), key=lambda row: (row[1], row[0]))
    total = len(ordered)
    running = 0.0
    output = []
    for index, (name, raw) in enumerate(ordered):
        adjusted = min(1.0, (total - index) * raw)
        running = max(running, adjusted)
        output.append(AdjustedHypothesis(name, raw, running, running <= alpha))
    return tuple(output)


def benjamini_hochberg(
    p_values: Mapping[str, float], *, alpha: float = 0.05
) -> tuple[AdjustedHypothesis, ...]:
    """Benjamini-Hochberg false-discovery-rate adjustment."""

    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    ordered = sorted(_validated(p_values), key=lambda row: (row[1], row[0]))
    total = len(ordered)
    adjusted = [1.0] * total
    running = 1.0
    for index in range(total - 1, -1, -1):
        rank = index + 1
        running = min(running, min(1.0, ordered[index][1] * total / rank))
        adjusted[index] = running
    return tuple(
        AdjustedHypothesis(name, raw, corrected, corrected <= alpha)
        for (name, raw), corrected in zip(ordered, adjusted)
    )


@dataclass(frozen=True)
class NonInferiorityDecision:
    estimate: float
    confidence_low: float
    confidence_high: float
    margin: float
    higher_is_better: bool
    passed: bool


def noninferiority_gate(
    *,
    estimate: float,
    confidence_low: float,
    confidence_high: float,
    margin: float,
    higher_is_better: bool = True,
) -> NonInferiorityDecision:
    values = tuple(float(value) for value in (estimate, confidence_low, confidence_high, margin))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("non-inferiority inputs must be finite.")
    estimate, lower, upper, selected_margin = values
    if lower > upper:
        raise ValueError("confidence_low may not exceed confidence_high.")
    if selected_margin < 0.0:
        raise ValueError("margin must be non-negative.")
    passed = lower >= -selected_margin if higher_is_better else upper <= selected_margin
    return NonInferiorityDecision(
        estimate,
        lower,
        upper,
        selected_margin,
        bool(higher_is_better),
        passed,
    )
