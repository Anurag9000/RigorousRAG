"""Finite-sample conformal calibration for retrieval and selective answering."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

_MAX_SAMPLES = 1_000_000


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite.")
    return parsed


def _unit(value: Any, label: str) -> float:
    parsed = _finite(value, label)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def _values(values: Iterable[Any], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a numeric iterable.")
    result: list[float] = []
    try:
        iterator = iter(values)
    except Exception as exc:
        raise ValueError(f"{label} must be safely iterable.") from exc
    for value in iterator:
        if len(result) >= _MAX_SAMPLES:
            raise ValueError(f"{label} exceeds the sample limit.")
        result.append(_finite(value, label))
    if not result:
        raise ValueError(f"{label} must not be empty.")
    return tuple(result)


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    threshold: float
    calibration_size: int

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha


def fit_nonconformity_threshold(
    nonconformity_scores: Iterable[Any],
    *,
    alpha: float = 0.1,
) -> ConformalCalibration:
    """Fit the standard finite-sample split-conformal quantile.

    The selected order statistic is ``ceil((n + 1) * (1 - alpha))`` capped at ``n``.
    This is the conservative finite-sample correction used by split conformal methods.
    """

    values = tuple(sorted(_values(nonconformity_scores, "nonconformity_scores")))
    selected_alpha = _unit(alpha, "alpha")
    if not 0.0 < selected_alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1.")
    rank = min(len(values), int(math.ceil((len(values) + 1) * (1.0 - selected_alpha))))
    threshold = values[rank - 1]
    return ConformalCalibration(selected_alpha, threshold, len(values))


def fit_retrieval_calibration(
    relevant_scores: Iterable[Any],
    *,
    alpha: float = 0.1,
) -> ConformalCalibration:
    """Calibrate a retrieval set from scores of known-relevant evidence.

    Retrieval scores are assumed to be normalized to ``[0, 1]`` and converted to the
    nonconformity score ``1 - score``.
    """

    scores = tuple(_unit(value, "relevant score") for value in _values(relevant_scores, "relevant_scores"))
    return fit_nonconformity_threshold((1.0 - value for value in scores), alpha=alpha)


def conformal_retrieval_set(
    candidate_scores: Mapping[str, Any],
    calibration: ConformalCalibration,
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """Return candidate IDs whose nonconformity does not exceed the fitted threshold."""

    if not isinstance(candidate_scores, Mapping) or len(candidate_scores) > _MAX_SAMPLES:
        raise ValueError("candidate_scores must be a bounded mapping.")
    if not isinstance(calibration, ConformalCalibration):
        raise ValueError("calibration must be ConformalCalibration.")
    if limit is not None:
        if isinstance(limit, bool):
            raise ValueError("limit must be an integer.")
        try:
            selected_limit = int(operator.index(limit))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("limit must be an integer.") from exc
        if not 1 <= selected_limit <= _MAX_SAMPLES:
            raise ValueError("limit is out of range.")
    else:
        selected_limit = _MAX_SAMPLES
    rows: list[tuple[str, float]] = []
    for identifier, raw_score in candidate_scores.items():
        if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 500:
            raise ValueError("candidate_scores contains an invalid ID.")
        score = _unit(raw_score, "candidate score")
        if 1.0 - score <= calibration.threshold:
            rows.append((identifier.strip(), score))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return tuple(identifier for identifier, _score in rows[:selected_limit])


def empirical_set_coverage(
    prediction_sets: Sequence[Sequence[str]],
    relevant_ids: Sequence[Sequence[str]],
) -> float:
    if isinstance(prediction_sets, (str, bytes, bytearray)) or isinstance(relevant_ids, (str, bytes, bytearray)):
        raise ValueError("prediction_sets and relevant_ids must be sequences.")
    if len(prediction_sets) != len(relevant_ids) or not prediction_sets:
        raise ValueError("prediction_sets and relevant_ids must have equal non-zero length.")
    covered = 0
    for predicted, relevant in zip(prediction_sets, relevant_ids):
        predicted_set = set(predicted)
        relevant_set = set(relevant)
        covered += bool(predicted_set & relevant_set)
    return covered / len(prediction_sets)


__all__ = [
    "ConformalCalibration",
    "conformal_retrieval_set",
    "empirical_set_coverage",
    "fit_nonconformity_threshold",
    "fit_retrieval_calibration",
]
