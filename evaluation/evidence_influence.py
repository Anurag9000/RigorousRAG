"""Counterfactual evidence influence analysis for RAG answers and scores."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

_MAX_EVIDENCE = 100
_MAX_SHAPLEY_EVIDENCE = 8


def _identifier(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("evidence IDs must be strings.")
    text = value.strip()
    if not text or len(text) > 500:
        raise ValueError("evidence ID is invalid.")
    return text


def _score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("scorer must return a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scorer must return a finite number.") from exc
    if not math.isfinite(parsed):
        raise ValueError("scorer must return a finite number.")
    return parsed


def _evidence_ids(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("evidence_ids must be an iterable.")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if len(result) >= _MAX_EVIDENCE:
            raise ValueError("evidence_ids exceeds the evidence limit.")
        identifier = _identifier(value)
        if identifier not in seen:
            seen.add(identifier)
            result.append(identifier)
    if not result:
        raise ValueError("evidence_ids must not be empty.")
    return tuple(result)


EvidenceScorer = Callable[[Sequence[str]], float]


@dataclass(frozen=True)
class EvidenceInfluence:
    evidence_id: str
    full_score: float
    removed_score: float
    influence: float


def leave_one_out_influence(
    evidence_ids: Iterable[str],
    scorer: EvidenceScorer,
) -> tuple[EvidenceInfluence, ...]:
    """Measure the score change caused by removing each evidence item."""

    ids = _evidence_ids(evidence_ids)
    if not callable(scorer):
        raise ValueError("scorer must be callable.")
    full_score = _score(scorer(ids))
    rows: list[EvidenceInfluence] = []
    for identifier in ids:
        subset = tuple(item for item in ids if item != identifier)
        removed_score = _score(scorer(subset))
        rows.append(EvidenceInfluence(identifier, full_score, removed_score, full_score - removed_score))
    rows.sort(key=lambda row: (-abs(row.influence), row.evidence_id))
    return tuple(rows)


@dataclass(frozen=True)
class ShapleyInfluence:
    evidence_id: str
    value: float


def exact_shapley_influence(
    evidence_ids: Iterable[str],
    scorer: EvidenceScorer,
) -> tuple[ShapleyInfluence, ...]:
    """Compute exact Shapley evidence contributions for at most eight evidence items."""

    ids = _evidence_ids(evidence_ids)
    if len(ids) > _MAX_SHAPLEY_EVIDENCE:
        raise ValueError("exact Shapley influence is limited to eight evidence items.")
    if not callable(scorer):
        raise ValueError("scorer must be callable.")
    cache: dict[tuple[str, ...], float] = {}

    def evaluate(subset: tuple[str, ...]) -> float:
        key = tuple(sorted(subset))
        if key not in cache:
            cache[key] = _score(scorer(key))
        return cache[key]

    count = len(ids)
    factorial = math.factorial
    normalizer = factorial(count)
    values: dict[str, float] = {identifier: 0.0 for identifier in ids}
    for identifier in ids:
        others = tuple(item for item in ids if item != identifier)
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(count - size - 1) / normalizer
            for subset in itertools.combinations(others, size):
                with_item = tuple(sorted((*subset, identifier)))
                without = tuple(sorted(subset))
                values[identifier] += weight * (evaluate(with_item) - evaluate(without))
    return tuple(
        ShapleyInfluence(identifier, values[identifier])
        for identifier in sorted(ids, key=lambda item: (-abs(values[item]), item))
    )


__all__ = [
    "EvidenceInfluence",
    "ShapleyInfluence",
    "exact_shapley_influence",
    "leave_one_out_influence",
]
