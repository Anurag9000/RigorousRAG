"""Answer and retrieval stability under corpus/index perturbations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


_SPACE = re.compile(r"\s+")


def normalized_answer_hash(text: str) -> str:
    normalized = _SPACE.sub(" ", str(text)).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def jaccard_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def rank_biased_overlap(
    left: Sequence[str],
    right: Sequence[str],
    *,
    persistence: float = 0.9,
) -> float:
    """Finite rank-biased overlap for top-weighted retrieval stability."""

    if not 0.0 < persistence < 1.0:
        raise ValueError("persistence must be in (0, 1).")
    depth = max(len(left), len(right))
    if depth == 0:
        return 1.0
    left_seen = set()
    right_seen = set()
    total = 0.0
    weight_sum = 0.0
    for index in range(depth):
        if index < len(left):
            left_seen.add(left[index])
        if index < len(right):
            right_seen.add(right[index])
        agreement = len(left_seen & right_seen) / (index + 1)
        weight = (1.0 - persistence) * persistence**index
        total += weight * agreement
        weight_sum += weight
    return total / weight_sum if weight_sum else 1.0


@dataclass(frozen=True)
class PerturbationRun:
    name: str
    retrieved_ids: tuple[str, ...]
    answer: str


@dataclass(frozen=True)
class StabilityReport:
    baseline: str
    retrieval_jaccard_mean: float
    retrieval_rbo_mean: float
    exact_answer_stability: float
    runs: int


def stability_report(
    runs: Sequence[PerturbationRun],
    *,
    baseline: str | None = None,
) -> StabilityReport:
    if not runs:
        raise ValueError("at least one perturbation run is required.")
    by_name: Mapping[str, PerturbationRun] = {run.name: run for run in runs}
    if len(by_name) != len(runs):
        raise ValueError("perturbation run names must be unique.")
    baseline_name = baseline or runs[0].name
    if baseline_name not in by_name:
        raise KeyError("baseline perturbation is unavailable.")
    base = by_name[baseline_name]
    comparisons = [run for run in runs if run.name != baseline_name]
    if not comparisons:
        return StabilityReport(baseline_name, 1.0, 1.0, 1.0, 1)
    base_hash = normalized_answer_hash(base.answer)
    jaccard = [jaccard_overlap(base.retrieved_ids, run.retrieved_ids) for run in comparisons]
    rbo = [rank_biased_overlap(base.retrieved_ids, run.retrieved_ids) for run in comparisons]
    exact = [normalized_answer_hash(run.answer) == base_hash for run in comparisons]
    return StabilityReport(
        baseline=baseline_name,
        retrieval_jaccard_mean=sum(jaccard) / len(jaccard),
        retrieval_rbo_mean=sum(rbo) / len(rbo),
        exact_answer_stability=sum(exact) / len(exact),
        runs=len(runs),
    )
