"""Deterministic adversarial robustness diagnostics for retrieval and citation systems."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

_MAX_RESULTS = 100_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _ids(values: Iterable[Any], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable of IDs.")
    result: list[str] = []
    for value in values:
        if len(result) >= _MAX_RESULTS:
            raise ValueError(f"{label} exceeds the result limit.")
        result.append(_identifier(value, label))
    return tuple(result)


@dataclass(frozen=True)
class CounterfactualCitationReport:
    cited_count: int
    expected_count: int
    decoy_count: int
    expected_citation_fraction: float
    decoy_citation_rate: float
    unknown_citation_rate: float


def counterfactual_citation_report(
    *,
    expected_source_ids: Iterable[str],
    decoy_source_ids: Iterable[str],
    cited_source_ids: Iterable[str],
) -> CounterfactualCitationReport:
    expected = set(_ids(expected_source_ids, "expected_source_id"))
    decoys = set(_ids(decoy_source_ids, "decoy_source_id"))
    cited = set(_ids(cited_source_ids, "cited_source_id"))
    if expected & decoys:
        raise ValueError("expected and decoy citation sets must not overlap.")
    expected_hits = cited & expected
    decoy_hits = cited & decoys
    unknown = cited - expected - decoys
    denominator = len(cited)
    return CounterfactualCitationReport(
        cited_count=denominator,
        expected_count=len(expected_hits),
        decoy_count=len(decoy_hits),
        expected_citation_fraction=(len(expected_hits) / len(expected) if expected else 0.0),
        decoy_citation_rate=(len(decoy_hits) / denominator if denominator else 0.0),
        unknown_citation_rate=(len(unknown) / denominator if denominator else 0.0),
    )


@dataclass(frozen=True)
class RankingRobustnessReport:
    k: int
    overlap_at_k: float
    clean_recall_at_k: float
    perturbed_recall_at_k: float
    recall_drop: float


def metadata_poisoning_report(
    *,
    clean_ranking: Sequence[str],
    perturbed_ranking: Sequence[str],
    relevant_ids: Iterable[str],
    k: int = 10,
) -> RankingRobustnessReport:
    selected_k = _integer(k, "k", 1, _MAX_RESULTS)
    clean = _ids(clean_ranking[:selected_k], "clean_result_id")
    perturbed = _ids(perturbed_ranking[:selected_k], "perturbed_result_id")
    relevant = set(_ids(relevant_ids, "relevant_id"))
    clean_set = set(clean)
    perturbed_set = set(perturbed)
    union = clean_set | perturbed_set
    overlap = len(clean_set & perturbed_set) / len(union) if union else 1.0
    clean_recall = len(clean_set & relevant) / len(relevant) if relevant else 0.0
    perturbed_recall = len(perturbed_set & relevant) / len(relevant) if relevant else 0.0
    return RankingRobustnessReport(
        k=selected_k,
        overlap_at_k=overlap,
        clean_recall_at_k=clean_recall,
        perturbed_recall_at_k=perturbed_recall,
        recall_drop=max(0.0, clean_recall - perturbed_recall),
    )


@dataclass(frozen=True)
class PositionBiasReport:
    positions: tuple[int, ...]
    mean_scores: tuple[float, ...]
    score_range: float
    edge_to_middle_gap: float
    monotonic_slope: float


def long_context_position_report(
    scores_by_position: Mapping[int, Sequence[Any]],
) -> PositionBiasReport:
    if not isinstance(scores_by_position, Mapping) or not scores_by_position:
        raise ValueError("scores_by_position must be a non-empty mapping.")
    if len(scores_by_position) > 10_000:
        raise ValueError("scores_by_position exceeds the position limit.")
    rows: list[tuple[int, float]] = []
    for raw_position, raw_scores in scores_by_position.items():
        position = _integer(raw_position, "position", 0, 10_000_000)
        if isinstance(raw_scores, (str, bytes, bytearray)) or not raw_scores:
            raise ValueError("each position requires a non-empty score sequence.")
        if len(raw_scores) > _MAX_RESULTS:
            raise ValueError("position score sequence exceeds the sample limit.")
        values = [_unit(value, "position score") for value in raw_scores]
        rows.append((position, sum(values) / len(values)))
    rows.sort()
    if len({position for position, _ in rows}) != len(rows):
        raise ValueError("positions must be unique.")
    positions = tuple(position for position, _ in rows)
    means = tuple(score for _, score in rows)
    score_range = max(means) - min(means)
    middle_index = len(means) // 2
    edge_mean = (means[0] + means[-1]) / 2.0
    middle = means[middle_index]
    if len(rows) == 1:
        slope = 0.0
    else:
        x_mean = sum(positions) / len(positions)
        y_mean = sum(means) / len(means)
        denominator = sum((value - x_mean) ** 2 for value in positions)
        slope = (
            sum((x - x_mean) * (y - y_mean) for x, y in rows) / denominator
            if denominator > 0.0
            else 0.0
        )
    return PositionBiasReport(
        positions=positions,
        mean_scores=means,
        score_range=score_range,
        edge_to_middle_gap=edge_mean - middle,
        monotonic_slope=slope,
    )


@dataclass(frozen=True)
class DocumentVersionHit:
    result_id: str
    logical_document_id: str
    version: int
    current_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "result_id", _identifier(self.result_id, "result_id"))
        object.__setattr__(
            self,
            "logical_document_id",
            _identifier(self.logical_document_id, "logical_document_id"),
        )
        version = _integer(self.version, "version", 1, 2**63 - 1)
        current = _integer(self.current_version, "current_version", 1, 2**63 - 1)
        if version > current:
            raise ValueError("retrieved version may not exceed the declared current version.")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "current_version", current)


@dataclass(frozen=True)
class VersionRobustnessReport:
    result_count: int
    unique_logical_documents: int
    duplicate_result_rate: float
    stale_result_rate: float
    stale_logical_document_rate: float


def stale_duplicate_report(
    results: Sequence[DocumentVersionHit],
) -> VersionRobustnessReport:
    if isinstance(results, (str, bytes, bytearray)) or len(results) > _MAX_RESULTS:
        raise ValueError("results must be a bounded sequence.")
    if not results:
        return VersionRobustnessReport(0, 0, 0.0, 0.0, 0.0)
    if any(not isinstance(item, DocumentVersionHit) for item in results):
        raise ValueError("every result must be DocumentVersionHit.")
    result_ids = [item.result_id for item in results]
    if len(set(result_ids)) != len(result_ids):
        raise ValueError("result IDs must be unique.")
    logical_counts: dict[str, int] = {}
    stale_logical: set[str] = set()
    stale = 0
    for item in results:
        logical_counts[item.logical_document_id] = logical_counts.get(item.logical_document_id, 0) + 1
        if item.version < item.current_version:
            stale += 1
            stale_logical.add(item.logical_document_id)
    duplicate_count = sum(max(0, count - 1) for count in logical_counts.values())
    return VersionRobustnessReport(
        result_count=len(results),
        unique_logical_documents=len(logical_counts),
        duplicate_result_rate=duplicate_count / len(results),
        stale_result_rate=stale / len(results),
        stale_logical_document_rate=(
            len(stale_logical) / len(logical_counts) if logical_counts else 0.0
        ),
    )


__all__ = [
    "CounterfactualCitationReport",
    "DocumentVersionHit",
    "PositionBiasReport",
    "RankingRobustnessReport",
    "VersionRobustnessReport",
    "counterfactual_citation_report",
    "long_context_position_report",
    "metadata_poisoning_report",
    "stale_duplicate_report",
]
