"""Privacy-safe observability for active-learning acquisition and gold yield."""

from __future__ import annotations

import math

from evaluation.active_learning import ActiveLearningBatch
from evaluation.active_learning_gold import ActiveLearningGoldManifest
from evaluation.quality_observability import MetricObservation
from orchestration.active_learning_adjudication import ActiveLearningMaterializationReceipt


def _count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def observations_from_active_learning_batch(
    batch: ActiveLearningBatch,
    *,
    candidate_count: int,
) -> tuple[MetricObservation, ...]:
    if not isinstance(batch, ActiveLearningBatch):
        raise ValueError("batch must be ActiveLearningBatch")
    total_candidates = _count(candidate_count, "candidate_count")
    if total_candidates < len(batch.selected):
        raise ValueError("candidate_count may not be smaller than selected count")
    selected_count = len(batch.selected)
    selection_rate = selected_count / total_candidates if total_candidates else 0.0
    scores = [row.acquisition_score for row in batch.selected]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0
    tags = (("metric_family", "active_learning"), ("variant", "selection"))
    source = "evaluation.active_learning_observability"
    return (
        MetricObservation("active_learning.candidate_count", float(total_candidates), "neutral", "count", total_candidates, source, tags),
        MetricObservation("active_learning.selected_count", float(selected_count), "neutral", "count", total_candidates, source, tags),
        MetricObservation("active_learning.selection_rate", selection_rate, "neutral", "ratio", total_candidates, source, tags),
        MetricObservation("active_learning.estimated_label_cost", batch.total_estimated_cost, "lower", "cost_unit", selected_count, source, tags),
        MetricObservation("active_learning.mean_acquisition_score", mean_score, "neutral", "score", selected_count, source, tags),
        MetricObservation("active_learning.max_acquisition_score", max_score, "neutral", "score", selected_count, source, tags),
    )


def observations_from_active_learning_materialization(
    receipt: ActiveLearningMaterializationReceipt,
) -> tuple[MetricObservation, ...]:
    if not isinstance(receipt, ActiveLearningMaterializationReceipt):
        raise ValueError("receipt must be ActiveLearningMaterializationReceipt")
    tags = (("metric_family", "active_learning"), ("variant", "materialization"))
    return (
        MetricObservation(
            "active_learning.materialized_case_count",
            float(len(receipt.cases)),
            "neutral",
            "count",
            len(receipt.cases),
            "evaluation.active_learning_observability",
            tags,
        ),
    )


def observations_from_active_learning_gold(
    manifest: ActiveLearningGoldManifest,
    *,
    materialized_case_count: int,
) -> tuple[MetricObservation, ...]:
    if not isinstance(manifest, ActiveLearningGoldManifest):
        raise ValueError("manifest must be ActiveLearningGoldManifest")
    materialized = _count(materialized_case_count, "materialized_case_count")
    if materialized < len(manifest.examples):
        raise ValueError("materialized_case_count may not be smaller than gold example count")
    yield_rate = len(manifest.examples) / materialized if materialized else 0.0
    rounds = [row.round_index for row in manifest.examples]
    revisions = [row.resolution_revision for row in manifest.examples]
    mean_round = sum(rounds) / len(rounds)
    mean_revision = sum(revisions) / len(revisions)
    if not math.isfinite(mean_round) or not math.isfinite(mean_revision):
        raise ValueError("gold resolution summary is not finite")
    base_tags = (("metric_family", "active_learning"), ("variant", "gold"))
    observations = [
        MetricObservation("active_learning.gold_count", float(len(manifest.examples)), "higher", "count", materialized, "evaluation.active_learning_observability", base_tags),
        MetricObservation("active_learning.gold_yield_rate", yield_rate, "higher", "ratio", materialized, "evaluation.active_learning_observability", base_tags),
        MetricObservation("active_learning.mean_resolution_round", mean_round, "lower", "round", len(manifest.examples), "evaluation.active_learning_observability", base_tags),
        MetricObservation("active_learning.mean_resolution_revision", mean_revision, "neutral", "revision", len(manifest.examples), "evaluation.active_learning_observability", base_tags),
    ]
    by_task: dict[str, int] = {}
    for row in manifest.examples:
        by_task[row.task_id] = by_task.get(row.task_id, 0) + 1
    for task_id in sorted(by_task):
        count = by_task[task_id]
        observations.append(
            MetricObservation(
                "active_learning.gold_task_count",
                float(count),
                "neutral",
                "count",
                count,
                "evaluation.active_learning_observability",
                (("metric_family", "active_learning"), ("variant", "gold"), ("task_id", task_id)),
            )
        )
    return tuple(observations)


__all__ = [
    "observations_from_active_learning_batch",
    "observations_from_active_learning_gold",
    "observations_from_active_learning_materialization",
]
