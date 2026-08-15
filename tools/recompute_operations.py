"""Operator-facing orchestration helpers for research recomputation.

These helpers never start background work. A deployment explicitly calls them from a
worker/CLI. They return bounded summaries without private query text or citation content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tools.recompute_executor import (
    RecomputeOutcome,
    ResearchRecomputeExecutor,
    requeue_failed_task,
)
from tools.security import normalize_owner_id


@dataclass(frozen=True)
class RecomputeCycleSummary:
    owner_id: str
    attempted: int
    succeeded: int
    failed: int
    replacements: int
    outcomes: tuple[dict[str, object], ...]


def run_recompute_cycle(
    executor: ResearchRecomputeExecutor,
    owner_id: str,
    *,
    max_tasks: int = 100,
) -> RecomputeCycleSummary:
    if not isinstance(executor, ResearchRecomputeExecutor):
        raise TypeError("executor must be ResearchRecomputeExecutor")
    owner = normalize_owner_id(owner_id)
    values = executor.drain(owner, max_tasks=max_tasks)
    rows: list[dict[str, object]] = []
    succeeded = 0
    failed = 0
    replacements = 0
    for value in values:
        if value.success:
            succeeded += 1
        else:
            failed += 1
        if value.replacement is not None and value.replacement != value.task.artifact:
            replacements += 1
        rows.append(
            {
                "task_id": value.task.task_id,
                "artifact_kind": value.task.artifact.kind,
                "artifact_id": value.task.artifact.resource_id,
                "success": value.success,
                "replacement_id": value.replacement.resource_id if value.replacement else "",
                "error_type": value.error_type,
            }
        )
    return RecomputeCycleSummary(
        owner,
        len(values),
        succeeded,
        failed,
        replacements,
        tuple(rows),
    )


def retry_failed_recompute(
    executor: ResearchRecomputeExecutor,
    owner_id: str,
    task_id: str,
) -> bool:
    if not isinstance(executor, ResearchRecomputeExecutor):
        raise TypeError("executor must be ResearchRecomputeExecutor")
    return requeue_failed_task(executor.invalidations, owner_id, task_id)


__all__ = [
    "RecomputeCycleSummary",
    "retry_failed_recompute",
    "run_recompute_cycle",
]
