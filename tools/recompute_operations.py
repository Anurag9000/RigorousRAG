"""Operator-facing orchestration helpers for research recomputation.

These helpers never start background work. A deployment explicitly calls them from a
worker/CLI. They return bounded summaries without private query text or citation content.
"""

from __future__ import annotations

from dataclasses import dataclass

from tools.distributed_recompute import DistributedRecomputeBridge
from tools.recompute_executor import ResearchRecomputeExecutor, requeue_failed_task
from tools.security import normalize_owner_id


@dataclass(frozen=True)
class RecomputeCycleSummary:
    owner_id: str
    attempted: int
    succeeded: int
    failed: int
    replacements: int
    outcomes: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class RecomputePublishSummary:
    owner_id: str
    handoffs: int


@dataclass(frozen=True)
class DistributedRecomputeCycleSummary:
    owner_id: str
    worker_id: str
    attempted: int
    completed: int
    failed: int
    busy: int
    duplicates: int
    invalid: int
    outcomes: tuple[dict[str, object], ...]


def _max_tasks(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
        raise ValueError("max_tasks is invalid")
    return value


def run_recompute_cycle(
    executor: ResearchRecomputeExecutor,
    owner_id: str,
    *,
    max_tasks: int = 100,
) -> RecomputeCycleSummary:
    if not isinstance(executor, ResearchRecomputeExecutor):
        raise TypeError("executor must be ResearchRecomputeExecutor")
    owner = normalize_owner_id(owner_id)
    values = executor.drain(owner, max_tasks=_max_tasks(max_tasks))
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


def publish_recompute_tasks(
    bridge: DistributedRecomputeBridge,
    *,
    limit: int = 1000,
) -> RecomputePublishSummary:
    if not isinstance(bridge, DistributedRecomputeBridge):
        raise TypeError("bridge must be DistributedRecomputeBridge")
    message_ids = bridge.publish_ready(limit=_max_tasks(limit))
    return RecomputePublishSummary(bridge.owner_id, len(message_ids))


def run_distributed_recompute_cycle(
    bridge: DistributedRecomputeBridge,
    *,
    worker_id: str,
    max_tasks: int = 100,
    visibility_timeout: float = 1800.0,
    busy_retry_delay: float = 30.0,
) -> DistributedRecomputeCycleSummary:
    if not isinstance(bridge, DistributedRecomputeBridge):
        raise TypeError("bridge must be DistributedRecomputeBridge")
    limit = _max_tasks(max_tasks)
    if not isinstance(worker_id, str) or not worker_id.strip() or len(worker_id.strip()) > 256:
        raise ValueError("worker_id is invalid")
    worker = worker_id.strip()
    rows: list[dict[str, object]] = []
    counts = {"completed": 0, "failed": 0, "busy": 0, "duplicate": 0, "invalid": 0}
    for _ in range(limit):
        value = bridge.work_one(
            worker_id=worker,
            visibility_timeout=visibility_timeout,
            busy_retry_delay=busy_retry_delay,
        )
        if value.state == "idle":
            break
        if value.state not in counts:
            raise RuntimeError(f"unexpected distributed recompute state: {value.state}")
        counts[value.state] += 1
        row: dict[str, object] = {
            "task_id": value.task_id,
            "state": value.state,
        }
        if value.outcome is not None:
            row["success"] = value.outcome.success
            row["error_type"] = value.outcome.error_type
        rows.append(row)
    return DistributedRecomputeCycleSummary(
        owner_id=bridge.owner_id,
        worker_id=worker,
        attempted=len(rows),
        completed=counts["completed"],
        failed=counts["failed"],
        busy=counts["busy"],
        duplicates=counts["duplicate"],
        invalid=counts["invalid"],
        outcomes=tuple(rows),
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
    "DistributedRecomputeCycleSummary",
    "RecomputeCycleSummary",
    "RecomputePublishSummary",
    "publish_recompute_tasks",
    "retry_failed_recompute",
    "run_distributed_recompute_cycle",
    "run_recompute_cycle",
]
