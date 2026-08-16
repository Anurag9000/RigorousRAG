"""Route an already-claimed authoritative task to its governed executor by artifact kind."""
from __future__ import annotations

from tools.dependency_invalidation import DependencyRef
from tools.hydrology_recompute_executor import HydrologyRecomputeExecutor
from tools.recompute_executor import RecomputeOutcome, ResearchRecomputeExecutor
from tools.recompute_ledger_ops import load_recompute_task
from tools.security import normalize_owner_id

_HYDROLOGY_KINDS = frozenset({"hydrology_plan", "hydrology_projection", "hydrology_report"})


class RoutedRecomputeExecutor:
    def __init__(
        self,
        *,
        invalidations,
        research: ResearchRecomputeExecutor,
        hydrology: HydrologyRecomputeExecutor,
    ) -> None:
        self.invalidations = invalidations
        self.research = research
        self.hydrology = hydrology

    def execute_claimed(self, owner_id: str, task_id: str) -> RecomputeOutcome:
        owner = normalize_owner_id(owner_id)
        task = load_recompute_task(self.invalidations, owner, task_id)
        if task.status != "claimed":
            raise RuntimeError("recompute task must be claimed before execution")
        if task.artifact.kind not in _HYDROLOGY_KINDS:
            return self.research.execute_claimed(owner, task.task_id)
        value = self.hydrology.execute_claimed(owner, task)
        replacement = None
        if value.new_fingerprint and value.new_fingerprint != task.artifact.resource_id:
            replacement = DependencyRef(task.artifact.kind, value.new_fingerprint)
        return RecomputeOutcome(
            task=task,
            success=value.status == "completed",
            replacement=replacement,
            error_type=value.error_type,
        )


__all__ = ["RoutedRecomputeExecutor"]
