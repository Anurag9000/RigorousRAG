"""Backend-neutral production wrapper for the research recompute executor."""
from __future__ import annotations

from tools.recompute_executor import ResearchRecomputeExecutor
from tools.recompute_ledger_ops import load_recompute_task
from tools.security import normalize_owner_id


class ProductionResearchRecomputeExecutor(ResearchRecomputeExecutor):
    """Use the selected invalidation backend when reloading an exact claimed task."""

    def _load_claimed_task(self, owner_id: str, task_id: str):
        task = load_recompute_task(self.invalidations, normalize_owner_id(owner_id), task_id)
        if task.status != "claimed":
            raise RuntimeError("recompute task must be claimed before execution")
        return task


__all__ = ["ProductionResearchRecomputeExecutor"]
