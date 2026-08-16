"""Operator-only deterministic recomputation for stale hydrology derivations.

No model/provider calls occur here. Plans, projections and reports are rebuilt from an
immutable unique recipe plus the current logical upstream generations. Raw engineering
packages are intentionally outside this executor because replacing/recompiling engineering
inputs requires an explicit ingestion/review action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tools.artifact_replacements import ArtifactReplacementStore
from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef, RecomputeTask
from tools.hydrology_derivation_store import plan_recipe, projection_recipe, report_recipe
from tools.hydrology_projection import build_hydrology_projection
from tools.hydrology_report import build_hydrology_report
from tools.hydrology_retrieval import plan_hydrology_retrieval
from tools.hydrology_store import HydrologyArtifactStore, decode_artifact, make_envelope, query_spec_from_payload
from tools.spatiotemporal_index import SpatiotemporalIndex

_KIND_MAP = {
    "hydrology_plan": "retrieval_plan",
    "hydrology_projection": "evidence_projection",
    "hydrology_report": "evidence_report",
}
_PRIORITY = ("hydrology_plan", "hydrology_projection", "hydrology_report")


@dataclass(frozen=True)
class HydrologyRecomputeOutcome:
    task_id: str
    artifact_kind: str
    old_fingerprint: str
    new_fingerprint: str
    project_id: str
    logical_id: str
    status: str
    error_type: str = ""


class HydrologyRecomputeExecutor:
    def __init__(
        self,
        *,
        invalidations: DependencyInvalidationStore,
        replacements: ArtifactReplacementStore,
        store: HydrologyArtifactStore,
        recipes: Any,
    ) -> None:
        self.invalidations = invalidations
        self.replacements = replacements
        self.store = store
        self.recipes = recipes

    def _recipe(self, owner_id: str, task: RecomputeTask):
        artifact_kind = _KIND_MAP[task.artifact.kind]
        return self.recipes.for_artifact(owner_id, artifact_kind, task.artifact.resource_id)

    def _current(self, owner_id: str, recipe: Any):
        return self.store.get(owner_id, recipe.project_id, recipe.artifact_kind, recipe.logical_id)

    def _record_replacement(self, owner_id: str, task: RecomputeTask, new_fingerprint: str) -> None:
        if new_fingerprint == task.artifact.resource_id:
            return
        self.replacements.put(
            owner_id,
            old=DependencyRef(task.artifact.kind, task.artifact.resource_id),
            new=DependencyRef(task.artifact.kind, new_fingerprint),
            reason=task.reason,
            triggering_event_sha256=task.triggering_event_sha256,
        )

    def _recompute_plan(self, owner_id: str, task: RecomputeTask, recipe: Any) -> str:
        current = self._current(owner_id, recipe)
        if current.fingerprint != task.artifact.resource_id:
            self._record_replacement(owner_id, task, current.fingerprint)
            return current.fingerprint
        inputs, parameters = recipe.inputs, recipe.parameters
        topology_id = str(inputs["topology_id"])
        package_id = str(inputs["package_id"])
        topology_envelope = self.store.get(owner_id, recipe.project_id, "topology", topology_id)
        package_envelope = self.store.get(owner_id, recipe.project_id, "engineering_package", package_id)
        network = decode_artifact("topology", topology_envelope.payload)
        package = decode_artifact("engineering_package", package_envelope.payload)
        if package.topology_fingerprint != network.fingerprint:
            raise RuntimeError("current package/topology generations are incompatible; recompile the engineering package first")
        spec_payload = parameters.get("spec")
        travel = parameters.get("reach_travel_seconds", {})
        limit = int(parameters.get("limit", 1000))
        if not isinstance(spec_payload, Mapping) or not isinstance(travel, Mapping):
            raise RuntimeError("plan derivation recipe is malformed")
        spec = query_spec_from_payload(spec_payload)
        index = SpatiotemporalIndex()
        package.populate_index(index)
        plan = plan_hydrology_retrieval(
            network,
            index,
            spec,
            reach_travel_seconds={str(key): float(value) for key, value in travel.items()},
            limit=limit,
            package=package,
            expected_index_fingerprint=index.fingerprint,
        )
        successor_recipe = plan_recipe(
            owner_id,
            recipe.project_id,
            logical_id=recipe.logical_id,
            artifact_fingerprint=plan.fingerprint,
            topology_id=topology_id,
            topology_fingerprint=network.fingerprint,
            package_id=package_id,
            package_fingerprint=package.fingerprint,
            spec=spec_payload,
            reach_travel_seconds={str(key): float(value) for key, value in travel.items()},
            limit=limit,
        )
        self.recipes.put(successor_recipe)
        stored = self.store.put(
            make_envelope(owner_id, recipe.project_id, "retrieval_plan", recipe.logical_id, plan),
            expected_current_fingerprint=current.fingerprint,
        )
        self._record_replacement(owner_id, task, stored.fingerprint)
        return stored.fingerprint

    def _recompute_projection(self, owner_id: str, task: RecomputeTask, recipe: Any) -> str:
        current = self._current(owner_id, recipe)
        if current.fingerprint != task.artifact.resource_id:
            self._record_replacement(owner_id, task, current.fingerprint)
            return current.fingerprint
        package_id = str(recipe.inputs["package_id"])
        plan_id = str(recipe.inputs["plan_id"])
        package_envelope = self.store.get(owner_id, recipe.project_id, "engineering_package", package_id)
        plan_envelope = self.store.get(owner_id, recipe.project_id, "retrieval_plan", plan_id)
        package = decode_artifact("engineering_package", package_envelope.payload)
        plan = decode_artifact("retrieval_plan", plan_envelope.payload)
        projection = build_hydrology_projection(package, plan, projection_id=recipe.logical_id)
        successor_recipe = projection_recipe(
            owner_id,
            recipe.project_id,
            logical_id=recipe.logical_id,
            artifact_fingerprint=projection.fingerprint,
            package_id=package_id,
            package_fingerprint=package.fingerprint,
            plan_id=plan_id,
            plan_fingerprint=plan.fingerprint,
        )
        self.recipes.put(successor_recipe)
        stored = self.store.put(
            make_envelope(owner_id, recipe.project_id, "evidence_projection", recipe.logical_id, projection),
            expected_current_fingerprint=current.fingerprint,
        )
        self._record_replacement(owner_id, task, stored.fingerprint)
        return stored.fingerprint

    def _recompute_report(self, owner_id: str, task: RecomputeTask, recipe: Any) -> str:
        current = self._current(owner_id, recipe)
        if current.fingerprint != task.artifact.resource_id:
            self._record_replacement(owner_id, task, current.fingerprint)
            return current.fingerprint
        projection_id = str(recipe.inputs["projection_id"])
        projection_envelope = self.store.get(owner_id, recipe.project_id, "evidence_projection", projection_id)
        projection = decode_artifact("evidence_projection", projection_envelope.payload)
        title = str(recipe.parameters["title"])
        research_question = str(recipe.parameters.get("research_question", ""))
        report = build_hydrology_report(
            projection,
            report_id=recipe.logical_id,
            project_id=recipe.project_id,
            title=title,
            research_question=research_question,
        )
        successor_recipe = report_recipe(
            owner_id,
            recipe.project_id,
            logical_id=recipe.logical_id,
            artifact_fingerprint=report.fingerprint,
            projection_id=projection_id,
            projection_fingerprint=projection.fingerprint,
            title=title,
            research_question=research_question,
        )
        self.recipes.put(successor_recipe)
        stored = self.store.put(
            make_envelope(owner_id, recipe.project_id, "evidence_report", recipe.logical_id, report),
            expected_current_fingerprint=current.fingerprint,
        )
        self._record_replacement(owner_id, task, stored.fingerprint)
        return stored.fingerprint

    def execute_claimed(self, owner_id: str, task: RecomputeTask) -> HydrologyRecomputeOutcome:
        if not isinstance(task, RecomputeTask) or task.status != "claimed":
            raise ValueError("task must be a claimed RecomputeTask")
        if task.artifact.kind not in _KIND_MAP:
            raise ValueError("task is not a deterministic hydrology recompute task")
        project_id = ""
        logical_id = ""
        try:
            recipe = self._recipe(owner_id, task)
            project_id, logical_id = recipe.project_id, recipe.logical_id
            if recipe.artifact_fingerprint != task.artifact.resource_id:
                raise RuntimeError("hydrology recompute recipe does not match the stale artifact fingerprint")
            if task.artifact.kind == "hydrology_plan":
                new_fingerprint = self._recompute_plan(owner_id, task, recipe)
            elif task.artifact.kind == "hydrology_projection":
                new_fingerprint = self._recompute_projection(owner_id, task, recipe)
            else:
                new_fingerprint = self._recompute_report(owner_id, task, recipe)
            self.invalidations.finish_recompute(owner_id, task.task_id, success=True)
            return HydrologyRecomputeOutcome(
                task.task_id,
                task.artifact.kind,
                task.artifact.resource_id,
                new_fingerprint,
                project_id,
                logical_id,
                "completed",
            )
        except Exception as exc:
            error_type = type(exc).__name__[:200]
            self.invalidations.finish_recompute(
                owner_id,
                task.task_id,
                success=False,
                error_type=error_type,
                acknowledge_stale=False,
            )
            return HydrologyRecomputeOutcome(
                task.task_id,
                task.artifact.kind,
                task.artifact.resource_id,
                "",
                project_id,
                logical_id,
                "failed",
                error_type,
            )

    def process_one(self, owner_id: str, *, max_attempts: int = 5) -> HydrologyRecomputeOutcome | None:
        for kind in _PRIORITY:
            task = self.invalidations.claim_recompute(owner_id, kinds=(kind,), max_attempts=max_attempts)
            if task is not None:
                return self.execute_claimed(owner_id, task)
        return None

    def drain(self, owner_id: str, *, limit: int = 100, max_attempts: int = 5) -> tuple[HydrologyRecomputeOutcome, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        output: list[HydrologyRecomputeOutcome] = []
        for _ in range(limit):
            outcome = self.process_one(owner_id, max_attempts=max_attempts)
            if outcome is None:
                break
            output.append(outcome)
        return tuple(output)


__all__ = ["HydrologyRecomputeExecutor", "HydrologyRecomputeOutcome"]
