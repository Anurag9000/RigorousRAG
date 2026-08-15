"""Operator-invoked recomputation for stale immutable research artifacts.

No background thread is started here. Operators or deployment workers explicitly call
``drain``/``process_one``. Result replay requires an encrypted replay recipe; hash-only
results fail closed rather than attempting to reconstruct private queries.
"""

from __future__ import annotations

import inspect
import sqlite3
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from tools.artifact_replacements import ArtifactReplacementStore
from tools.dependency_invalidation import DependencyInvalidationStore, DependencyRef, RecomputeTask
from tools.models import AgentAnswer
from tools.replay_recipe_store import EncryptedReplayRecipeStore
from tools.research_dependencies import register_report_dependencies, register_result_dependencies, stale_reasons
from tools.research_report import ReportSection, ResearchReport
from tools.research_report_store import ResearchReportStore
from tools.research_result_store import ResearchResultStore
from tools.research_workspace import ResearchProject
from tools.runtime_composition import RuntimeComposition
from tools.security import normalize_owner_id


class WorkspaceStore(Protocol):
    def get_project(self, owner_id: str, project_id: str) -> ResearchProject: ...


class RecomputeBlocked(RuntimeError):
    """Raised when an artifact cannot be recomputed without missing governed inputs."""


@dataclass(frozen=True)
class RecomputeOutcome:
    task: RecomputeTask
    success: bool
    replacement: DependencyRef | None = None
    error_type: str = ""


CustomHandler = Callable[[str, RecomputeTask], DependencyRef | None]


class ResearchRecomputeExecutor:
    def __init__(
        self,
        *,
        invalidations: DependencyInvalidationStore,
        replacements: ArtifactReplacementStore,
        results: ResearchResultStore,
        reports: ResearchReportStore,
        workspace: WorkspaceStore,
        composition: RuntimeComposition,
        agent_factory: Callable[[str, str | None], Any],
        replay_recipes: EncryptedReplayRecipeStore | None = None,
        custom_handlers: Mapping[str, CustomHandler] | None = None,
    ) -> None:
        self.invalidations = invalidations
        self.replacements = replacements
        self.results = results
        self.reports = reports
        self.workspace = workspace
        self.composition = composition
        self.agent_factory = agent_factory
        self.replay_recipes = replay_recipes
        self.custom_handlers = dict(custom_handlers or {})

    def _recompute_result(self, owner_id: str, task: RecomputeTask) -> DependencyRef:
        if self.replay_recipes is None:
            raise RecomputeBlocked("encrypted replay recipes are not configured")
        old = self.results.get(owner_id, task.artifact.resource_id)
        recipe = self.replay_recipes.get(owner_id, old.result_id)
        if recipe.query_sha256 != old.query_sha256:
            raise RuntimeError("replay recipe query identity does not match stored result")
        agent = self.agent_factory(owner_id, recipe.model or None)
        answer = agent.run(recipe.query)
        if inspect.isawaitable(answer):
            raise RuntimeError("synchronous recompute executor cannot await the configured agent")
        if not isinstance(answer, AgentAnswer):
            raise RuntimeError("recomputed agent result is invalid")
        model = str(getattr(agent, "model", recipe.model))
        new = self.results.put(
            owner_id,
            query_sha256=recipe.query_sha256,
            answer=answer,
            strategy=recipe.strategy,
            model=model,
        )
        register_result_dependencies(
            self.invalidations,
            owner_id,
            new,
            composition=self.composition,
        )
        self.replay_recipes.put(
            owner_id,
            result_id=new.result_id,
            query_sha256=recipe.query_sha256,
            query=recipe.query,
            model=model,
            strategy=recipe.strategy,
        )
        return DependencyRef("result", new.result_id)

    def _current_result_for_report(self, owner_id: str, result_id: str) -> str:
        ref = self.replacements.current(owner_id, DependencyRef("result", result_id))
        stale = stale_reasons(self.invalidations, owner_id, ref, maximum=20)
        if stale:
            raise RecomputeBlocked("report dependency result is still stale")
        return ref.resource_id

    def _recompute_report(self, owner_id: str, task: RecomputeTask) -> DependencyRef:
        old = self.reports.get(owner_id, task.artifact.resource_id)
        result_id = self._current_result_for_report(owner_id, old.result_id)
        result = self.results.get(owner_id, result_id)
        try:
            project = self.workspace.get_project(owner_id, old.project_id)
            title = project.title
            question = project.research_question
        except Exception:
            title = old.report.title
            question = old.report.question
        if len(result.citations) > 100:
            raise RecomputeBlocked("current result exceeds report citation limit")
        rebuilt = ResearchReport(
            title=title,
            question=question,
            search_strategy=result.strategy,
            sections=(
                ReportSection(
                    heading="Synthesis",
                    body=result.answer,
                    citation_ids=result.citation_ids,
                ),
            ),
            evidence_matrix=old.report.evidence_matrix,
            citations=result.citations,
            conflicts=old.report.conflicts,
            limitations=old.report.limitations,
            warnings=result.warnings,
        )
        new = self.reports.put(
            owner_id,
            result_id=result.result_id,
            project_id=old.project_id,
            report=rebuilt,
        )
        register_report_dependencies(self.invalidations, owner_id, new)
        return DependencyRef("report", new.report_id)

    def _handler(self, task: RecomputeTask) -> CustomHandler:
        if task.artifact.kind == "result":
            return self._recompute_result
        if task.artifact.kind == "report":
            return self._recompute_report
        handler = self.custom_handlers.get(task.artifact.kind)
        if handler is None:
            raise RecomputeBlocked(f"no recompute handler is registered for {task.artifact.kind}")
        return handler

    def _complete(
        self,
        owner_id: str,
        task: RecomputeTask,
        replacement: DependencyRef | None,
    ) -> RecomputeOutcome:
        old = task.artifact
        if replacement is not None:
            if not isinstance(replacement, DependencyRef):
                raise TypeError("recompute handler must return DependencyRef or null")
            if replacement.kind != old.kind:
                raise RuntimeError("recompute handler changed artifact kind")
            if replacement != old:
                self.replacements.put(
                    owner_id,
                    old=old,
                    new=replacement,
                    reason=task.reason,
                    triggering_event_sha256=task.triggering_event_sha256,
                )
        self.invalidations.finish_recompute(owner_id, task.task_id, success=True)
        return RecomputeOutcome(task, True, replacement)

    def process_one(self, owner_id: str, *, kinds: tuple[str, ...] = ()) -> RecomputeOutcome | None:
        owner = normalize_owner_id(owner_id)
        task = self.invalidations.claim_recompute(owner, kinds=kinds)
        if task is None:
            return None
        try:
            replacement = self._handler(task)(owner, task)
            return self._complete(owner, task, replacement)
        except Exception as exc:
            error_type = type(exc).__name__[:200]
            self.invalidations.finish_recompute(
                owner,
                task.task_id,
                success=False,
                error_type=error_type,
                acknowledge_stale=False,
            )
            return RecomputeOutcome(task, False, None, error_type)

    def drain(self, owner_id: str, *, max_tasks: int = 100) -> tuple[RecomputeOutcome, ...]:
        if isinstance(max_tasks, bool) or not isinstance(max_tasks, int) or not 1 <= max_tasks <= 10_000:
            raise ValueError("max_tasks is invalid")
        owner = normalize_owner_id(owner_id)
        output: list[RecomputeOutcome] = []
        # Upstream result replacements must exist before downstream reports are rebuilt.
        phases = (("result",), ("report",), tuple(sorted(self.custom_handlers)))
        for kinds in phases:
            if not kinds:
                continue
            while len(output) < max_tasks:
                outcome = self.process_one(owner, kinds=kinds)
                if outcome is None:
                    break
                output.append(outcome)
            if len(output) >= max_tasks:
                break
        return tuple(output)


def requeue_failed_task(
    store: DependencyInvalidationStore,
    owner_id: str,
    task_id: str,
) -> bool:
    """Explicit operator retry for a failed recomputation task."""

    owner = normalize_owner_id(owner_id)
    if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 256:
        raise ValueError("task_id is invalid")
    connection = sqlite3.connect(str(store.path), timeout=30.0, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=30000")
        cursor = connection.execute(
            """UPDATE recompute_tasks
               SET status='queued',claimed_at=NULL,completed_at=NULL,error_type=''
               WHERE owner_id=? AND task_id=? AND status='failed'""",
            (owner, task_id.strip()),
        )
        return bool(cursor.rowcount)
    finally:
        connection.close()


__all__ = [
    "RecomputeBlocked",
    "RecomputeOutcome",
    "ResearchRecomputeExecutor",
    "requeue_failed_task",
]
