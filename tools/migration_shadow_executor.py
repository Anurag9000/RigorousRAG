"""Lease-aware migration shadow build and validation without live cutover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from tools.migration_shadow_store import (
    MigrationShadowStore,
    ShadowArtifactManifest,
    ShadowBuild,
)
from tools.migration_types import MigrationTask, identifier


class _MigrationJournal(Protocol):
    def mark_validated(
        self,
        *,
        task_id: str,
        worker_id: str,
        validation_digest: str,
        now: float | None = None,
    ) -> MigrationTask: ...

    def mark_failed(
        self,
        *,
        task_id: str,
        worker_id: str,
        failure_type: str,
        now: float | None = None,
    ) -> MigrationTask: ...


class _GenerationStore(Protocol):
    def current(self, *, owner_id: str, doc_id: str) -> Any: ...


@dataclass(frozen=True)
class ShadowExecutionResult:
    task_id: str
    outcome: str
    task_state: str
    validation_digest: str | None = None
    vector_count: int = 0
    sparse_count: int = 0
    failure_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", identifier(self.task_id, "task_id", 64))
        if self.outcome not in {"validated", "already_validated", "failed"}:
            raise ValueError("shadow execution outcome is invalid.")
        if self.task_state not in {"validated", "failed"}:
            raise ValueError("shadow execution task state is invalid.")
        if self.validation_digest is not None and len(self.validation_digest) != 64:
            raise ValueError("validation_digest is invalid.")
        if self.vector_count < 0 or self.sparse_count < 0:
            raise ValueError("shadow counts must be non-negative.")
        if self.failure_type is not None:
            object.__setattr__(
                self,
                "failure_type",
                identifier(self.failure_type, "failure_type", 200),
            )


def _generation_matches(task: MigrationTask, generation: Any) -> bool:
    return bool(
        generation is not None
        and getattr(generation, "state", None) in {"active", "restored"}
        and getattr(generation, "sequence", None) == task.source_sequence
        and getattr(generation, "profile_fingerprint", None)
        == task.source_profile_fingerprint
    )


def _manifest_matches_task(
    task: MigrationTask,
    manifest: ShadowArtifactManifest,
) -> bool:
    return bool(
        manifest.task_id == task.task_id
        and manifest.owner_id == task.owner_id
        and manifest.doc_id == task.doc_id
        and manifest.source_sequence == task.source_sequence
        and manifest.source_profile_fingerprint
        == task.source_profile_fingerprint
        and manifest.target_profile_name == task.target_profile_name
        and manifest.target_profile_fingerprint
        == task.target_profile_fingerprint
    )


def build_and_validate_shadow(
    task: MigrationTask,
    *,
    worker_id: str,
    journal: _MigrationJournal,
    generations: _GenerationStore,
    shadows: MigrationShadowStore,
    builder: Callable[[MigrationTask], ShadowBuild],
    now: float | None = None,
) -> ShadowExecutionResult:
    """Build one running task in isolation and record only a validation digest."""

    if not isinstance(task, MigrationTask):
        raise ValueError("task must be a MigrationTask.")
    worker = identifier(worker_id, "worker_id", 128)
    if task.state not in {"running", "validated"}:
        raise ValueError("shadow execution requires a running or validated task.")
    if task.lease_owner != worker:
        raise ValueError("shadow execution worker does not own the task lease.")

    if task.state == "validated":
        manifest = shadows.validate(task.task_id)
        if not _manifest_matches_task(task, manifest):
            raise RuntimeError("validated shadow artifact does not match its task.")
        if task.validation_digest != manifest.validation_digest:
            raise RuntimeError("journal validation digest does not match shadow artifacts.")
        return ShadowExecutionResult(
            task_id=task.task_id,
            outcome="already_validated",
            task_state="validated",
            validation_digest=manifest.validation_digest,
            vector_count=manifest.vector_count,
            sparse_count=manifest.sparse_count,
        )

    before = generations.current(owner_id=task.owner_id, doc_id=task.doc_id)
    if not _generation_matches(task, before):
        raise RuntimeError("source generation changed before shadow execution.")
    if not callable(builder):
        raise ValueError("builder must be callable.")
    build = builder(task)
    if not isinstance(build, ShadowBuild):
        raise ValueError("builder must return a ShadowBuild.")
    manifest = shadows.write(task=task, build=build, now=now)
    if not _manifest_matches_task(task, manifest):
        raise RuntimeError("shadow artifact does not match its migration task.")
    if manifest.content_sha256 != getattr(before, "content_sha256", None):
        raise RuntimeError("shadow content hash does not match the source generation.")

    after = generations.current(owner_id=task.owner_id, doc_id=task.doc_id)
    if not _generation_matches(task, after):
        raise RuntimeError("source generation changed during shadow execution.")
    if getattr(after, "content_sha256", None) != manifest.content_sha256:
        raise RuntimeError("source content hash changed during shadow execution.")

    validated = journal.mark_validated(
        task_id=task.task_id,
        worker_id=worker,
        validation_digest=manifest.validation_digest,
        now=now,
    )
    return ShadowExecutionResult(
        task_id=task.task_id,
        outcome="validated",
        task_state=validated.state,
        validation_digest=manifest.validation_digest,
        vector_count=manifest.vector_count,
        sparse_count=manifest.sparse_count,
    )


def execute_shadow_task(
    task: MigrationTask,
    *,
    worker_id: str,
    journal: _MigrationJournal,
    generations: _GenerationStore,
    shadows: MigrationShadowStore,
    builder: Callable[[MigrationTask], ShadowBuild],
    now: float | None = None,
) -> ShadowExecutionResult:
    """Contain one shadow-build failure and persist only its generic type."""

    try:
        return build_and_validate_shadow(
            task,
            worker_id=worker_id,
            journal=journal,
            generations=generations,
            shadows=shadows,
            builder=builder,
            now=now,
        )
    except Exception as exc:
        failed = journal.mark_failed(
            task_id=task.task_id,
            worker_id=identifier(worker_id, "worker_id", 128),
            failure_type=type(exc).__name__,
            now=now,
        )
        return ShadowExecutionResult(
            task_id=task.task_id,
            outcome="failed",
            task_state=failed.state,
            failure_type=type(exc).__name__,
        )


__all__ = [
    "ShadowExecutionResult",
    "build_and_validate_shadow",
    "execute_shadow_task",
]
