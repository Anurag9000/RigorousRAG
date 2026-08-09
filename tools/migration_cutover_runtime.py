"""Runtime factories and preparation orchestration for non-executing cutover plans."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

from tools.migration_cutover_control import CutoverOperation, CutoverPreparation, build_cutover_preparation
from tools.migration_cutover_journal import MigrationCutoverJournal
from tools.migration_rollback_artifact import rollback_key_from_environment
from tools.migration_rollback_reconstruction import reconstruct_rollback_snapshots
from tools.migration_rollback_staging import verify_in_isolated_staging
from tools.migration_types import identifier

_LOCK = threading.RLock()
_JOURNALS: dict[str, MigrationCutoverJournal] = {}


def get_migration_cutover_journal(
    path: str | os.PathLike[str] | None = None,
) -> MigrationCutoverJournal:
    selected = path if path is not None else os.getenv(
        "MIGRATION_CUTOVER_DB_PATH",
        "data/migration_cutovers.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = MigrationCutoverJournal(key)
            _JOURNALS[key] = journal
        return journal


def clear_migration_cutover_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


def resolve_cutover_preparation(task_id: str) -> CutoverPreparation:
    selected = identifier(task_id, "task_id", 64)
    from tools.migration_cutover_preflight_runtime import (
        get_migration_cutover_preflight_store,
    )
    from tools.migration_promotion_runtime import get_migration_promotion_store
    from tools.migration_rollback_runtime import get_migration_rollback_store
    from tools.migration_runtime import get_migration_journal
    from tools.sparse_runtime import get_generation_store

    task = get_migration_journal().get(selected)
    if task is None:
        raise FileNotFoundError(selected)
    preflight = get_migration_cutover_preflight_store().read(selected)
    promotion = get_migration_promotion_store().read(selected)
    payload, rollback_manifest = get_migration_rollback_store().load(
        preflight=preflight,
        key=rollback_key_from_environment(),
    )
    reconstructed = reconstruct_rollback_snapshots(preflight, payload)
    staging = verify_in_isolated_staging(preflight, reconstructed)
    generation = get_generation_store().current(
        owner_id=task.owner_id,
        doc_id=task.doc_id,
    )
    return build_cutover_preparation(
        task=task,
        preflight=preflight,
        promotion=promotion,
        rollback_manifest=rollback_manifest,
        staging=staging,
        generation=generation,
    )


def prepare_cutover_operation(
    task_id: str,
    *,
    worker_id: str,
    lease_seconds: int = 300,
    max_attempts: int = 3,
    journal: MigrationCutoverJournal | None = None,
    resolver: Callable[[str], CutoverPreparation] | None = None,
) -> CutoverOperation:
    selected = identifier(task_id, "task_id", 64)
    worker = identifier(worker_id, "worker_id", 128)
    selected_journal = journal or get_migration_cutover_journal()
    selected_resolver = resolver or resolve_cutover_preparation
    first = selected_resolver(selected)
    if not isinstance(first, CutoverPreparation) or first.task_id != selected:
        raise RuntimeError("cutover prerequisite resolver returned invalid preparation.")
    operation = selected_journal.seed(first)
    if operation.state == "ready":
        return operation
    claimed: CutoverOperation | None = None
    try:
        claimed = selected_journal.claim(
            operation.operation_id,
            worker_id=worker,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )
        second = selected_resolver(selected)
        if not isinstance(second, CutoverPreparation):
            raise RuntimeError("cutover prerequisite resolver returned invalid preparation.")
        if second.operation_id != first.operation_id:
            raise RuntimeError("cutover prerequisites changed during leased preparation.")
        return selected_journal.mark_ready(
            operation.operation_id,
            worker_id=worker,
            fencing_token=claimed.fencing_token,
        )
    except Exception as exc:
        if claimed is not None:
            try:
                selected_journal.mark_failed(
                    operation.operation_id,
                    worker_id=worker,
                    fencing_token=claimed.fencing_token,
                    failure_type=type(exc).__name__,
                )
            except Exception:
                pass
        raise


__all__ = [
    "clear_migration_cutover_journal_cache",
    "get_migration_cutover_journal",
    "prepare_cutover_operation",
    "resolve_cutover_preparation",
]