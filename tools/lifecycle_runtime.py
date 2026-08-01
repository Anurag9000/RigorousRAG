"""Process-local lifecycle-outbox factories and bounded reconciliation."""

from __future__ import annotations

import math
import operator
import os
import threading
from pathlib import Path
from typing import Any

from tools.lifecycle_outbox import LifecycleOutbox, LifecycleReconcileResult
from tools.lifecycle_reconciliation import (
    get_cleanup_journal,
    reconcile_claimed_operations,
)

_LOCK = threading.RLock()
_OUTBOXES: dict[str, LifecycleOutbox] = {}
_STARTUP_COMPLETE: set[str] = set()


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _positive(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= maximum:
        raise ValueError(f"{label} must be greater than zero and at most {maximum}.")
    return parsed


def _selected_path(value: str | os.PathLike[str] | None = None) -> str:
    selected = (
        value
        if value is not None
        else os.getenv("LIFECYCLE_OUTBOX_DB_PATH", "data/lifecycle_outbox.sqlite3")
    )
    return str(Path(selected).expanduser().absolute())


def get_lifecycle_outbox(
    path: str | os.PathLike[str] | None = None,
) -> LifecycleOutbox:
    selected = _selected_path(path)
    with _LOCK:
        outbox = _OUTBOXES.get(selected)
        if outbox is None:
            outbox = LifecycleOutbox(selected)
            _OUTBOXES[selected] = outbox
        elif not outbox.ping():
            raise RuntimeError("lifecycle outbox is unavailable.")
        return outbox


def remove_source_idempotently(registry: Any, source_path: str) -> bool:
    """Remove a validated retained file or accept an already-absent file."""

    if not isinstance(source_path, str) or not source_path or len(source_path) > 4096:
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in source_path):
        return False
    try:
        root = Path(registry.upload_root).absolute()
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        candidate = candidate.absolute()
        candidate.relative_to(root)
    except Exception:
        return False
    try:
        if not candidate.exists():
            return True
    except OSError:
        return False
    try:
        return bool(registry.remove_source(str(candidate)))
    except Exception:
        return False


def reconcile_lifecycle_pending(
    *,
    path: str | os.PathLike[str] | None = None,
    limit: int | None = None,
    lease_seconds: float | None = None,
    worker_id: str | None = None,
) -> tuple[LifecycleReconcileResult, ...]:
    from tools.document_store import get_document_store
    from tools.index_coordinator import _document_lock
    from tools.sparse_runtime import get_generation_store

    maximum = _integer(
        limit
        if limit is not None
        else int(os.getenv("LIFECYCLE_RECONCILE_LIMIT", "100")),
        "limit",
        1,
        10_000,
    )
    lease = _positive(
        lease_seconds
        if lease_seconds is not None
        else float(os.getenv("LIFECYCLE_LEASE_SECONDS", "60")),
        "lease_seconds",
        86_400.0,
    )
    worker = worker_id or f"lifecycle-{os.getpid()}-{threading.get_ident()}"
    outbox = get_lifecycle_outbox(path)
    claimed = outbox.claim(
        worker_id=worker,
        limit=maximum,
        lease_seconds=lease,
    )
    if not claimed:
        return ()
    generations = get_generation_store()
    registry = get_document_store()
    cleanup = get_cleanup_journal()
    results: list[LifecycleReconcileResult] = []
    for operation in claimed:
        with _document_lock(operation.owner_id, operation.doc_id):
            results.extend(
                reconcile_claimed_operations(
                    (operation,),
                    outbox=outbox,
                    generations=generations,
                    registry=registry,
                    worker_id=worker,
                    cleanup=cleanup,
                    remove_source=lambda value: remove_source_idempotently(
                        registry, value
                    ),
                )
            )
    return tuple(results)


def reconcile_lifecycle_before_retrieval(
    path: str | os.PathLike[str] | None = None,
) -> tuple[LifecycleReconcileResult, ...]:
    selected = _selected_path(path)
    with _LOCK:
        if selected in _STARTUP_COMPLETE:
            return ()
    results = reconcile_lifecycle_pending(path=selected)
    if any(result.outcome in {"error", "failed"} for result in results):
        raise RuntimeError("lifecycle reconciliation failed before retrieval.")
    with _LOCK:
        _STARTUP_COMPLETE.add(selected)
    return results


def clear_lifecycle_runtime_caches() -> None:
    with _LOCK:
        _OUTBOXES.clear()
        _STARTUP_COMPLETE.clear()


__all__ = [
    "clear_lifecycle_runtime_caches",
    "get_lifecycle_outbox",
    "remove_source_idempotently",
    "reconcile_lifecycle_before_retrieval",
    "reconcile_lifecycle_pending",
]
