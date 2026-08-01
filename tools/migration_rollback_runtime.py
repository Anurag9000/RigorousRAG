"""Process-local factory for encrypted migration rollback artifacts."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.migration_rollback_store import MigrationRollbackStore

_LOCK = threading.RLock()
_STORES: dict[str, MigrationRollbackStore] = {}


def get_migration_rollback_store(
    root: str | os.PathLike[str] | None = None,
) -> MigrationRollbackStore:
    selected = root if root is not None else os.getenv(
        "MIGRATION_ROLLBACK_ROOT",
        "data/migration_rollbacks",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = MigrationRollbackStore(key)
            _STORES[key] = store
        return store


def clear_migration_rollback_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_migration_rollback_store_cache",
    "get_migration_rollback_store",
]
