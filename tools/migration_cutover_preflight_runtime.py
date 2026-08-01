"""Process-local factory for migration cutover-preflight storage."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.migration_cutover_preflight_store import MigrationCutoverPreflightStore

_LOCK = threading.RLock()
_STORES: dict[str, MigrationCutoverPreflightStore] = {}


def get_migration_cutover_preflight_store(
    root: str | os.PathLike[str] | None = None,
) -> MigrationCutoverPreflightStore:
    selected = root if root is not None else os.getenv(
        "MIGRATION_CUTOVER_PREFLIGHT_ROOT",
        "data/migration_cutover_preflights",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = MigrationCutoverPreflightStore(key)
            _STORES[key] = store
        return store


def clear_migration_cutover_preflight_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_migration_cutover_preflight_store_cache",
    "get_migration_cutover_preflight_store",
]
