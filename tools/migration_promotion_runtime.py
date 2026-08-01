"""Process-local factory for append-only migration promotion reports."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.migration_promotion_store import MigrationPromotionStore

_LOCK = threading.RLock()
_STORES: dict[str, MigrationPromotionStore] = {}


def get_migration_promotion_store(
    root: str | os.PathLike[str] | None = None,
) -> MigrationPromotionStore:
    selected = root if root is not None else os.getenv(
        "MIGRATION_PROMOTION_ROOT",
        "data/migration_promotions",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = MigrationPromotionStore(key)
            _STORES[key] = store
        return store


def clear_migration_promotion_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_migration_promotion_store_cache",
    "get_migration_promotion_store",
]
