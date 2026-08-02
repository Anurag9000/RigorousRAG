"""Process-local runtime for signed-retirement restore custody manifests."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_custody_manifest_boundary import (
    GovernedSignedRetirementRestoreCustodyStore,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_custody.sqlite3"
_LOCK = threading.RLock()
_STORES: dict[str, GovernedSignedRetirementRestoreCustodyStore] = {}


def _canonical(value: str | os.PathLike[str]) -> Path:
    candidate = Path(os.fspath(value))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return os.path.samefile(left, right)
    except (FileNotFoundError, OSError):
        return False


def get_signed_retirement_restore_custody_store(
    path: str | os.PathLike[str] | None = None,
    *,
    target_db_path: str | os.PathLike[str] | None = None,
) -> GovernedSignedRetirementRestoreCustodyStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH",
        _DEFAULT_PATH,
    )
    custody_path = _canonical(selected)
    protected: list[Path] = []
    if target_db_path is not None:
        protected.append(_canonical(target_db_path))
    for variable in (
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
    ):
        value = os.getenv(variable)
        if value:
            protected.append(_canonical(value))
    if any(_same_file(custody_path, value) for value in protected):
        raise RuntimeError(
            "restore custody store must not alias hold, restore, retirement, "
            "target, or publication databases."
        )
    key = str(custody_path)
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = GovernedSignedRetirementRestoreCustodyStore(custody_path)
            _STORES[key] = store
        return store


def clear_signed_retirement_restore_custody_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_signed_retirement_restore_custody_store_cache",
    "get_signed_retirement_restore_custody_store",
]
