"""Process-local runtime for restore-intent deletion authorizations."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_deletion_boundary import (
    GovernedSignedRetirementRestoreDeletionAuthorizationStore,
)

_DEFAULT_PATH = (
    "data/evidence_graph_set_signed_retirement_deletion_authorizations.sqlite3"
)
_LOCK = threading.RLock()
_STORES: dict[str, GovernedSignedRetirementRestoreDeletionAuthorizationStore] = {}


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


def get_signed_retirement_restore_deletion_authorization_store(
    path: str | os.PathLike[str] | None = None,
) -> GovernedSignedRetirementRestoreDeletionAuthorizationStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_AUTH_DB_PATH",
        _DEFAULT_PATH,
    )
    authorization_path = _canonical(selected)
    protected: list[Path] = []
    for variable in (
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
    ):
        value = os.getenv(variable)
        if value:
            protected.append(_canonical(value))
    if any(_same_file(authorization_path, value) for value in protected):
        raise RuntimeError(
            "deletion authorization store must not alias restore, hold, custody, "
            "retirement, or publication databases."
        )
    key = str(authorization_path)
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = GovernedSignedRetirementRestoreDeletionAuthorizationStore(
                authorization_path
            )
            _STORES[key] = store
        return store


def clear_signed_retirement_restore_deletion_authorization_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_signed_retirement_restore_deletion_authorization_store_cache",
    "get_signed_retirement_restore_deletion_authorization_store",
]
