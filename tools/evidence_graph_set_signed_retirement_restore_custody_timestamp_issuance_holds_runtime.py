"""Process-local runtime for custody timestamp issuance legal holds."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds import (
    CustodyTimestampIssuanceHoldStore,
)

_DEFAULT_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_issuance_holds.sqlite3"
)
_LOCK = threading.RLock()
_STORES: dict[str, CustodyTimestampIssuanceHoldStore] = {}


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


def get_custody_timestamp_issuance_hold_store(
    path: str | os.PathLike[str] | None = None,
) -> CustodyTimestampIssuanceHoldStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_HOLD_DB_PATH",
        _DEFAULT_PATH,
    )
    store_path = _canonical(selected)
    protected: list[Path] = []
    for variable in (
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH",
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH",
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_ADMIN_USE_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
    ):
        value = os.getenv(variable)
        if value:
            protected.append(_canonical(value))
    if any(_same_file(store_path, value) for value in protected):
        raise RuntimeError("timestamp issuance hold store must not alias another journal.")
    key = str(store_path)
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = CustodyTimestampIssuanceHoldStore(store_path)
            _STORES[key] = store
        return store


def clear_custody_timestamp_issuance_hold_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_custody_timestamp_issuance_hold_store_cache",
    "get_custody_timestamp_issuance_hold_store",
]
