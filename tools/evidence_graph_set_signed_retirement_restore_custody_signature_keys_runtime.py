"""Runtime factory for governed external custody signer public keys."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_custody_signature_key_contracts import (
    validated_path,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys import (
    CustodySignerKeyRegistry,
)

_DEFAULT = "data/evidence_graph_restore_custody_signer_keys.sqlite3"
_LOCK = threading.RLock()
_CACHE: dict[Path, CustodySignerKeyRegistry] = {}


def get_custody_signer_key_registry(
    path: str | os.PathLike[str] | None = None,
) -> CustodySignerKeyRegistry:
    selected = path or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_KEY_DB_PATH",
        _DEFAULT,
    )
    canonical = validated_path(selected, label="signer_key_registry_path")
    with _LOCK:
        value = _CACHE.get(canonical)
        if value is None:
            value = CustodySignerKeyRegistry(canonical)
            _CACHE[canonical] = value
        return value


def clear_custody_signer_key_registry_cache() -> None:
    with _LOCK:
        _CACHE.clear()


__all__ = [
    "clear_custody_signer_key_registry_cache",
    "get_custody_signer_key_registry",
]
