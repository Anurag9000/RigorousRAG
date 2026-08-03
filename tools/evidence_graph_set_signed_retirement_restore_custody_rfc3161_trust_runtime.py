"""Runtime factory for the external RFC 3161 TSA trust registry."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust import (
    Rfc3161TrustRegistry,
    _registry_path,
)

_DEFAULT = "data/evidence_graph_restore_custody_rfc3161_trust.sqlite3"
_LOCK = threading.RLock()
_CACHE: dict[Path, Rfc3161TrustRegistry] = {}


def get_rfc3161_trust_registry(
    path: str | os.PathLike[str] | None = None,
) -> Rfc3161TrustRegistry:
    selected = path or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_RFC3161_TRUST_DB_PATH",
        _DEFAULT,
    )
    canonical = _registry_path(selected)
    with _LOCK:
        value = _CACHE.get(canonical)
        if value is None:
            value = Rfc3161TrustRegistry(canonical)
            _CACHE[canonical] = value
        return value


def clear_rfc3161_trust_registry_cache() -> None:
    with _LOCK:
        _CACHE.clear()


__all__ = [
    "clear_rfc3161_trust_registry_cache",
    "get_rfc3161_trust_registry",
]
