"""Process-local runtime for the custody signer public-key registry."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_custody_signer_registry import (
    CustodySignerKeyRegistry,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_custody_signers.sqlite3"
_LOCK = threading.RLock()
_REGISTRIES: dict[str, CustodySignerKeyRegistry] = {}


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


def get_custody_signer_key_registry(
    path: str | os.PathLike[str] | None = None,
) -> CustodySignerKeyRegistry:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
        _DEFAULT_PATH,
    )
    registry_path = _canonical(selected)
    protected: list[Path] = []
    for variable in (
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
    if any(_same_file(registry_path, value) for value in protected):
        raise RuntimeError("custody signer registry must not alias another journal.")
    key = str(registry_path)
    with _LOCK:
        registry = _REGISTRIES.get(key)
        if registry is None:
            registry = CustodySignerKeyRegistry(registry_path)
            _REGISTRIES[key] = registry
        return registry


def clear_custody_signer_key_registry_cache() -> None:
    with _LOCK:
        _REGISTRIES.clear()


__all__ = [
    "clear_custody_signer_key_registry_cache",
    "get_custody_signer_key_registry",
]
