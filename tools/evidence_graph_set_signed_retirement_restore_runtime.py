"""Process-local runtime for signed retirement snapshot restore intents."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_journal import (
    SignedRetirementRestoreJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_mutation import (
    canonical_target_path,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_restores.sqlite3"
_LOCK = threading.RLock()
_JOURNALS: dict[str, SignedRetirementRestoreJournal] = {}


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


def get_signed_retirement_restore_journal(
    path: str | os.PathLike[str] | None = None,
    *,
    target_db_path: str | os.PathLike[str] | None = None,
) -> SignedRetirementRestoreJournal:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        _DEFAULT_PATH,
    )
    restore_path = _canonical(selected)
    protected: list[Path] = []
    if target_db_path is not None:
        protected.append(canonical_target_path(target_db_path))
    for variable in (
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
    ):
        value = os.getenv(variable)
        if value:
            protected.append(_canonical(value))
    if any(_same_file(restore_path, value) for value in protected):
        raise RuntimeError(
            "restore intent journal must not alias retirement or publication databases."
        )
    key = str(restore_path)
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = SignedRetirementRestoreJournal(restore_path)
            _JOURNALS[key] = journal
        return journal


def clear_signed_retirement_restore_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = [
    "clear_signed_retirement_restore_journal_cache",
    "get_signed_retirement_restore_journal",
]
