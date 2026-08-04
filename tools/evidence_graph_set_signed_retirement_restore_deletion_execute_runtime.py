"""Process-local runtime for restore-intent deletion attempt journals."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_deletion_journal import (
    SignedRetirementRestoreDeletionJournal,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_deletions.sqlite3"
_LOCK = threading.RLock()
_JOURNALS: dict[str, SignedRetirementRestoreDeletionJournal] = {}


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


def get_signed_retirement_restore_deletion_journal(
    path: str | os.PathLike[str] | None = None,
) -> SignedRetirementRestoreDeletionJournal:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_DB_PATH",
        _DEFAULT_PATH,
    )
    deletion_path = _canonical(selected)
    protected: list[Path] = []
    for variable in (
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DELETION_AUTH_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_KEY_DB_PATH",
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
    ):
        value = os.getenv(variable)
        if value:
            protected.append(_canonical(value))
    if any(_same_file(deletion_path, value) for value in protected):
        raise RuntimeError(
            "deletion journal must not alias authorization, restore, hold, "
            "custody, signer, retirement, or publication databases."
        )
    key = str(deletion_path)
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = SignedRetirementRestoreDeletionJournal(deletion_path)
            _JOURNALS[key] = journal
        return journal


def clear_signed_retirement_restore_deletion_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = [
    "clear_signed_retirement_restore_deletion_journal_cache",
    "get_signed_retirement_restore_deletion_journal",
]
