"""Process-local runtime for custody artifact publication attempts."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_journal_boundary import (
    GovernedRestoreCustodyArtifactJournal,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3"
_LOCK = threading.RLock()
_JOURNALS: dict[str, GovernedRestoreCustodyArtifactJournal] = {}


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


def get_restore_custody_artifact_journal(
    path: str | os.PathLike[str] | None = None,
    *,
    protected_paths: tuple[str | os.PathLike[str], ...] = (),
) -> GovernedRestoreCustodyArtifactJournal:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
        _DEFAULT_PATH,
    )
    journal_path = _canonical(selected)
    protected: list[Path] = [_canonical(value) for value in protected_paths]
    for variable in (
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
    if any(_same_file(journal_path, value) for value in protected):
        raise RuntimeError(
            "custody artifact journal must not alias artifact outputs, target, "
            "custody, hold, restore, retirement, or publication databases."
        )
    key = str(journal_path)
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = GovernedRestoreCustodyArtifactJournal(journal_path)
            _JOURNALS[key] = journal
        return journal


def clear_restore_custody_artifact_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = [
    "clear_restore_custody_artifact_journal_cache",
    "get_restore_custody_artifact_journal",
]
