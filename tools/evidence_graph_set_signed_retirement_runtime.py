"""Process-local factory for the signed publication retirement saga journal."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_signed_retirement_journal import (
    SignedPublicationRetirementJournal,
)

_DEFAULT_RETIREMENT_PATH = "data/evidence_graph_set_signed_retirements.sqlite3"
_DEFAULT_PUBLICATION_PATH = "data/evidence_graph_set_publications.sqlite3"
_DEFAULT_SIGNED_PUBLICATION_PATH = (
    "data/evidence_graph_set_signed_publications.sqlite3"
)
_LOCK = threading.RLock()
_JOURNALS: dict[str, SignedPublicationRetirementJournal] = {}


def _absolute(value: str | os.PathLike[str]) -> Path:
    candidate = Path(os.fspath(value))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(candidate))


def _aliases(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        if left.exists() and right.exists():
            return os.path.samefile(left, right)
    except OSError as exc:
        raise RuntimeError("retirement journal alias could not be validated.") from exc
    return False


def get_signed_publication_retirement_journal(
    path: str | os.PathLike[str] | None = None,
) -> SignedPublicationRetirementJournal:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_DB_PATH",
        _DEFAULT_RETIREMENT_PATH,
    )
    candidate = _absolute(selected)
    publication = _absolute(
        os.getenv(
            "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
            _DEFAULT_PUBLICATION_PATH,
        )
    )
    signed_publication = _absolute(
        os.getenv(
            "EVIDENCE_GRAPH_SET_SIGNED_PUBLICATION_DB_PATH",
            _DEFAULT_SIGNED_PUBLICATION_PATH,
        )
    )
    if _aliases(candidate, publication) or _aliases(candidate, signed_publication):
        raise RuntimeError(
            "retirement, authorization-only publication and signed publication "
            "journals must use distinct files."
        )
    key = str(candidate)
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = SignedPublicationRetirementJournal(candidate)
            _JOURNALS[key] = journal
        return journal


def clear_signed_publication_retirement_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = [
    "clear_signed_publication_retirement_journal_cache",
    "get_signed_publication_retirement_journal",
]
