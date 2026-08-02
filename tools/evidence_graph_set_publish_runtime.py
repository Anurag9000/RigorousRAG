"""Process-local factory for durable graph-set publication attempts."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_set_publish_attempts import (
    EvidenceGraphSetPublicationJournal,
)

_LOCK = threading.RLock()
_JOURNALS: dict[str, EvidenceGraphSetPublicationJournal] = {}


def get_evidence_graph_set_publication_journal(
    path: str | os.PathLike[str] | None = None,
) -> EvidenceGraphSetPublicationJournal:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_SET_PUBLICATION_DB_PATH",
        "data/evidence_graph_set_publications.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = EvidenceGraphSetPublicationJournal(key)
            _JOURNALS[key] = journal
        return journal


def clear_evidence_graph_set_publication_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = [
    "clear_evidence_graph_set_publication_journal_cache",
    "get_evidence_graph_set_publication_journal",
]
