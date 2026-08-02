"""Process-local factory for derived evidence-graph reconciliation jobs."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_jobs import EvidenceGraphJobJournal

_LOCK = threading.RLock()
_JOURNALS: dict[str, EvidenceGraphJobJournal] = {}


def get_evidence_graph_job_journal(
    path: str | os.PathLike[str] | None = None,
) -> EvidenceGraphJobJournal:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_JOB_DB_PATH",
        "data/evidence_graph_jobs.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        journal = _JOURNALS.get(key)
        if journal is None:
            journal = EvidenceGraphJobJournal(key)
            _JOURNALS[key] = journal
        return journal


def clear_evidence_graph_job_journal_cache() -> None:
    with _LOCK:
        _JOURNALS.clear()


__all__ = [
    "clear_evidence_graph_job_journal_cache",
    "get_evidence_graph_job_journal",
]
