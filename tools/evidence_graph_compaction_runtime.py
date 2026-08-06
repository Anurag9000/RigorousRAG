"""Process-local factory for durable evidence-graph compaction receipts."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_compaction import EvidenceGraphCompactionStore

_LOCK = threading.RLock()
_STORES: dict[str, EvidenceGraphCompactionStore] = {}


def get_evidence_graph_compaction_store(
    path: str | os.PathLike[str] | None = None,
) -> EvidenceGraphCompactionStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_COMPACTION_DB_PATH",
        "data/evidence_graph_compaction.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = EvidenceGraphCompactionStore(key)
            _STORES[key] = store
        return store


def clear_evidence_graph_compaction_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_evidence_graph_compaction_store_cache",
    "get_evidence_graph_compaction_store",
]
