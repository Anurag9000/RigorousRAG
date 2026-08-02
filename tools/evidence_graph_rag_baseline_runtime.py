"""Process-local factory for governed GraphRAG historical baselines."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_rag_baseline import GraphRAGBaselineStore

_LOCK = threading.RLock()
_STORES: dict[str, GraphRAGBaselineStore] = {}


def get_graph_rag_baseline_store(
    path: str | os.PathLike[str] | None = None,
) -> GraphRAGBaselineStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_RAG_BASELINE_DB_PATH",
        "data/evidence_graph_rag_baselines.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = GraphRAGBaselineStore(key)
            _STORES[key] = store
        return store


def clear_graph_rag_baseline_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_graph_rag_baseline_store_cache",
    "get_graph_rag_baseline_store",
]
