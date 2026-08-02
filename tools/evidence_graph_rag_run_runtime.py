"""Process-local factory for resumable evidence-graph benchmark runs."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_rag_run_store import GraphRAGBenchmarkRunStore

_LOCK = threading.RLock()
_STORES: dict[str, GraphRAGBenchmarkRunStore] = {}


def get_graph_rag_run_store(
    path: str | os.PathLike[str] | None = None,
) -> GraphRAGBenchmarkRunStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_RAG_RUN_DB_PATH", "data/evidence_graph_rag_runs.sqlite3"
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = GraphRAGBenchmarkRunStore(key)
            _STORES[key] = store
        return store


def clear_graph_rag_run_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = ["clear_graph_rag_run_store_cache", "get_graph_rag_run_store"]
