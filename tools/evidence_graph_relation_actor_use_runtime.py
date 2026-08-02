"""Process-local factory for signed review actor-use journals."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_relation_actor_use_store import SignedActorUseStore

_LOCK = threading.RLock()
_STORES: dict[str, SignedActorUseStore] = {}


def get_signed_actor_use_store(
    path: str | os.PathLike[str] | None = None,
) -> SignedActorUseStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_REVIEW_ACTOR_USE_DB_PATH",
        "data/evidence_graph_review_actor_uses.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = SignedActorUseStore(key)
            _STORES[key] = store
        return store


def clear_signed_actor_use_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = ["clear_signed_actor_use_store_cache", "get_signed_actor_use_store"]
