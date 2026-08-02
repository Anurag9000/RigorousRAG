"""Process-local factory for governed relation-review authorization receipts."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_relation_authorization_store import (
    RelationReviewAuthorizationStore,
)

_LOCK = threading.RLock()
_STORES: dict[str, RelationReviewAuthorizationStore] = {}


def get_relation_review_authorization_store(
    path: str | os.PathLike[str] | None = None,
) -> RelationReviewAuthorizationStore:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_REVIEW_AUTH_DB_PATH",
        "data/evidence_graph_review_authorizations.sqlite3",
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        store = _STORES.get(key)
        if store is None:
            store = RelationReviewAuthorizationStore(key)
            _STORES[key] = store
        return store


def clear_relation_review_authorization_store_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_relation_review_authorization_store_cache",
    "get_relation_review_authorization_store",
]
