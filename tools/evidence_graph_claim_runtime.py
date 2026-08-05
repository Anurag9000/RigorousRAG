"""Runtime factories for scientific claim extraction review."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_claim_review import load_claim_review_policy
from tools.evidence_graph_claim_store import ScientificClaimReviewStore

_DEFAULT_PATH = "data/evidence_graph_claim_reviews.sqlite3"
_LOCK = threading.RLock()
_STORES: dict[Path, ScientificClaimReviewStore] = {}


def _canonical(value: str | os.PathLike[str]) -> Path:
    path = Path(os.fspath(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def get_scientific_claim_review_store(
    path: str | os.PathLike[str] | None = None,
) -> ScientificClaimReviewStore:
    selected = path or os.getenv("EVIDENCE_GRAPH_CLAIM_REVIEW_DB_PATH") or _DEFAULT_PATH
    canonical = _canonical(selected)
    with _LOCK:
        store = _STORES.get(canonical)
        if store is None:
            store = ScientificClaimReviewStore(canonical)
            _STORES[canonical] = store
        return store


def get_scientific_claim_review_policy():
    return load_claim_review_policy()


def clear_scientific_claim_review_runtime_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_scientific_claim_review_runtime_cache",
    "get_scientific_claim_review_policy",
    "get_scientific_claim_review_store",
]
