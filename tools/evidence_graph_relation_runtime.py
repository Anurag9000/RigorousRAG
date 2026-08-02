"""Process-local factory for reviewed cross-document relation proposals."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_relation_review import RelationReviewLedger

_LOCK = threading.RLock()
_LEDGERS: dict[str, RelationReviewLedger] = {}


def get_relation_review_ledger(
    path: str | os.PathLike[str] | None = None,
) -> RelationReviewLedger:
    selected = path if path is not None else os.getenv(
        "EVIDENCE_GRAPH_RELATION_DB_PATH", "data/evidence_graph_relations.sqlite3"
    )
    candidate = Path(os.fspath(selected))
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    key = str(candidate.absolute())
    with _LOCK:
        ledger = _LEDGERS.get(key)
        if ledger is None:
            ledger = RelationReviewLedger(key)
            _LEDGERS[key] = ledger
        return ledger


def clear_relation_review_ledger_cache() -> None:
    with _LOCK:
        _LEDGERS.clear()


__all__ = ["clear_relation_review_ledger_cache", "get_relation_review_ledger"]
