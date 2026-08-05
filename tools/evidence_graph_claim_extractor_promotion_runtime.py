"""Runtime factories for governed scientific claim extractor promotion."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_claim_extractor_promotion import (
    load_claim_extractor_promotion_policy,
)
from tools.evidence_graph_claim_extractor_promotion_transactional import (
    TransactionalScientificClaimExtractorPromotionStore,
)
from tools.evidence_graph_claim_extractor_runtime import (
    get_scientific_claim_extractor_registry,
)

_DEFAULT_PATH = "data/evidence_graph_claim_extractor_promotions.sqlite3"
_LOCK = threading.RLock()
_STORES: dict[Path, TransactionalScientificClaimExtractorPromotionStore] = {}


def _canonical(value: str | os.PathLike[str]) -> Path:
    path = Path(os.fspath(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def get_scientific_claim_extractor_promotion_store(
    path: str | os.PathLike[str] | None = None,
) -> TransactionalScientificClaimExtractorPromotionStore:
    selected = (
        path
        or os.getenv("EVIDENCE_GRAPH_CLAIM_EXTRACTOR_PROMOTION_DB_PATH")
        or _DEFAULT_PATH
    )
    canonical = _canonical(selected)
    registry = get_scientific_claim_extractor_registry()
    if canonical == registry.path:
        raise RuntimeError("promotion database must not alias extractor registry.")
    if canonical.exists() and registry.path.exists():
        promotion_info = canonical.lstat()
        registry_info = registry.path.lstat()
        if (
            int(promotion_info.st_dev),
            int(promotion_info.st_ino),
        ) == (
            int(registry_info.st_dev),
            int(registry_info.st_ino),
        ):
            raise RuntimeError("promotion database must not hard-link to extractor registry.")
    with _LOCK:
        store = _STORES.get(canonical)
        if store is None:
            store = TransactionalScientificClaimExtractorPromotionStore(canonical)
            _STORES[canonical] = store
        return store


def get_scientific_claim_extractor_promotion_policy():
    return load_claim_extractor_promotion_policy()


def clear_scientific_claim_extractor_promotion_runtime_cache() -> None:
    with _LOCK:
        _STORES.clear()


__all__ = [
    "clear_scientific_claim_extractor_promotion_runtime_cache",
    "get_scientific_claim_extractor_promotion_policy",
    "get_scientific_claim_extractor_promotion_store",
]
