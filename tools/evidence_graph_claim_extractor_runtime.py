"""Runtime factories for governed scientific claim extractor profiles."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from tools.evidence_graph_claim_extractor_registry import (
    ScientificClaimExtractorRegistry,
    load_claim_extractor_governance_policy,
)

_DEFAULT_PATH = "data/evidence_graph_claim_extractors.sqlite3"
_LOCK = threading.RLock()
_REGISTRIES: dict[Path, ScientificClaimExtractorRegistry] = {}


def _canonical(value: str | os.PathLike[str]) -> Path:
    path = Path(os.fspath(value))
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def get_scientific_claim_extractor_registry(
    path: str | os.PathLike[str] | None = None,
) -> ScientificClaimExtractorRegistry:
    selected = (
        path
        or os.getenv("EVIDENCE_GRAPH_CLAIM_EXTRACTOR_REGISTRY_DB_PATH")
        or _DEFAULT_PATH
    )
    canonical = _canonical(selected)
    with _LOCK:
        registry = _REGISTRIES.get(canonical)
        if registry is None:
            registry = ScientificClaimExtractorRegistry(canonical)
            _REGISTRIES[canonical] = registry
        return registry


def get_scientific_claim_extractor_policy():
    return load_claim_extractor_governance_policy()


def clear_scientific_claim_extractor_runtime_cache() -> None:
    with _LOCK:
        _REGISTRIES.clear()


__all__ = [
    "clear_scientific_claim_extractor_runtime_cache",
    "get_scientific_claim_extractor_policy",
    "get_scientific_claim_extractor_registry",
]
