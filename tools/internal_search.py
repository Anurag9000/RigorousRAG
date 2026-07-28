"""Adapter from the classic academic index to canonical citations."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from Searching import AcademicSearchEngine, SearchHit
from tools.models import Citation

_ENGINE_INSTANCE: Optional[AcademicSearchEngine] = None
_ENGINE_SIGNATURE: Optional[Tuple[str, Tuple[Tuple[str, int, int], ...]]] = None
_ENGINE_LOCK = threading.Lock()


def _storage_signature(storage_dir: str) -> Tuple[str, Tuple[Tuple[str, int, int], ...]]:
    root = Path(storage_dir).resolve()
    paths = [
        root / "snapshot_manifest.json",
        root / "crawl_state.json",
        root / "index.json",
        root / "pagerank.json",
    ]
    entries = []
    for path in paths:
        try:
            stat = path.lstat()
            entries.append((path.name, int(stat.st_mtime_ns), int(stat.st_size)))
        except OSError:
            entries.append((path.name, -1, -1))
    return str(root), tuple(entries)


def get_engine() -> AcademicSearchEngine:
    """Return an engine reloaded whenever the committed storage signature changes."""

    global _ENGINE_INSTANCE, _ENGINE_SIGNATURE
    storage_dir = os.getenv("CLASSIC_STORAGE_DIR", "data")
    signature = _storage_signature(storage_dir)
    with _ENGINE_LOCK:
        if _ENGINE_INSTANCE is None or _ENGINE_SIGNATURE != signature:
            _ENGINE_INSTANCE = AcademicSearchEngine(storage_dir=storage_dir)
            _ENGINE_SIGNATURE = _storage_signature(storage_dir)
        return _ENGINE_INSTANCE


class InternalSearchInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


INTERNAL_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_internal",
        "description": "Search the locally crawled academic web index.",
        "parameters": InternalSearchInput.model_json_schema(),
    },
}


def search_internal(query: str, limit: int = 5) -> List[Citation]:
    query = (query or "").strip()
    if not query:
        return []
    if len(query) > 2000:
        raise ValueError("Internal-search queries may contain at most 2,000 characters.")
    limit = max(1, min(int(limit), 20))
    hits: List[SearchHit] = get_engine().search(query, limit=limit)
    return [
        Citation(
            label=f"[{index}]",
            title=hit.title,
            url=hit.url,
            source_type="academic_index",
            snippet=hit.snippet,
            source_id=hit.url,
            metadata={
                "combined_score": hit.score,
                "cosine": hit.cosine,
                "pagerank": hit.pagerank,
            },
        )
        for index, hit in enumerate(hits, start=1)
    ]
