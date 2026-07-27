"""Adapter from the classic academic index to canonical citations."""

from __future__ import annotations

import threading
from typing import List

from pydantic import BaseModel, Field

from Searching import AcademicSearchEngine, SearchHit
from tools.models import Citation

_ENGINE_INSTANCE = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> AcademicSearchEngine:
    global _ENGINE_INSTANCE
    with _ENGINE_LOCK:
        if _ENGINE_INSTANCE is None:
            _ENGINE_INSTANCE = AcademicSearchEngine()
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
