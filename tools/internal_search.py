from typing import List, Dict, Any
from pydantic import BaseModel, Field
from Searching import AcademicSearchEngine, SearchHit
from tools.models import Citation

# Global instance to avoid reloading index on every call
_ENGINE_INSTANCE = None

def get_engine() -> AcademicSearchEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        # Initialize with defaults; this loads existing index/crawl state
        _ENGINE_INSTANCE = AcademicSearchEngine()
    return _ENGINE_INSTANCE

class InternalSearchInput(BaseModel):
    query: str = Field(..., description="The search query.")
    limit: int = Field(5, description="Maximum number of results to return.")

INTERNAL_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_internal",
        "description": "Search the internal academic crawler index.",
        "parameters": InternalSearchInput.model_json_schema()
    }
}

def search_internal(query: str, limit: int = 5) -> List[Citation]:
    """
    Searches the internal academic index.
    """
    engine = get_engine()
    hits: List[SearchHit] = engine.search(query, limit=limit)
    
    citations: List[Citation] = []
    for idx, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                label=f"[{idx}]",
                title=hit.title,
                url=hit.url,
                source_type="internal_index",
                snippet=hit.snippet
            )
        )
    return citations
