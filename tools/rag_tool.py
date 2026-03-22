"""
RAG document search tool with HyDE query expansion, multi-query retrieval,
and per-owner document isolation for multi-tenant support.
"""

import os
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from tools.rag import get_rag_layer
from tools.models import Citation

# OpenAI client used for HyDE hypothetical document generation
_hyde_client = None
if OpenAI is not None:
    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        _hyde_client = OpenAI(api_key=_api_key)

# ---------------------------------------------------------------------------
# Tool schema definition
# ---------------------------------------------------------------------------

RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs",
        "description": (
            "Search specifically within user-uploaded documents (PDFs, Word files, etc.). "
            "Use this when the user asks about 'uploaded', 'my files', or 'ingested' documents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question or topic to search for in uploaded documents.",
                }
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Search implementation
# ---------------------------------------------------------------------------


def search_uploaded_docs(
    query: str,
    use_hyde: bool = True,
    owner_id: Optional[str] = None,
) -> List[Citation]:
    """
    Search ingested documents using semantic vector retrieval.

    Enhancements enabled:
    - **HyDE (Goal 19):** Generates a hypothetical ideal answer and uses its
      embedding instead of the raw query to improve recall.
    - **Multi-query expansion:** Expands the query into 3 variations and merges
      results, deduplicating by text to improve coverage.
    - **Owner isolation (Gap 3):** When `owner_id` is provided (and is not the
      sentinel "default_user"), the ChromaDB `where` filter restricts results
      to documents belonging to that owner.
    - **Parent-context expansion (Goal 20):** Returns parent-chunk text when
      available for richer citation snippets.
    """
    rag = get_rag_layer()

    # --- HyDE query expansion ---
    hyde_query = query
    if use_hyde and _hyde_client:
        try:
            resp = _hyde_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate a short, technically accurate hypothetical scientific "
                            "snippet that would answer the user's query.  This will be used "
                            "for vector similarity search — aim for domain-specific vocabulary."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=300,
            )
            hyde_query = resp.choices[0].message.content or query
        except Exception:
            hyde_query = query

    # --- Multi-tenant owner isolation ---
    where_filter: Optional[dict] = None
    if owner_id and owner_id != "default_user":
        where_filter = {"owner_id": owner_id}

    # --- Multi-query retrieval (Gap 6 enabled) ---
    chunks = rag.query(
        hyde_query,
        n_results=5,
        where=where_filter,
        use_multi_query=True,          # previously hard-coded False — now active
        agent_client=_hyde_client,
    )

    # --- Map results to Citation objects ---
    citations: List[Citation] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata or {}
        # Parent context expansion (Goal 20): use parent_text for richer snippet
        snippet = chunk.text
        if "parent_text" in meta:
            snippet = f"... {meta['parent_text']} ..."

        citations.append(
            Citation(
                label=f"[doc-{idx}]",
                title=meta.get("filename", "Uploaded Document"),
                url=f"local://{meta.get('doc_id', 'unknown')}",
                source_type="internal_index",
                snippet=snippet,
            )
        )

    return citations
