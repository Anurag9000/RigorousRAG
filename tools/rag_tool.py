import os
from typing import List, Optional, Any
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None # type: ignore

from tools.rag import get_rag_layer
from tools.models import Citation

# Global client for HyDe
_hyde_client = None
if OpenAI is not None:
    _api_key = os.getenv("OPENAI_API_KEY")
    if _api_key:
        _hyde_client = OpenAI(api_key=_api_key)

class RagSearchInput(BaseModel):
    query: str = Field(..., description="Query to search in the uploaded documents.")

RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs",
        "description": "Search specifically within user-uploaded documents (PDFs, Word docs, etc). use this when the user asks about 'uploaded' or 'my' files.",
        "parameters": RagSearchInput.model_json_schema()
    }
}

def search_uploaded_docs(query: str, use_hyde: bool = True) -> List[Citation]:
    rag = get_rag_layer()

    hyde_query = query
    if use_hyde and _hyde_client:
        try:
            # Goal 19: HyDe (Hypothetical Document Embeddings)
            resp = _hyde_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Generate a short, technically accurate, hypothetical scientific snippet that would answer the user's query. This will be used for vector similarity search."},
                    {"role": "user", "content": query}
                ],
                max_tokens=300
            )
            hyde_query = resp.choices[0].message.content or query
        except Exception:
            hyde_query = query

    chunks = rag.query(hyde_query, n_results=5, use_multi_query=False)

    citations = []
    for idx, chunk in enumerate(chunks, start=1):
        # Goal 20: Provide rich parent context
        text = chunk.text
        if chunk.metadata and "parent_text" in chunk.metadata:
             text = f"... {chunk.metadata['parent_text']} ..."

        citations.append(Citation(
            label=f"[doc-{idx}]",
            title=chunk.metadata.get("filename", "Uploaded Doc") if chunk.metadata else "Uploaded Doc",
            url=f"local://{chunk.metadata.get('doc_id')}" if chunk.metadata else "local://unknown",
            source_type="internal_index",
            snippet=text
        ))
    return citations
