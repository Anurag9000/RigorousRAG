from typing import List, Optional
from pydantic import BaseModel, Field
from tools.rag import get_rag_layer
from tools.models import Citation

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
    
    # In a real system, we'd pass the agent's client here for HyDe
    # For now, let's assume get_rag_layer can access a default client or just use raw query
    chunks = rag.query(query, n_results=5, use_hyde=use_hyde)
    
    citations = []
    for idx, chunk in enumerate(chunks, start=1):
        # Goal 20: Provide rich parent context
        text = chunk.text
        if "parent_text" in chunk.metadata:
             text = f"... {chunk.metadata['parent_text']} ..."
             
        citations.append(Citation(
            label=f"[doc-{idx}]",
            title=chunk.metadata.get("filename", "Uploaded Doc"),
            url=f"local://{chunk.metadata.get('doc_id')}",
            source_type="internal_index",
            snippet=text
        ))
    return citations
