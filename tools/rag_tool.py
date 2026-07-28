"""Agent tool for owner-scoped uploaded-document retrieval."""

from __future__ import annotations

from typing import Any, List, Optional

from tools.models import Citation
from tools.rag import get_rag_layer
from tools.security import normalize_owner_id

RAG_SEARCH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "search_uploaded_docs",
        "description": (
            "Search only the authenticated user's uploaded documents. Optionally "
            "restrict retrieval to one document ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 10_000,
                    "description": "Question or topic to search for.",
                },
                "doc_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": "Optional exact document ID from the document library.",
                },
                "use_hyde": {
                    "type": "boolean",
                    "description": "Use one hypothetical evidence passage to improve recall.",
                    "default": False,
                },
                "use_multi_query": {
                    "type": "boolean",
                    "description": "Generate a small number of alternative retrieval queries.",
                    "default": False,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def search_uploaded_docs(
    query: str,
    *,
    owner_id: str = "default_user",
    doc_id: Optional[str] = None,
    use_hyde: bool = False,
    use_multi_query: bool = False,
    agent_client: Optional[Any] = None,
    expansion_model: str = "gpt-4o-mini",
    n_results: int = 5,
) -> List[Citation]:
    """Retrieve evidence with mandatory owner and document provenance checks."""

    retrieval_query = (query or "").strip()
    if not retrieval_query:
        return []
    if len(retrieval_query) > 10_000:
        raise ValueError("Uploaded-document queries may contain at most 10,000 characters.")
    owner = normalize_owner_id(owner_id)
    document_id = (doc_id or "").strip() or None
    if document_id is not None and len(document_id) > 200:
        raise ValueError("doc_id may contain at most 200 characters.")

    rag = get_rag_layer()
    if use_hyde:
        retrieval_query = rag.generate_hyde_query(
            retrieval_query,
            agent_client,
            model=expansion_model,
        )
    chunks = rag.query(
        retrieval_query,
        n_results=n_results,
        owner_id=owner,
        doc_id=document_id,
        use_multi_query=use_multi_query,
        agent_client=agent_client,
        expansion_model=expansion_model,
    )

    citations: List[Citation] = []
    for chunk in chunks:
        metadata = chunk.metadata or {}
        metadata_owner = str(metadata.get("owner_id") or "").strip()
        actual_doc_id = str(metadata.get("doc_id") or "").strip()
        if metadata_owner != owner or not actual_doc_id or len(actual_doc_id) > 200:
            continue
        if document_id is not None and actual_doc_id != document_id:
            continue
        parent_text = str(metadata.get("parent_text") or chunk.text).strip()
        page_number = metadata.get("page_number")
        if not isinstance(page_number, int) or page_number < 1:
            page_number = None
        citations.append(
            Citation(
                label=f"[{len(citations) + 1}]",
                title=str(metadata.get("filename") or "Uploaded document"),
                url=f"local://{actual_doc_id}",
                source_type="uploaded_document",
                snippet=parent_text,
                quote=chunk.text,
                source_id=chunk.id,
                doc_id=actual_doc_id,
                chunk_id=chunk.id,
                page_number=page_number,
                metadata={
                    "section_title": metadata.get("section_title"),
                    "relevance": round(chunk.score, 6),
                },
            )
        )
    return citations
