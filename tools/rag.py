import os
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Configuration
CHROMA_PATH = "rag_storage"
COLLECTION_NAME = "academic_rag"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class Chunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float = 0.0

class RAGLayer:
    def __init__(self, persist_directory: str = CHROMA_PATH):
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use simple default embedding function (sentence-transformers)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=DEFAULT_EMBEDDING_MODEL
        )
        
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any], chunk_size: int = 1000, overlap: int = 100):
        """
        Splits a document into parent and child chunks.
        Child chunks are indexed for retrieval, parents provide context.
        """
        parent_size = chunk_size * 3
        parent_overlap = overlap * 2
        
        parents = self._chunk_text(text, parent_size, parent_overlap)
        total_child_count = 0
        
        for p_idx, p_text in enumerate(parents):
            parent_id = f"{doc_id}_p{p_idx}"
            # Split parent into children
            children = self._chunk_text(p_text, chunk_size, overlap)
            
            ids = []
            documents = []
            metadatas = []
            
            for c_idx, c_text in enumerate(children):
                child_id = f"{parent_id}_c{c_idx}"
                
                chunk_meta = metadata.copy()
                chunk_meta.update({
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "parent_text": p_text[:2000], # Store parent context in child metadata
                    "is_child": True
                })
                
                ids.append(child_id)
                documents.append(c_text)
                metadatas.append(chunk_meta)
            
            if ids:
                self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
                total_child_count += len(ids)
                
        return total_child_count

    def generate_hyde_query(self, query: str, agent_client: Optional[Any] = None) -> str:
        """
        Implements HyDe (Hypothetical Document Embeddings).
        Generates a 'ideal' answer to embed for better retrieval.
        """
        if not agent_client:
            return query # Fallback to raw query if no LLM
            
        try:
            response = agent_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert academic assistant. Generate a short, factual, hypothetical paragraph that would perfectly answer the user's query. This will be used for vector retrieval."},
                    {"role": "user", "content": query}
                ],
                max_tokens=200
            )
            hyde_answer = response.choices[0].message.content or query
            return f"{query}\n{hyde_answer}"
        except Exception:
            return query

    def generate_expanded_queries(self, query: str, agent_client: Optional[Any] = None) -> List[str]:
        """
        Generates 3-5 variations of a technical query to improve recall.
        """
        if not agent_client:
            return [query]
            
        try:
            response = agent_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a research assistant. Generate 3 unique search variations of the user's technical query to maximize academic document recall. Output as a comma-separated list."},
                    {"role": "user", "content": query}
                ],
                max_tokens=100
            )
            variations = response.choices[0].message.content.split(',')
            return [query] + [v.strip() for v in variations if v.strip()]
        except Exception:
            return [query]

    def query(self, query_text: str, n_results: int = 5, where: Optional[Dict] = None, use_multi_query: bool = False, agent_client: Optional[Any] = None) -> List[Chunk]:
        """
        Retrieves top-k relevant chunks, optionally using Multi-Query Expansion.
        """
        queries = [query_text]
        if use_multi_query:
            queries = self.generate_expanded_queries(query_text, agent_client)

        all_results = []
        for q in queries:
            results = self.collection.query(
                query_texts=[q],
                n_results=n_results,
                where=where
            )
            # Process results...
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    all_results.append(Chunk(
                        id=results['ids'][0][i],
                        text=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        score=results['distances'][0][i] if 'distances' in results else 0
                    ))
        
        # Deduplicate and sort by score (if distances present)
        seen = set()
        unique_chunks = []
        for c in sorted(all_results, key=lambda x: x.score):
            if c.text not in seen:
                unique_chunks.append(c)
                seen.add(c.text)
                
        return unique_chunks[:n_results]

    def _chunk_text(self, text: str, size: int, overlap: int) -> List[str]:
        # Very simple sliding window chunker
        if size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += (size - overlap)
            
        return chunks

# Global instance
_RAG_INSTANCE = None

def get_rag_layer() -> RAGLayer:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        _RAG_INSTANCE = RAGLayer()
    return _RAG_INSTANCE
