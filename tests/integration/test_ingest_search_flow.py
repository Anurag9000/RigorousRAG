import pytest
import os
from unittest.mock import MagicMock, patch
from tools.rag import RAGLayer
from tools.ingestion import ingest_file

class TestIngestionRagFlow:

    def test_ingest_and_retrieve(self, tmp_path):
        # 1. Create a dummy file
        doc_path = tmp_path / "research_notes.txt"
        doc_path.write_text("Project Alpha results: 95% accuracy in 2024.", encoding="utf-8")
        
        # 2. Ingest
        # ingest_file parses the file. It does NOT write to RAG directly.
        result = ingest_file(str(doc_path))
        assert result.success
        assert result.document is not None
        assert "Project Alpha" in result.document.text
        
        # 3. Add to RAG (Integration Step)
        
        # We need a RAGLayer instance.
        # Ideally we use a real minimal one or mocked one handling embeddings.
        # Real one needs chromadb and sentence-transformers model (heavy).
        # For integration test purity, we should use real components IF FAST.
        # But this might download models.
        # Let's mock the embedding function generation to be deterministic and fast,
        # but keep ChromaDB in-memory or temp dir.
        
        rag_dir = tmp_path / "rag_db"
        
        with patch('tools.rag.embedding_functions.SentenceTransformerEmbeddingFunction') as mock_emb:
             # Mock embedding to return fixed vector
             # Signature must match __call__(self, input)
             mock_emb.return_value = MagicMock(side_effect=lambda input: [[0.1]*384 for _ in input])
             # Wait, usually side_effect receives proper args. 
             # If we mock the instance, instance(input) -> side_effect(input).
             # If Chromadb calls instance(input=...), keywargs might be used.
             # Safe mock for ChromaDB's signature check
             class MockEmbeddingFunction:
                 def __call__(self, input):
                     return [[0.1]*384 for _ in input]
                 def embed_documents(self, input):
                     return self.__call__(input)
                 def embed_query(self, input):
                     return self.__call__(input)
                 def name(self):
                     return "default"
             
             mock_emb.return_value = MockEmbeddingFunction()
             
             rag = RAGLayer(persist_directory=str(rag_dir))
             
             # Manually add sections
             for section in result.document.sections:
                 rag.add_document(
                     doc_id="doc_1",
                     text=section.content,
                     metadata=result.document.metadata
                 )
             
             # 4. Retrieve
             # Query
             hits = rag.query("accuracy")
             # Since we mocked embedding to be constant, all docs match any query with constant score/distance?
             # Actually, if query embedding is [0.1...], and doc is [0.1...], distance is 0.
             assert len(hits) >= 1
             assert "Project Alpha" in hits[0].text
