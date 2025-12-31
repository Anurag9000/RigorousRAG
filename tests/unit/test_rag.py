import pytest
from unittest.mock import MagicMock, patch
from tools.rag import RAGLayer

class TestRAGLayer:

    @patch('tools.rag.chromadb.PersistentClient')
    @patch('tools.rag.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_init_and_add(self, mock_emb_fn, mock_chroma):
        # Setup mocks
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        # We don't skip client init because the class does it eagerly.
        rag = RAGLayer(persist_directory="dummy")
        
        # Test add_document
        rag.add_document("doc1", "This is a test document text.", {"owner": "me"})
        
        # Verify collection.add called
        assert mock_collection.add.called
        call_args = mock_collection.add.call_args[1]
        assert "ids" in call_args
        assert len(call_args["documents"]) > 0

    @patch('tools.rag.chromadb.PersistentClient')
    @patch('tools.rag.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_query(self, mock_emb_fn, mock_chroma):
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock query result
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "documents": [["Doc Content"]],
            "metadatas": [[{"foo": "bar"}]],
            "distances": [[0.1]]
        }
        
        rag = RAGLayer()
        results = rag.query("test query")
        assert len(results) == 1
        assert results[0].text == "Doc Content"

    def test_hyde_query_gen(self):
        # We can test this without mocking RAG init if we are careful,
        # OR we mock the init behavior via patch for the class.
        # But generate_hyde_query is an instance method.
        # Let's mock the whole class or just patch the client init in setUp?
        
        with patch('tools.rag.chromadb.PersistentClient'), \
             patch('tools.rag.embedding_functions.SentenceTransformerEmbeddingFunction'):
             
            rag = RAGLayer(persist_directory="dummy")
            
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value.choices[0].message.content = "Hypothetical Answer"
            
            hyde = rag.generate_hyde_query("question", mock_client)
            assert "Hypothetical Answer" in hyde
