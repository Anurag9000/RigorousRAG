import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import os
import io

# Mocking before import to prevent expensive initialisations
with patch('search_agent.SearchAgent'), patch('tools.rag.get_rag_layer'):
    from server import app, UPLOAD_DIR

client = TestClient(app)

class TestServer:
    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Academic Search Engine API is running."}

    @patch('server.agent')
    def test_run_query(self, mock_agent):
        mock_agent.run.return_value = MagicMock(
            answer="Test Answer",
            citations=[],
            metadata={}
        )
        # Use a fake dict that Pydantic can validate as AgentAnswer
        mock_agent.run.return_value.model_dump.return_value = {
            "answer": "Test Answer",
            "citations": [],
            "metadata": {}
        }
        
        response = client.post("/query", json={"query": "test query", "model": "gpt-fake"})
        assert response.status_code == 200
        assert "Test Answer" in response.json()["answer"]
        assert mock_agent.model == "gpt-fake"

    @patch('server.process_ingestion')
    def test_ingest_document(self, mock_process):
        # Create a dummy file
        file_content = b"fake pdf content"
        file = io.BytesIO(file_content)
        
        response = client.post(
            "/ingest",
            files={"file": ("test.pdf", file, "application/pdf")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["filename"] == "test.pdf"
        assert "job_" in data["job_id"]
        
        # Verify file was saved
        saved_file = UPLOAD_DIR / "test.pdf"
        assert saved_file.exists()
        assert saved_file.read_bytes() == file_content
        
        # Cleanup
        saved_file.unlink()

    @patch('server.ingest_file')
    @patch('server.get_rag_layer')
    def test_process_ingestion_worker(self, mock_get_rag, mock_ingest):
        mock_doc = MagicMock()
        mock_doc.id = "id1"
        mock_doc.text = "text"
        mock_doc.filename = "f.txt"
        mock_doc.mime_type = "text/plain"
        
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.document = mock_doc
        mock_ingest.return_value = mock_result
        
        mock_rag = mock_get_rag.return_value
        
        from server import process_ingestion
        process_ingestion("fake_path", "job_123")
        
        assert mock_ingest.called
        assert mock_rag.add_document.called
        assert mock_rag.add_document.call_args[1]["doc_id"] == "id1"

    def test_run_query_error(self):
        with patch('server.agent.run', side_effect=Exception("Agent crash")):
            response = client.post("/query", json={"query": "fail"})
            assert response.status_code == 500
            assert "Agent crash" in response.json()["detail"]
