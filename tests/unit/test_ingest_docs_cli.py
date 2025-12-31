import pytest
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from ingest_docs import main

class TestIngestDocsCLI:
    @pytest.fixture
    def mock_ingestion_result(self):
        doc = MagicMock()
        doc.id = "doc1"
        doc.text = "Hello world"
        doc.filename = "test.txt"
        doc.mime_type = "text/plain"
        doc.model_dump.return_value = {"id": "doc1", "text": "Hello world"}
        
        result = MagicMock()
        result.success = True
        result.document = doc
        result.error = None
        return result

    def test_main_single_file(self, tmp_path, mock_ingestion_result):
        f = tmp_path / "test.txt"
        f.write_text("test")
        
        with patch('sys.argv', ['ingest_docs.py', str(f)]):
            with patch('ingest_docs.ingest_file', return_value=mock_ingestion_result) as mock_ingest:
                with patch('ingest_docs.get_rag_layer', return_value=None): # Mock none/missing
                    with patch('builtins.print'):
                        main()
        
        assert mock_ingest.called

    def test_main_directory_logic(self, tmp_path, mock_ingestion_result):
        # Create a temp dir structure
        d = tmp_path / "subdir"
        d.mkdir()
        f1 = d / "file1.txt"
        f1.write_text("content1")
        f2 = tmp_path / "file2.txt"
        f2.write_text("content2")
        
        with patch('sys.argv', ['ingest_docs.py', str(tmp_path), '--recursive', '--output', str(tmp_path / 'out.json')]):
            with patch('ingest_docs.ingest_file', return_value=mock_ingestion_result) as mock_ingest:
                with patch('ingest_docs.get_rag_layer') as mock_rag_init:
                    mock_rag = mock_rag_init.return_value
                    with patch('builtins.print'):
                        main()
                        
                    # Should find 2 files
                    assert mock_ingest.call_count == 2
                    assert mock_rag.add_document.call_count == 2
                    
        # Check output file
        out_file = tmp_path / 'out.json'
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 2

    def test_main_rag_failure(self, tmp_path, mock_ingestion_result):
        f = tmp_path / "test.txt"
        f.write_text("test")
        
        # Test the exception catch in RAG initialization
        with patch('sys.argv', ['ingest_docs.py', str(f)]):
            with patch('ingest_docs.ingest_file', return_value=mock_ingestion_result):
                with patch('ingest_docs.get_rag_layer', side_effect=Exception("Initialization failed")):
                    with patch('builtins.print') as mock_print:
                        main()
                        mock_print.assert_any_call("Warning: RAG Layer could not be initialized. Indexing will be skipped. Error: Initialization failed")

    def test_main_indexing_failure(self, tmp_path, mock_ingestion_result):
        f = tmp_path / "test.txt"
        f.write_text("test")
        
        # Test the exception catch during actual indexing
        with patch('sys.argv', ['ingest_docs.py', str(f)]):
            with patch('ingest_docs.ingest_file', return_value=mock_ingestion_result):
                with patch('ingest_docs.get_rag_layer') as mock_rag_init:
                    mock_rag = mock_rag_init.return_value
                    mock_rag.add_document.side_effect = Exception("Indexing failed")
                    
                    with patch('builtins.print') as mock_print:
                        main()
                        mock_print.assert_any_call("(Index Failed: Indexing failed)", end=" ")
