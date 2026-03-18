import pytest
import json
import logging
from unittest.mock import patch, MagicMock
from tools.integrity import generate_comparison_matrix, detect_conflicts, extract_limitations
from tools.bib import export_to_bibtex
from tools.logger import log_activity
from tools.models import Citation, AgentAnswer

class TestMetricsTools:
    def test_compare_matrix(self):
        # Mock RAG and Client
        with patch('tools.integrity.get_rag_layer') as mock_rag_init, \
             patch('tools.integrity._client') as mock_client:
            mock_rag = mock_rag_init.return_value
            mock_rag.query.return_value = [MagicMock(text="extracted data")]
            
            mock_ext = MagicMock()
            mock_ext.choices[0].message.content = "Value"
            mock_client.chat.completions.create.return_value = mock_ext

            output = generate_comparison_matrix(doc_ids=["A", "B"], metrics=["Acc"])
            assert "| Acc |" in output
            assert "Value" in output

    def test_detect_conflicts(self):
        with patch('tools.integrity._client') as mock_client:
            mock_resp = MagicMock()
            mock_resp.choices[0].message.content = json.dumps({
                "topic": "Topic",
                "conflicts": [{"claim_a": "A", "claim_b": "B", "source_a": "S1", "source_b": "S2"}],
                "synthesis": "Test synth"
            })
            mock_client.chat.completions.create.return_value = mock_resp

            js = detect_conflicts("Topic", "Context")
            data = json.loads(js)
            assert "conflicts" in data
            assert data["topic"] == "Topic"

    def test_bibtex_export(self):
        citations = [{"title": "T", "authors": "A", "year": "2020", "doi": "10.1/1", "url": "u"}]
        bib = export_to_bibtex(citations)
        assert "@article{ref_1" in bib
        assert "title = {T}" in bib

    def test_logger(self, tmp_path, monkeypatch):
        # Patch LOG_FILE in logger
        import tools.logger
        test_log = tmp_path / "test.jsonl"
        monkeypatch.setattr(tools.logger, "LOG_FILE", str(test_log))
        
        log_activity("test_event", {"foo": "bar"})
        
        with open(test_log, "r") as f:
            line = f.readline()
            assert "test_event" in line
            assert "bar" in line
