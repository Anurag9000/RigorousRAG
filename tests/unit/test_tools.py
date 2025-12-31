import pytest
import json
import logging
from tools.integrity import generate_comparison_matrix, detect_conflicts, extract_limitations
from tools.bib import export_to_bibtex
from tools.logger import log_activity
from tools.models import Citation, AgentAnswer

class TestMetricsTools:
    def test_compare_matrix(self):
        output = generate_comparison_matrix(doc_ids=["A", "B"], metrics=["Acc"])
        assert "| Acc |" in output
        assert "[Extracted Data]" in output

    def test_detect_conflicts(self):
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
