import os
import json
import pytest
from tools.logger import log_activity, log_tool_call, log_agent_run, LOG_FILE
from tools.bib import export_to_bibtex

class TestUtilitiesExtended:
    def test_logger_workflow(self, tmp_path):
        # Change LOG_FILE to a temp one for testing
        test_log = tmp_path / "test_metrics.jsonl"
        with patch('tools.logger.LOG_FILE', str(test_log)):
            log_activity("test", {"key": "value"})
            log_tool_call("tool1", 1.23, True, 100)
            log_agent_run("query 1", 2.5, 3)
            
            assert test_log.exists()
            lines = test_log.read_text().splitlines()
            assert len(lines) == 3
            entry1 = json.loads(lines[0])
            assert entry1["type"] == "test"
            assert entry1["details"]["key"] == "value"
            
            entry2 = json.loads(lines[1])
            assert entry2["type"] == "tool_call"
            assert entry2["details"]["tool"] == "tool1"
            
            entry3 = json.loads(lines[2])
            assert entry3["type"] == "agent_run"
            assert entry3["details"]["query"] == "query 1"

    def test_export_to_bibtex(self):
        citations = [
            {"title": "Paper 1", "authors": "Author A", "year": "2021", "url": "http://a.com", "doi": "10.1"},
            {"title": "Paper 2"} # Test defaults
        ]
        bib = export_to_bibtex(citations)
        assert "@article{ref_1" in bib
        assert "title = {Paper 1}" in bib
        assert "author = {Author A}" in bib
        assert "10.1" in bib
        
        assert "@article{ref_2" in bib
        assert "author = {Unknown}" in bib
        assert "year = {n.d.}" in bib

# Import patch here for helper
from unittest.mock import patch
