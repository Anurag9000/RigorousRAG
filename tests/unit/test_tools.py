import json
from pathlib import Path

from tools.bib import export_to_bibtex
from tools.logger import log_agent_run, log_tool_call


def test_bibtex_preserves_venue_and_escapes_special_characters():
    output = export_to_bibtex([{
        "entry_type": "article",
        "title": "RAG {Safety} & Reliability_Study",
        "authors": "Doe, Jane",
        "year": 2026,
        "journal": "Journal of Tests",
        "doi": "10.1000/example",
    }])
    assert output.startswith("@article{")
    assert "journal = {Journal of Tests}" in output
    assert r"\{Safety\}" in output
    assert r"\&" in output
    assert r"Reliability\_Study" in output


def test_bibtex_keys_are_deterministic_and_unique():
    citation = {"title": "Same", "authors": "Doe, Jane", "year": 2026}
    first = export_to_bibtex([citation, citation])
    second = export_to_bibtex([citation, citation])
    assert first == second
    keys = [line.split("{", 1)[1].rstrip(",") for line in first.splitlines() if line.startswith("@")]
    assert len(keys) == len(set(keys)) == 2


def test_telemetry_hashes_queries_and_is_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setattr("tools.logger.LOG_FILE", str(path))
    log_agent_run("sensitive research query", 1.25, 2, owner_id="alice")
    log_tool_call("web_search", 0.5, False, error_type="TimeoutError")
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["details"]["query_length"] == len("sensitive research query")
    assert "sensitive research query" not in path.read_text(encoding="utf-8")
    assert entries[1]["details"]["success"] is False
    assert entries[1]["details"]["error_type"] == "TimeoutError"
