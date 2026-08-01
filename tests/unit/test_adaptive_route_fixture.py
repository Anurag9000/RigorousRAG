from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.adaptive_route_experiments import run_route_benchmark
from tools.adaptive_route_fixture import fixture_adapters, load_route_fixtures


def fixture_line():
    return {
        "case": {
            "case_id": "case-1",
            "query": "latest policy update",
            "scope": "public",
            "domain": "general",
            "relevant_ids": ["web-source"],
        },
        "routes": {
            "web": {
                "cost_units": 4,
                "latency_ms": 30,
                "evidence": [
                    {
                        "source_id": "web-source",
                        "doc_id": "web-doc",
                        "score": 0.95,
                        "page_number": 1,
                        "source_kind": "web",
                    }
                ],
            },
            "dense": {
                "evidence": [
                    {"source_id": "dense-source", "doc_id": "dense-doc", "score": 0.1}
                ]
            },
        },
    }


def test_fixture_loader_and_benchmark_are_reproducible(tmp_path):
    path = tmp_path / "routes.jsonl"
    path.write_text(json.dumps(fixture_line()) + "\n", encoding="utf-8")
    fixtures = load_route_fixtures(path)
    report = run_route_benchmark(
        [fixture.case for fixture in fixtures], adapters=fixture_adapters(fixtures)
    )
    assert report.case_count == 1
    assert report.cases[0].selected_route == "web"
    assert report.cases[0].selected_success is True
    assert "latest policy update" not in repr(report)


def test_fixture_loader_rejects_duplicate_keys_nonstandard_numbers_and_unknown_fields(tmp_path):
    path = tmp_path / "routes.jsonl"
    path.write_text(
        '{"case":{"case_id":"a","case_id":"b","query":"q"},"routes":{"dense":{}}}\n'
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        load_route_fixtures(path)
    path.write_text(
        '{"case":{"case_id":"a","query":"q"},"routes":{"dense":{"cost_units":NaN}}}\n'
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        load_route_fixtures(path)
    payload = fixture_line()
    payload["extra"] = True
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="unknown fields"):
        load_route_fixtures(path)


def test_fixture_loader_rejects_symlinked_paths(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text(json.dumps(fixture_line()) + "\n")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="symbolic links"):
        load_route_fixtures(link)


def test_cli_report_contains_no_raw_query_or_evidence_text(tmp_path):
    path = tmp_path / "routes.jsonl"
    path.write_text(json.dumps(fixture_line()) + "\n")
    completed = subprocess.run(
        [sys.executable, "scripts/run_adaptive_route_benchmark.py", str(path)],
        cwd=str(Path(__file__).resolve().parents[2]),
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["case_count"] == 1
    assert "latest policy update" not in completed.stdout
    assert "web-source" not in completed.stdout
