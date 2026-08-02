from __future__ import annotations

import json

import pytest

from tools import evidence_graph_rag_benchmark_cli as cli

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
QUERY = "c" * 64
SELECTION = "d" * 64
NODE = "1" * 64


def fixture():
    return {
        "benchmark_id": "graph-rag-v1",
        "schema_version": 1,
        "runs": [
            {
                "run_id": "run-1",
                "seed": 1,
                "cases": [
                    {
                        "gold": {
                            "query_id": "q1",
                            "graph_set_id": SET_ID,
                            "graph_set_digest": SET_DIGEST,
                            "query_digest": QUERY,
                            "relevant_nodes": [
                                {"doc_id": "doc-a", "generation": 1, "node_id": NODE}
                            ],
                            "required_edge_ids": [],
                            "should_abstain": False,
                        },
                        "observation": {
                            "graph_set_id": SET_ID,
                            "graph_set_digest": SET_DIGEST,
                            "query_digest": QUERY,
                            "selection_digest": SELECTION,
                            "selected_nodes": [
                                {"doc_id": "doc-a", "generation": 1, "node_id": NODE}
                            ],
                            "traversal_edge_ids": [],
                            "expanded_lineage_valid": [],
                            "abstained": False,
                            "evidence_count": 1,
                            "traversal_count": 0,
                            "estimated_work_units": 2,
                        },
                    }
                ],
            }
        ],
    }


def parse(capsys):
    value = capsys.readouterr()
    return (
        json.loads(value.out) if value.out else None,
        json.loads(value.err) if value.err else None,
    )


def test_inspect_and_run_are_text_free(tmp_path, capsys):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")
    assert cli.main(["inspect", str(path)]) == 0
    output, error = parse(capsys)
    assert error is None and output["run_count"] == 1
    assert output["contains_raw_query"] is False
    report_path = tmp_path / "report.json"
    assert cli.main(["run", str(path), "--output-file", str(report_path)]) == 0
    output, error = parse(capsys)
    assert error is None
    assert output["aggregate"]["macro_node_f1"] == 1.0
    assert output["contains_evidence_text"] is False
    assert json.loads(report_path.read_text())["report_digest"] == output["report_digest"]


def test_duplicate_keys_nan_and_unknown_fields_fail_closed(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text('{"benchmark_id":"a","benchmark_id":"b"}', encoding="utf-8")
    assert cli.main(["inspect", str(path)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    path.write_text('{"benchmark_id":"a","runs":[],"schema_version":NaN}', encoding="utf-8")
    assert cli.main(["inspect", str(path)]) == 2
    parse(capsys)
    value = fixture()
    value["raw_query"] = "private"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert cli.main(["inspect", str(path)]) == 2


def test_symlink_fixture_and_output_are_refused(tmp_path, capsys):
    target = tmp_path / "fixture.json"
    target.write_text(json.dumps(fixture()), encoding="utf-8")
    link = tmp_path / "fixture-link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert cli.main(["inspect", str(link)]) == 2
    parse(capsys)
    output_target = tmp_path / "output.json"
    output_target.write_text("{}", encoding="utf-8")
    output_link = tmp_path / "output-link.json"
    output_link.symlink_to(output_target)
    assert cli.main(["run", str(target), "--output-file", str(output_link)]) == 2


def test_report_write_is_atomic_and_replaces_regular_file(tmp_path, capsys):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")
    output = tmp_path / "report.json"
    output.write_text("old", encoding="utf-8")
    assert cli.main(["run", str(path), "--output-file", str(output)]) == 0
    parse(capsys)
    value = json.loads(output.read_text())
    assert value["benchmark_id"] == "graph-rag-v1"
    assert len(value["report_digest"]) == 64
