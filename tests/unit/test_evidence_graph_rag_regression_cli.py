from __future__ import annotations

import json
from dataclasses import asdict

from tools import evidence_graph_rag_regression_cli as cli
from tools.evidence_graph_rag_benchmark import (
    GraphRAGBenchmarkCase,
    GraphRAGBenchmarkFixture,
    GraphRAGBenchmarkRun,
    GraphRAGSelectionObservation,
    run_graph_rag_benchmark,
)
from tools.evidence_graph_rag_evaluation import GraphNodeLocator, GraphRAGGoldCase

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
QUERY = "c" * 64
NODE = "1" * 64


def report(selected=True, work=10):
    gold = GraphRAGGoldCase(
        query_id="q",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        relevant_nodes=(GraphNodeLocator("doc-a", 1, NODE),),
        required_edge_ids=(),
        should_abstain=False,
    )
    runs = []
    for index in range(3):
        observation = GraphRAGSelectionObservation(
            graph_set_id=SET_ID,
            graph_set_digest=SET_DIGEST,
            query_digest=QUERY,
            selection_digest=f"{index + 1:x}" * 64,
            selected_nodes=(GraphNodeLocator("doc-a", 1, NODE),) if selected else (),
            traversal_edge_ids=(),
            expanded_lineage_valid=(),
            abstained=not selected,
            evidence_count=1 if selected else 0,
            traversal_count=0,
            estimated_work_units=work,
        )
        runs.append(
            GraphRAGBenchmarkRun(
                f"run-{index}", index, (GraphRAGBenchmarkCase(gold, observation),)
            )
        )
    return run_graph_rag_benchmark(GraphRAGBenchmarkFixture("bench", tuple(runs)))


def payload(value):
    result = asdict(value)
    result["report_digest"] = value.report_digest
    result["contains_raw_query"] = False
    result["contains_evidence_text"] = False
    return result


def parse(capsys):
    value = capsys.readouterr()
    return (
        json.loads(value.out) if value.out else None,
        json.loads(value.err) if value.err else None,
    )


def test_eligible_compare_writes_privacy_safe_report(tmp_path, capsys):
    baseline = report()
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(payload(baseline)))
    candidate_path.write_text(json.dumps(payload(baseline)))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"min_cases_per_run": 1}))
    output_path = tmp_path / "regression.json"
    assert (
        cli.main(
            [
                "compare",
                str(base_path),
                str(candidate_path),
                "--policy-file",
                str(policy_path),
                "--output-file",
                str(output_path),
            ]
        )
        == 0
    )
    output, error = parse(capsys)
    assert error is None and output["decision"] == "eligible"
    assert output["runtime_policy_changed"] is False
    assert output["contains_raw_query"] is False
    assert json.loads(output_path.read_text())["report_digest"] == output["report_digest"]


def test_blocked_compare_returns_one(tmp_path, capsys):
    baseline = report(selected=True)
    candidate = report(selected=False)
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(payload(baseline)))
    candidate_path.write_text(json.dumps(payload(candidate)))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"min_cases_per_run": 1}))
    assert (
        cli.main(
            [
                "compare",
                str(base_path),
                str(candidate_path),
                "--policy-file",
                str(policy_path),
            ]
        )
        == 1
    )
    output, error = parse(capsys)
    assert error is None and output["decision"] == "blocked"
    assert "macro_node_f1_below_floor" in output["reason_codes"]


def test_forged_digest_and_unknown_policy_fail_closed(tmp_path, capsys):
    baseline = report()
    value = payload(baseline)
    value["report_digest"] = "f" * 64
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    base_path.write_text(json.dumps(value))
    candidate_path.write_text(json.dumps(payload(baseline)))
    assert cli.main(["compare", str(base_path), str(candidate_path)]) == 2
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    base_path.write_text(json.dumps(payload(baseline)))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"invented": 1}))
    assert (
        cli.main(
            [
                "compare",
                str(base_path),
                str(candidate_path),
                "--policy-file",
                str(policy),
            ]
        )
        == 2
    )
