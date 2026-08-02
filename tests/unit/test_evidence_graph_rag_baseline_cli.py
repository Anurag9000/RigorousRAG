from __future__ import annotations

import json
from dataclasses import asdict

from tools import evidence_graph_rag_baseline_cli as cli
from tools.evidence_graph_rag_baseline import GraphRAGBaselineStore
from tools.evidence_graph_rag_benchmark import (
    GraphRAGBenchmarkCase,
    GraphRAGBenchmarkFixture,
    GraphRAGBenchmarkRun,
    GraphRAGSelectionObservation,
    run_graph_rag_benchmark,
)
from tools.evidence_graph_rag_evaluation import GraphNodeLocator, GraphRAGGoldCase
from tools.evidence_graph_rag_regression import (
    GraphRAGRegressionPolicy,
    evaluate_graph_rag_regression,
)

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
QUERY = "c" * 64
NODE = "1" * 64


def report(*, selected=True, work=10):
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
            selected_nodes=(
                (GraphNodeLocator("doc-a", 1, NODE),) if selected else ()
            ),
            traversal_edge_ids=(),
            expanded_lineage_valid=(),
            abstained=not selected,
            evidence_count=1 if selected else 0,
            traversal_count=0,
            estimated_work_units=work,
        )
        runs.append(
            GraphRAGBenchmarkRun(
                f"run-{index}",
                index,
                (GraphRAGBenchmarkCase(gold, observation),),
            )
        )
    return run_graph_rag_benchmark(
        GraphRAGBenchmarkFixture("bench", tuple(runs))
    )


def report_payload(value):
    result = asdict(value)
    result["report_digest"] = value.report_digest
    result["contains_raw_query"] = False
    result["contains_evidence_text"] = False
    return result


def regression_payload(value):
    result = asdict(value)
    result["report_digest"] = value.report_digest
    result["paired_interval_method"] = (
        "normal_approximation_over_run_deltas"
    )
    result["contains_raw_query"] = False
    result["contains_evidence_text"] = False
    result["runtime_policy_changed"] = False
    return result


def parse(capsys):
    captured = capsys.readouterr()
    return (
        json.loads(captured.out) if captured.out else None,
        json.loads(captured.err) if captured.err else None,
    )


def install(tmp_path, monkeypatch):
    store = GraphRAGBaselineStore(tmp_path / "baselines.sqlite3")
    monkeypatch.setattr(
        cli, "get_graph_rag_baseline_store", lambda: store
    )
    return store


def test_initialize_status_and_history_are_text_free(
    tmp_path, monkeypatch, capsys
):
    install(tmp_path, monkeypatch)
    value = report()
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(report_payload(value)))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "policy_id": "policy-v1",
                "min_cases_per_run": 1,
            }
        )
    )
    assert (
        cli.main(
            [
                "initialize",
                str(candidate_path),
                "--policy-file",
                str(policy_path),
                "--expect-no-current",
            ]
        )
        == 0
    )
    initialized, error = parse(capsys)
    assert error is None
    assert initialized["runtime_policy_changed"] is False
    assert initialized["contains_evidence_text"] is False
    baseline_digest = initialized["baseline_digest"]
    assert (
        cli.main(
            [
                "status",
                value.benchmark_fingerprint,
                "--policy-id",
                "policy-v1",
            ]
        )
        == 0
    )
    status, _error = parse(capsys)
    assert status["baseline_digest"] == baseline_digest
    assert status["mutation_performed"] is False
    assert (
        cli.main(
            [
                "history",
                value.benchmark_fingerprint,
                "--policy-id",
                "policy-v1",
            ]
        )
        == 0
    )
    history, _error = parse(capsys)
    assert history["count"] == 1


def test_promote_requires_exact_eligible_regression(
    tmp_path, monkeypatch, capsys
):
    install(tmp_path, monkeypatch)
    baseline = report()
    candidate = report(work=11)
    policy = GraphRAGRegressionPolicy(
        policy_id="policy-v1",
        min_cases_per_run=1,
    )
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "policy.json"
    base_path.write_text(json.dumps(report_payload(baseline)))
    candidate_path.write_text(json.dumps(report_payload(candidate)))
    policy_path.write_text(
        json.dumps(
            {
                "policy_id": "policy-v1",
                "min_cases_per_run": 1,
            }
        )
    )
    assert (
        cli.main(
            [
                "initialize",
                str(base_path),
                "--policy-file",
                str(policy_path),
                "--expect-no-current",
            ]
        )
        == 0
    )
    initial, _error = parse(capsys)
    regression = evaluate_graph_rag_regression(
        baseline, candidate, policy
    )
    regression_path = tmp_path / "regression.json"
    regression_path.write_text(json.dumps(regression_payload(regression)))
    assert (
        cli.main(
            [
                "promote",
                str(candidate_path),
                str(regression_path),
                "--policy-file",
                str(policy_path),
                "--expected-current-baseline-digest",
                initial["baseline_digest"],
            ]
        )
        == 0
    )
    promoted, error = parse(capsys)
    assert error is None
    assert (
        promoted["previous_baseline_digest"]
        == initial["baseline_digest"]
    )
    assert (
        promoted["activation_regression_digest"]
        == regression.report_digest
    )


def test_blocked_regression_and_missing_status_are_bounded(
    tmp_path, monkeypatch, capsys
):
    install(tmp_path, monkeypatch)
    baseline = report()
    candidate = report(selected=False)
    policy = GraphRAGRegressionPolicy(
        policy_id="policy-v1",
        min_cases_per_run=1,
    )
    base_path = tmp_path / "base.json"
    candidate_path = tmp_path / "candidate.json"
    policy_path = tmp_path / "policy.json"
    base_path.write_text(json.dumps(report_payload(baseline)))
    candidate_path.write_text(json.dumps(report_payload(candidate)))
    policy_path.write_text(
        json.dumps(
            {
                "policy_id": "policy-v1",
                "min_cases_per_run": 1,
            }
        )
    )
    cli.main(
        [
            "initialize",
            str(base_path),
            "--policy-file",
            str(policy_path),
            "--expect-no-current",
        ]
    )
    initial, _error = parse(capsys)
    blocked = evaluate_graph_rag_regression(
        baseline, candidate, policy
    )
    regression_path = tmp_path / "blocked.json"
    regression_path.write_text(json.dumps(regression_payload(blocked)))
    assert (
        cli.main(
            [
                "promote",
                str(candidate_path),
                str(regression_path),
                "--policy-file",
                str(policy_path),
                "--expected-current-baseline-digest",
                initial["baseline_digest"],
            ]
        )
        == 2
    )
    _output, error = parse(capsys)
    assert error == {"error": "invalid_or_unavailable"}
    assert (
        cli.main(
            [
                "status",
                "f" * 64,
                "--policy-id",
                "policy-v1",
            ]
        )
        == 1
    )
    _output, error = parse(capsys)
    assert error == {"error": "not_found"}
