from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

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
    policy_from_mapping,
    report_from_mapping,
)

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
QUERY = "c" * 64
NODE = "1" * 64


def report(*, selected=True, work=10, runs=3, cases=10):
    gold = GraphRAGGoldCase(
        query_id="q",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        relevant_nodes=(GraphNodeLocator("doc-a", 1, NODE),),
        required_edge_ids=(),
        should_abstain=False,
    )
    values = []
    for run_index in range(runs):
        run_cases = []
        for case_index in range(cases):
            case_gold = replace(gold, query_id=f"q-{case_index}")
            observation = GraphRAGSelectionObservation(
                graph_set_id=SET_ID,
                graph_set_digest=SET_DIGEST,
                query_digest=QUERY,
                selection_digest=f"{run_index + 1:x}" * 64,
                selected_nodes=(GraphNodeLocator("doc-a", 1, NODE),) if selected else (),
                traversal_edge_ids=(),
                expanded_lineage_valid=(),
                abstained=not selected,
                evidence_count=1 if selected else 0,
                traversal_count=0,
                estimated_work_units=work,
            )
            run_cases.append(GraphRAGBenchmarkCase(case_gold, observation))
        values.append(GraphRAGBenchmarkRun(f"run-{run_index}", run_index, tuple(run_cases)))
    return run_graph_rag_benchmark(GraphRAGBenchmarkFixture("bench", tuple(values)))


def test_equal_reports_are_eligible_under_default_policy():
    baseline = report()
    result = evaluate_graph_rag_regression(baseline, baseline)
    assert result.decision == "eligible"
    assert result.reason_codes == ()
    assert result.work_ratio == 1.0
    assert all(value.lower == 0 for value in result.paired_intervals.values())
    assert len(result.report_digest) == 64


def test_quality_floor_and_noninferiority_block_regression():
    baseline = report(selected=True)
    candidate = report(selected=False)
    result = evaluate_graph_rag_regression(baseline, candidate)
    assert result.decision == "blocked"
    assert "macro_node_f1_below_floor" in result.reason_codes
    assert "macro_node_f1_noninferiority_failed" in result.reason_codes
    assert "abstention_accuracy_below_floor" in result.reason_codes


def test_work_ratio_and_minimum_dimensions_are_enforced():
    baseline = report(work=10, runs=1, cases=1)
    candidate = report(work=30, runs=1, cases=1)
    policy = GraphRAGRegressionPolicy(min_run_count=3, min_seed_count=3, min_cases_per_run=5)
    result = evaluate_graph_rag_regression(baseline, candidate, policy)
    assert "run_count_below_minimum" in result.reason_codes
    assert "seed_count_below_minimum" in result.reason_codes
    assert "case_count_below_minimum" in result.reason_codes
    assert "estimated_work_ratio_exceeds_limit" in result.reason_codes


def test_fingerprint_and_run_contract_mismatch_are_refused():
    baseline = report()
    candidate = replace(baseline, benchmark_fingerprint="f" * 64)
    with pytest.raises(ValueError, match="fingerprints differ"):
        evaluate_graph_rag_regression(baseline, candidate)
    changed_run = replace(
        baseline.run_reports[0], run_contract_digest="e" * 64
    )
    candidate = replace(
        baseline,
        run_reports=(changed_run,) + baseline.run_reports[1:],
    )
    with pytest.raises(ValueError, match="run contracts differ"):
        evaluate_graph_rag_regression(baseline, candidate)


def test_report_and_policy_mapping_are_strict():
    baseline = report()
    value = json.loads(json.dumps(asdict(baseline)))
    value["report_digest"] = baseline.report_digest
    value["contains_raw_query"] = False
    value["contains_evidence_text"] = False
    assert report_from_mapping(value).report_digest == baseline.report_digest
    with pytest.raises(ValueError, match="schema"):
        report_from_mapping({**value, "raw_query": "private"})
    value["report_digest"] = "f" * 64
    with pytest.raises(ValueError, match="digest"):
        report_from_mapping(value)
    assert policy_from_mapping({"max_work_ratio": 2.0}).max_work_ratio == 2.0
    with pytest.raises(ValueError, match="unknown"):
        policy_from_mapping({"invented": 1})
