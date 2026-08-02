from __future__ import annotations

from dataclasses import asdict

import pytest

from tools.evidence_graph_rag_benchmark import (
    GraphRAGBenchmarkCase,
    GraphRAGBenchmarkFixture,
    GraphRAGBenchmarkRun,
    GraphRAGSelectionObservation,
    fixture_from_mapping,
    run_graph_rag_benchmark,
)
from tools.evidence_graph_rag_evaluation import GraphNodeLocator, GraphRAGGoldCase

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
QUERY = "c" * 64
SELECTION = "d" * 64
NODE_A = "1" * 64
NODE_B = "2" * 64
EDGE = "3" * 64


def gold(query_id="q1"):
    return GraphRAGGoldCase(
        query_id=query_id,
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        relevant_nodes=(
            GraphNodeLocator("doc-a", 1, NODE_A),
            GraphNodeLocator("doc-b", 1, NODE_B),
        ),
        required_edge_ids=(EDGE,),
        should_abstain=False,
    )


def observation(*, selected=True, work=20):
    return GraphRAGSelectionObservation(
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        selection_digest=SELECTION,
        selected_nodes=(
            (GraphNodeLocator("doc-a", 1, NODE_A), GraphNodeLocator("doc-b", 1, NODE_B))
            if selected
            else ()
        ),
        traversal_edge_ids=(EDGE,) if selected else (),
        expanded_lineage_valid=(True,) if selected else (),
        abstained=not selected,
        evidence_count=2 if selected else 0,
        traversal_count=1 if selected else 0,
        estimated_work_units=work,
    )


def fixture():
    case = GraphRAGBenchmarkCase(gold=gold(), observation=observation())
    return GraphRAGBenchmarkFixture(
        benchmark_id="graph-rag-v1",
        runs=(
            GraphRAGBenchmarkRun("run-1", 1, (case,)),
            GraphRAGBenchmarkRun("run-2", 2, (case,)),
        ),
    )


def test_report_is_reproducible_query_digest_only():
    report = run_graph_rag_benchmark(fixture())
    assert report.run_count == 2
    assert report.seed_count == 2
    assert report.case_count_per_run == 1
    assert report.aggregate.macro_node_f1 == 1.0
    assert report.aggregate.complete_required_path_rate == 1.0
    assert len(report.benchmark_fingerprint) == 64
    assert len(report.report_digest) == 64
    rendered = str(asdict(report)).lower()
    assert "raw_query" not in rendered
    assert "private text" not in rendered


def test_fingerprint_tracks_contract_not_selection_results():
    base = fixture()
    changed_case = GraphRAGBenchmarkCase(gold=gold(), observation=observation(selected=False))
    changed = GraphRAGBenchmarkFixture(
        benchmark_id="graph-rag-v1",
        runs=(
            GraphRAGBenchmarkRun("run-1", 1, (changed_case,)),
            GraphRAGBenchmarkRun("run-2", 2, (changed_case,)),
        ),
    )
    assert base.benchmark_fingerprint == changed.benchmark_fingerprint
    assert run_graph_rag_benchmark(base).report_digest != run_graph_rag_benchmark(changed).report_digest


def test_all_runs_require_the_same_ordered_gold_contract():
    first = GraphRAGBenchmarkCase(gold=gold("q1"), observation=observation())
    second = GraphRAGBenchmarkCase(gold=gold("q2"), observation=observation())
    with pytest.raises(ValueError, match="same ordered"):
        GraphRAGBenchmarkFixture(
            benchmark_id="bad",
            runs=(
                GraphRAGBenchmarkRun("run-1", 1, (first,)),
                GraphRAGBenchmarkRun("run-2", 2, (second,)),
            ),
        )


def test_observation_counts_and_abstention_are_exact():
    with pytest.raises(ValueError, match="evidence_count"):
        GraphRAGSelectionObservation(
            graph_set_id=SET_ID,
            graph_set_digest=SET_DIGEST,
            query_digest=QUERY,
            selection_digest=SELECTION,
            selected_nodes=(GraphNodeLocator("doc-a", 1, NODE_A),),
            traversal_edge_ids=(),
            expanded_lineage_valid=(),
            abstained=False,
            evidence_count=2,
            traversal_count=0,
            estimated_work_units=1,
        )
    with pytest.raises(ValueError, match="exactly reflect"):
        GraphRAGSelectionObservation(
            graph_set_id=SET_ID,
            graph_set_digest=SET_DIGEST,
            query_digest=QUERY,
            selection_digest=SELECTION,
            selected_nodes=(),
            traversal_edge_ids=(),
            expanded_lineage_valid=(),
            abstained=False,
            evidence_count=0,
            traversal_count=0,
            estimated_work_units=1,
        )


def test_strict_mapping_rejects_raw_query_and_unknown_fields():
    value = {
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
                                {"doc_id": "doc-a", "generation": 1, "node_id": NODE_A}
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
                                {"doc_id": "doc-a", "generation": 1, "node_id": NODE_A}
                            ],
                            "traversal_edge_ids": [],
                            "expanded_lineage_valid": [],
                            "abstained": False,
                            "evidence_count": 1,
                            "traversal_count": 0,
                            "estimated_work_units": 1,
                        },
                    }
                ],
            }
        ],
    }
    assert fixture_from_mapping(value).benchmark_id == "graph-rag-v1"
    with pytest.raises(ValueError, match="schema"):
        fixture_from_mapping({**value, "raw_query": "private"})
    value["runs"][0]["cases"][0]["observation"]["node_text"] = "private"
    with pytest.raises(ValueError, match="schema"):
        fixture_from_mapping(value)


def test_report_aggregates_partial_runs_macro_style():
    good = GraphRAGBenchmarkCase(gold=gold(), observation=observation())
    bad = GraphRAGBenchmarkCase(gold=gold(), observation=observation(selected=False, work=5))
    value = GraphRAGBenchmarkFixture(
        benchmark_id="mixed",
        runs=(
            GraphRAGBenchmarkRun("good", 1, (good,)),
            GraphRAGBenchmarkRun("bad", 2, (bad,)),
        ),
    )
    report = run_graph_rag_benchmark(value)
    assert report.aggregate.macro_node_recall == 0.5
    assert report.aggregate.complete_required_path_rate == 0.5
    assert report.aggregate.mean_estimated_work_units == 12.5
