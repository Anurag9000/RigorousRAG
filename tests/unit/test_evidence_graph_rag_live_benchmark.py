from __future__ import annotations

from dataclasses import asdict

import pytest

from tools.evidence_graph_rag import GraphEvidenceItem, GraphEvidenceSelection
from tools.evidence_graph_rag_evaluation import GraphNodeLocator, GraphRAGGoldCase
from tools.evidence_graph_rag_live_benchmark import (
    GraphRAGLiveBenchmarkPlan,
    execute_authoritative_graph_rag_benchmark,
    execute_live_graph_rag_benchmark,
    observation_from_selection,
    plan_from_mapping,
)

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
AUTHORITY = "c" * 64
QUERY_TEXT = "what is alpha"
QUERY = __import__("hashlib").sha256(QUERY_TEXT.encode()).hexdigest()
NODE = "1" * 64
GRAPH = "2" * 64
PROVENANCE = "3" * 64


def gold():
    return GraphRAGGoldCase(
        query_id="q1",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        relevant_nodes=(GraphNodeLocator("doc-a", 1, NODE),),
        required_edge_ids=(),
        should_abstain=False,
    )


def selection(query):
    item = GraphEvidenceItem(
        owner_id="alice",
        doc_id="doc-a",
        generation=1,
        graph_digest=GRAPH,
        node_id=NODE,
        node_type="claim",
        label="Claim",
        text="private evidence text",
        page_number=1,
        section="Results",
        score=1.0,
        matched_terms=("alpha",),
        provenance_digest=PROVENANCE,
        origin="lexical",
        lineage_step_digests=(),
    )
    return GraphEvidenceSelection(
        owner_id="alice",
        graph_set_key="review",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        authority_digest=AUTHORITY,
        query_digest=__import__("hashlib").sha256(query.encode()).hexdigest(),
        items=(item,),
        traversals=(),
        lexical_seed_count=1,
        expanded_count=0,
        estimated_work_units=5,
        abstained=False,
    )


def plan():
    return GraphRAGLiveBenchmarkPlan(
        benchmark_id="live-v1",
        run_seeds=(1, 2, 3),
        gold_cases=(gold(),),
        selector_config={"max_total_items": 10},
    )


def test_live_execution_reduces_text_to_observations():
    observed = []

    def runner(**kwargs):
        observed.append((kwargs["query_id"], kwargs["seed"], kwargs["selector_config"]))
        return selection(kwargs["query"])

    result = execute_live_graph_rag_benchmark(
        plan(), query_resolver=lambda query_id: QUERY_TEXT, selection_runner=runner
    )
    assert result.report.run_count == 3
    assert result.report.aggregate.macro_node_f1 == 1.0
    assert result.query_text_persisted is False
    assert result.evidence_text_persisted is False
    rendered = str(asdict(result)).lower()
    assert QUERY_TEXT not in rendered
    assert "private evidence text" not in rendered
    assert observed[0] == ("q1", 1, {"max_total_items": 10})


def test_query_digest_mismatch_refuses_runner_call():
    called = False

    def runner(**kwargs):
        nonlocal called
        called = True
        return selection(kwargs["query"])

    with pytest.raises(ValueError, match="digest differs"):
        execute_live_graph_rag_benchmark(
            plan(), query_resolver=lambda query_id: "wrong", selection_runner=runner
        )
    assert called is False


def test_selection_identity_mismatch_is_refused():
    wrong = selection(QUERY_TEXT)
    wrong = __import__("dataclasses").replace(wrong, graph_set_id="f" * 64)
    with pytest.raises(ValueError, match="identities differ"):
        execute_live_graph_rag_benchmark(
            plan(), query_resolver=lambda query_id: QUERY_TEXT, selection_runner=lambda **kwargs: wrong
        )


def test_observation_conversion_contains_no_text():
    value = observation_from_selection(selection(QUERY_TEXT))
    rendered = str(asdict(value)).lower()
    assert "private evidence text" not in rendered
    assert value.selected_nodes == (GraphNodeLocator("doc-a", 1, NODE),)
    assert value.evidence_count == 1


def test_plan_mapping_is_strict_and_selector_fields_are_closed():
    value = {
        "benchmark_id": "live-v1",
        "run_seeds": [1, 2],
        "gold_cases": [
            {
                "query_id": "q1",
                "graph_set_id": SET_ID,
                "graph_set_digest": SET_DIGEST,
                "query_digest": QUERY,
                "relevant_nodes": [
                    {"doc_id": "doc-a", "generation": 1, "node_id": NODE}
                ],
                "required_edge_ids": [],
                "should_abstain": False,
            }
        ],
        "selector_config": {"node_types": ["claim"], "max_total_items": 10},
        "schema_version": 1,
    }
    loaded = plan_from_mapping(value)
    assert loaded.selector_config["node_types"] == ("claim",)
    with pytest.raises(ValueError, match="schema"):
        plan_from_mapping({**value, "raw_query": QUERY_TEXT})
    value["selector_config"] = {"invented": 1}
    with pytest.raises(ValueError, match="unsupported"):
        plan_from_mapping(value)


def test_authoritative_execution_uses_current_set_selector(monkeypatch):
    observed = []

    def selector(**kwargs):
        observed.append(kwargs)
        return selection(kwargs["query"])

    monkeypatch.setattr(
        "tools.evidence_graph_rag_live_benchmark.select_current_graph_set_evidence",
        selector,
    )
    result = execute_authoritative_graph_rag_benchmark(
        plan(),
        owner_id="alice",
        graph_set_key="review",
        query_resolver=lambda query_id: QUERY_TEXT,
        set_store=object(),
        generations=object(),
        graphs=object(),
    )
    assert result.report.run_count == 3
    assert observed[0]["owner_id"] == "alice"
    assert observed[0]["graph_set_key"] == "review"
    assert observed[0]["max_total_items"] == 10
