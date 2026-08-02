from __future__ import annotations

import sqlite3

import pytest

from tools.evidence_graph_rag import GraphEvidenceItem, GraphEvidenceSelection
from tools.evidence_graph_rag_evaluation import GraphNodeLocator, GraphRAGGoldCase
from tools.evidence_graph_rag_live_benchmark import GraphRAGLiveBenchmarkPlan
from tools.evidence_graph_rag_run_store import (
    GraphRAGBenchmarkRunStore,
    execute_resumable_live_graph_rag_benchmark,
)

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
AUTHORITY = "c" * 64
QUERY_TEXT = "alpha query"
QUERY = __import__("hashlib").sha256(QUERY_TEXT.encode()).hexdigest()
NODE = "1" * 64
GRAPH = "2" * 64
PROVENANCE = "3" * 64


def plan(config=None):
    gold = GraphRAGGoldCase(
        query_id="q1",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        relevant_nodes=(GraphNodeLocator("doc-a", 1, NODE),),
        required_edge_ids=(),
        should_abstain=False,
    )
    return GraphRAGLiveBenchmarkPlan(
        benchmark_id="live-v1",
        run_seeds=(1, 2, 3),
        gold_cases=(gold,),
        selector_config=config or {"max_total_items": 10},
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


def test_resumable_execution_reuses_completed_runs_without_resolving_queries(tmp_path):
    store = GraphRAGBenchmarkRunStore(tmp_path / "runs.sqlite3")
    calls = []

    def resolver(query_id):
        calls.append(("query", query_id))
        return QUERY_TEXT

    def runner(**kwargs):
        calls.append(("selection", kwargs["seed"]))
        return selection(kwargs["query"])

    first = execute_resumable_live_graph_rag_benchmark(
        plan(), query_resolver=resolver, selection_runner=runner, store=store, now=lambda: 1.0
    )
    assert first.executed_run_count == 3 and first.reused_run_count == 0
    assert first.result.report.aggregate.macro_node_f1 == 1.0
    calls.clear()
    second = execute_resumable_live_graph_rag_benchmark(
        plan(),
        query_resolver=lambda query_id: (_ for _ in ()).throw(AssertionError("should not resolve")),
        selection_runner=lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run")),
        store=store,
        now=lambda: 2.0,
    )
    assert second.executed_run_count == 0 and second.reused_run_count == 3
    assert calls == []
    assert second.result.report.report_digest == first.result.report.report_digest


def test_partial_resume_executes_only_missing_runs(tmp_path):
    store = GraphRAGBenchmarkRunStore(tmp_path / "runs.sqlite3")
    value = plan()

    def failing_runner(**kwargs):
        if kwargs["seed"] == 2:
            raise RuntimeError("interrupted")
        return selection(kwargs["query"])

    with pytest.raises(RuntimeError, match="interrupted"):
        execute_resumable_live_graph_rag_benchmark(
            value,
            query_resolver=lambda query_id: QUERY_TEXT,
            selection_runner=failing_runner,
            store=store,
            now=lambda: 1.0,
        )
    assert len(store.list_plan(value.plan_fingerprint)) == 1
    observed = []

    def resolver(query_id):
        observed.append(query_id)
        return QUERY_TEXT

    resumed = execute_resumable_live_graph_rag_benchmark(
        value,
        query_resolver=resolver,
        selection_runner=lambda **kwargs: selection(kwargs["query"]),
        store=store,
        now=lambda: 2.0,
    )
    assert resumed.executed_run_count == 2
    assert resumed.reused_run_count == 1
    assert observed == ["q1", "q1"]
    assert resumed.result.report.run_count == 3


def test_selector_configuration_changes_plan_not_benchmark_contract(tmp_path):
    first = plan({"max_total_items": 10})
    second = plan({"max_total_items": 20})
    assert first.plan_fingerprint != second.plan_fingerprint
    store = GraphRAGBenchmarkRunStore(tmp_path / "runs.sqlite3")
    a = execute_resumable_live_graph_rag_benchmark(
        first,
        query_resolver=lambda query_id: QUERY_TEXT,
        selection_runner=lambda **kwargs: selection(kwargs["query"]),
        store=store,
    )
    b = execute_resumable_live_graph_rag_benchmark(
        second,
        query_resolver=lambda query_id: QUERY_TEXT,
        selection_runner=lambda **kwargs: selection(kwargs["query"]),
        store=store,
    )
    assert a.result.report.benchmark_fingerprint == b.result.report.benchmark_fingerprint
    assert len(store.list_plan(first.plan_fingerprint)) == 3
    assert len(store.list_plan(second.plan_fingerprint)) == 3


def test_stored_payload_contains_no_query_or_evidence_text(tmp_path):
    path = tmp_path / "runs.sqlite3"
    store = GraphRAGBenchmarkRunStore(path)
    execute_resumable_live_graph_rag_benchmark(
        plan(),
        query_resolver=lambda query_id: QUERY_TEXT,
        selection_runner=lambda **kwargs: selection(kwargs["query"]),
        store=store,
    )
    with sqlite3.connect(path) as connection:
        payloads = " ".join(row[0] for row in connection.execute("SELECT payload_json FROM graph_rag_runs"))
    assert QUERY_TEXT not in payloads
    assert "private evidence text" not in payloads
    assert "raw_query" not in payloads


def test_tamper_and_database_replacement_are_detected(tmp_path):
    path = tmp_path / "runs.sqlite3"
    store = GraphRAGBenchmarkRunStore(path)
    result = execute_resumable_live_graph_rag_benchmark(
        plan(),
        query_resolver=lambda query_id: QUERY_TEXT,
        selection_runner=lambda **kwargs: selection(kwargs["query"]),
        store=store,
    )
    run = store.list_plan(result.result.plan_fingerprint)[0]
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE graph_rag_runs SET payload_json=? WHERE plan_fingerprint=? AND run_id=?",
            ('{"run_id":"bad"}', run.plan_fingerprint, run.run_id),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get(run.plan_fingerprint, run.run_id)
    guarded_path = tmp_path / "guarded.sqlite3"
    guarded = GraphRAGBenchmarkRunStore(guarded_path)
    guarded_path.rename(tmp_path / "old.sqlite3")
    guarded_path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.list_plan("f" * 64)


def test_plan_removal_requires_exact_confirmation(tmp_path):
    store = GraphRAGBenchmarkRunStore(tmp_path / "runs.sqlite3")
    result = execute_resumable_live_graph_rag_benchmark(
        plan(),
        query_resolver=lambda query_id: QUERY_TEXT,
        selection_runner=lambda **kwargs: selection(kwargs["query"]),
        store=store,
    )
    fingerprint = result.result.plan_fingerprint
    with pytest.raises(ValueError, match="exactly match"):
        store.remove_plan(fingerprint, confirm_plan_fingerprint="f" * 64)
    assert store.remove_plan(fingerprint, confirm_plan_fingerprint=fingerprint) is True
    assert store.list_plan(fingerprint) == ()
