import sqlite3

import pytest

from tools.multihop_budget import allocate_multihop_budget
from tools.multihop_retrieval import run_multihop_retrieval
from tools.multihop_trace_store import MultiHopTraceStore
from tools.query_decomposition import build_decomposition_plan


def successful():
    plan = build_decomposition_plan(
        "Compare A and B.",
        proposed_subquestions=[
            {"question_id": "a", "text": "Find A."},
            {"question_id": "b", "text": "Find B."},
            {
                "question_id": "compare",
                "text": "Compare A and B.",
                "depends_on": ["a", "b"],
                "relation": "compare",
            },
        ],
    )

    def search(question, dependencies):
        return [{
            "source_id": f"{question.question_id}-source",
            "doc_id": "shared",
            "text": "private evidence text",
            "score": 0.9,
        }]

    retrieval = run_multihop_retrieval(plan, search=search)
    budget = allocate_multihop_budget(
        plan, top_k=5, total_limit=300, per_hop_limit=100
    )
    return plan, retrieval, budget


def test_store_records_query_free_summary_hops_and_aggregates(tmp_path):
    path = tmp_path / "traces.sqlite3"
    store = MultiHopTraceStore(path)
    plan, retrieval, budget = successful()
    run_id = store.record_result(
        owner_id="alice",
        plan=plan,
        retrieval=retrieval,
        budget=budget,
        used_model=False,
        planner_quality=0.8,
        run_id="run-1",
        started_at=1.0,
        completed_at=2.0,
    )
    assert run_id == "run-1"
    record = store.get_run(owner_id="alice", run_id=run_id)
    assert record is not None
    assert record.summary.plan_fingerprint == plan.fingerprint
    assert record.summary.evidence_count == 3
    assert record.summary.join_count == 1
    assert record.summary.abstain is False
    assert len(record.hops) == 3
    raw = path.read_bytes()
    assert b"Compare A and B" not in raw
    assert b"private evidence text" not in raw
    aggregate = store.aggregate(owner_id="alice")
    assert aggregate.run_count == 1
    assert aggregate.abstention_count == 0
    assert aggregate.hop_statuses == (("success", 3),)


def test_store_is_owner_scoped_and_idempotent(tmp_path):
    store = MultiHopTraceStore(tmp_path / "traces.sqlite3")
    plan, retrieval, budget = successful()
    kwargs = dict(
        owner_id="alice",
        plan=plan,
        retrieval=retrieval,
        budget=budget,
        used_model=False,
        planner_quality=0.8,
        run_id="stable",
        started_at=1.0,
        completed_at=2.0,
    )
    assert store.record_result(**kwargs) == "stable"
    assert store.record_result(**kwargs) == "stable"
    assert store.get_run(owner_id="bob", run_id="stable") is None
    with pytest.raises(ValueError, match="different multi-hop trace"):
        store.record_result(**{**kwargs, "planner_quality": 0.7})


def test_store_records_only_generic_errors_and_timeouts(tmp_path):
    store = MultiHopTraceStore(tmp_path / "traces.sqlite3")
    plan = build_decomposition_plan("Question")

    def fail(*args):
        raise RuntimeError("secret provider message")

    retrieval = run_multihop_retrieval(plan, search=fail)
    budget = allocate_multihop_budget(
        plan, top_k=5, total_limit=100, per_hop_limit=100
    )
    store.record_result(
        owner_id="alice",
        plan=plan,
        retrieval=retrieval,
        budget=budget,
        used_model=True,
        planner_quality=0.4,
        run_id="failed",
        started_at=1.0,
        completed_at=2.0,
    )
    record = store.get_run(owner_id="alice", run_id="failed")
    assert record is not None
    assert record.hops[0].error_type == "RuntimeError"
    assert b"secret provider message" not in store.path.read_bytes()
    aggregate = store.aggregate(owner_id="alice")
    assert aggregate.error_run_count == 1
    assert aggregate.model_plan_count == 1


def test_store_detects_identity_replacement_and_corrupt_hop_count(tmp_path):
    path = tmp_path / "traces.sqlite3"
    store = MultiHopTraceStore(path)
    plan, retrieval, budget = successful()
    store.record_result(
        owner_id="alice", plan=plan, retrieval=retrieval, budget=budget,
        used_model=False, planner_quality=0.8, run_id="run",
        started_at=1.0, completed_at=2.0,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM multihop_hops WHERE run_id='run' AND sequence=0"
        )
        connection.commit()
    with pytest.raises(RuntimeError, match="corrupt"):
        store.get_run(owner_id="alice", run_id="run")

    path.rename(tmp_path / "old.sqlite3")
    path.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="replaced"):
        store.list_runs(owner_id="alice")
    assert store.ping() is False


def test_prune_is_owner_scoped(tmp_path):
    store = MultiHopTraceStore(tmp_path / "traces.sqlite3")
    plan, retrieval, budget = successful()
    for index in range(3):
        store.record_result(
            owner_id="alice", plan=plan, retrieval=retrieval, budget=budget,
            used_model=False, planner_quality=0.8, run_id=f"a-{index}",
            started_at=float(index), completed_at=float(index+1),
        )
    store.record_result(
        owner_id="bob", plan=plan, retrieval=retrieval, budget=budget,
        used_model=False, planner_quality=0.8, run_id="bob",
        started_at=1.0, completed_at=2.0,
    )
    assert store.prune_owner(owner_id="alice", retain_latest=1) == 2
    assert len(store.list_runs(owner_id="alice")) == 1
    assert len(store.list_runs(owner_id="bob")) == 1


from tools.multihop_trace_runtime import (
    clear_multihop_trace_store_cache,
    get_multihop_trace_store,
)


def test_runtime_is_disabled_without_configuration(monkeypatch):
    clear_multihop_trace_store_cache()
    monkeypatch.delenv("MULTIHOP_TRACE_DB_PATH", raising=False)
    assert get_multihop_trace_store() is None


def test_runtime_is_path_keyed(monkeypatch, tmp_path):
    clear_multihop_trace_store_cache()
    path = tmp_path / "traces.sqlite3"
    monkeypatch.setenv("MULTIHOP_TRACE_DB_PATH", str(path))
    first = get_multihop_trace_store()
    assert first is get_multihop_trace_store()
    clear_multihop_trace_store_cache()
    assert first is not get_multihop_trace_store()


def test_runtime_rejects_padded_configuration(monkeypatch):
    clear_multihop_trace_store_cache()
    monkeypatch.setenv("MULTIHOP_TRACE_DB_PATH", " padded.sqlite3 ")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        get_multihop_trace_store()


import tools.multihop_rag_tool as tool


def test_public_multihop_api_persists_optional_trace(monkeypatch, tmp_path):
    class Adaptive:
        evidence = (
            {
                "source_id": "source",
                "doc_id": "doc",
                "text": "evidence",
                "score": 0.9,
            },
        )

    monkeypatch.setattr(
        tool, "search_uploaded_docs_adaptive", lambda *args, **kwargs: Adaptive()
    )
    store = MultiHopTraceStore(tmp_path / "multi.sqlite3")
    result = tool.search_uploaded_docs_multihop(
        "Question",
        trace_store=store,
        trace_run_id="api-run",
        max_total_estimated_cost=100,
        max_estimated_cost=100,
    )
    assert result.abstain is False
    record = store.get_run(owner_id="default_user", run_id="api-run")
    assert record is not None
    assert record.summary.plan_fingerprint == result.retrieval.plan_fingerprint
