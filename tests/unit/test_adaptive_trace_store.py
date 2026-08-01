from __future__ import annotations

import sqlite3

import pytest

from tools.adaptive_retrieval_runner import run_adaptive_retrieval
from tools.adaptive_trace_store import AdaptiveTraceStore


def evidence(index: int, score: float = 0.95):
    return {
        "doc_id": f"doc-{index}",
        "source_id": f"source-{index}",
        "score": score,
        "page_number": index + 1,
        "generation_sequence": 1,
        "source_kind": "hybrid",
    }


def sufficient_result():
    return run_adaptive_retrieval(
        "Explain the evidence",
        search=lambda *a, **k: [evidence(index) for index in range(5)],
        owner_id="alice",
        max_attempts=1,
    )


def test_trace_store_persists_only_query_fingerprint_and_aggregates(tmp_path):
    path = tmp_path / "adaptive.sqlite3"
    store = AdaptiveTraceStore(path)
    query = "Explain the private evidence phrase"
    result = sufficient_result()
    run_id = store.record_result(
        query=query,
        owner_id="alice",
        result=result,
        run_id="run-1",
        started_at=10.0,
        completed_at=11.0,
    )
    assert run_id == "run-1"
    record = store.get_run(owner_id="alice", run_id="run-1")
    assert record is not None
    assert record.summary.query_sha256 != query
    assert len(record.summary.query_sha256) == 64
    assert record.summary.final_decision == "sufficient"
    assert record.attempts[0].mode in {"dense", "corpus-hybrid", "corpus-sparse"}
    assert query.encode() not in path.read_bytes()
    aggregate = store.aggregate(owner_id="alice")
    assert aggregate.run_count == 1
    assert aggregate.abstention_count == 0
    assert aggregate.decisions == (("sufficient", 1),)
    assert sum(count for _, count in aggregate.route_attempts) == 1


def test_trace_store_is_owner_scoped_and_exact_replay_is_idempotent(tmp_path):
    store = AdaptiveTraceStore(tmp_path / "adaptive.sqlite3")
    result = sufficient_result()
    kwargs = dict(
        query="Explain the evidence",
        owner_id="alice",
        result=result,
        run_id="stable-run",
        started_at=1.0,
        completed_at=2.0,
    )
    assert store.record_result(**kwargs) == "stable-run"
    assert store.record_result(**kwargs) == "stable-run"
    assert store.get_run(owner_id="bob", run_id="stable-run") is None
    assert store.list_runs(owner_id="alice")[0].run_id == "stable-run"
    with pytest.raises(ValueError, match="different adaptive trace"):
        store.record_result(**{**kwargs, "query": "different query"})


def test_trace_store_records_generic_error_types_without_messages(tmp_path):
    store = AdaptiveTraceStore(tmp_path / "adaptive.sqlite3")

    def fail(*args, **kwargs):
        raise RuntimeError("sensitive provider message and secret-token")

    result = run_adaptive_retrieval(
        "Question", search=fail, owner_id="alice", max_attempts=1
    )
    store.record_result(
        query="Question",
        owner_id="alice",
        result=result,
        run_id="failed-run",
        started_at=1.0,
        completed_at=2.0,
    )
    record = store.get_run(owner_id="alice", run_id="failed-run")
    assert record is not None
    assert record.attempts[0].error_type == "RuntimeError"
    raw = store.path.read_bytes()
    assert b"sensitive provider message" not in raw
    assert b"secret-token" not in raw
    aggregate = store.aggregate(owner_id="alice")
    assert aggregate.error_run_count == 1
    assert aggregate.abstention_count == 1
    assert aggregate.exhausted_count == 1


def test_trace_store_detects_database_identity_replacement(tmp_path):
    path = tmp_path / "adaptive.sqlite3"
    store = AdaptiveTraceStore(path)
    original = tmp_path / "original.sqlite3"
    path.rename(original)
    path.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="replaced"):
        store.list_runs(owner_id="alice")
    assert store.ping() is False


def test_trace_store_rejects_redirected_paths(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable")
    with pytest.raises(ValueError, match="symbolic links"):
        AdaptiveTraceStore(linked / "adaptive.sqlite3")


def test_trace_store_detects_attempt_count_corruption(tmp_path):
    store = AdaptiveTraceStore(tmp_path / "adaptive.sqlite3")
    store.record_result(
        query="Explain the evidence",
        owner_id="alice",
        result=sufficient_result(),
        run_id="run-1",
        started_at=1.0,
        completed_at=2.0,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM adaptive_attempts WHERE run_id='run-1'")
        connection.commit()
    with pytest.raises(RuntimeError, match="count is corrupt"):
        store.get_run(owner_id="alice", run_id="run-1")


def test_trace_store_prunes_only_the_selected_owner(tmp_path):
    store = AdaptiveTraceStore(tmp_path / "adaptive.sqlite3")
    result = sufficient_result()
    for index in range(3):
        store.record_result(
            query=f"Question {index}",
            owner_id="alice",
            result=result,
            run_id=f"alice-{index}",
            started_at=float(index),
            completed_at=float(index + 1),
        )
    store.record_result(
        query="Bob question",
        owner_id="bob",
        result=result,
        run_id="bob-1",
        started_at=1.0,
        completed_at=2.0,
    )
    assert store.prune_owner(owner_id="alice", retain_latest=1) == 2
    assert len(store.list_runs(owner_id="alice")) == 1
    assert len(store.list_runs(owner_id="bob")) == 1
