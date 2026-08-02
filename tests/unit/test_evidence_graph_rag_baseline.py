from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest

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


def policy(policy_id="policy-v1"):
    return GraphRAGRegressionPolicy(
        policy_id=policy_id,
        min_run_count=3,
        min_seed_count=3,
        min_cases_per_run=1,
    )


def test_initial_baseline_requires_explicit_empty_pointer(tmp_path):
    store = GraphRAGBaselineStore(tmp_path / "baselines.sqlite3")
    candidate = report()
    selected = policy()
    value = store.activate(
        candidate,
        selected,
        expected_current_baseline_digest=None,
        now=1.0,
    )
    assert value.previous_baseline_digest is None
    assert value.activation_regression_digest is None
    assert value.benchmark_report.report_digest == candidate.report_digest
    assert (
        store.current(
            benchmark_fingerprint=candidate.benchmark_fingerprint,
            policy_id=selected.policy_id,
        )
        == value
    )
    assert store.history(
        benchmark_fingerprint=candidate.benchmark_fingerprint,
        policy_id=selected.policy_id,
    ) == (value,)
    with pytest.raises(RuntimeError, match="explicit expectation"):
        store.activate(
            candidate,
            selected,
            expected_current_baseline_digest=None,
            now=2.0,
        )


def test_replacement_requires_exact_eligible_regression(tmp_path):
    store = GraphRAGBaselineStore(tmp_path / "baselines.sqlite3")
    baseline = report()
    candidate = report(work=11)
    selected = policy()
    initial = store.activate(
        baseline,
        selected,
        expected_current_baseline_digest=None,
        now=1.0,
    )
    regression = evaluate_graph_rag_regression(
        baseline, candidate, selected
    )
    assert regression.decision == "eligible"
    replacement = store.activate(
        candidate,
        selected,
        expected_current_baseline_digest=initial.baseline_digest,
        regression=regression,
        now=2.0,
    )
    assert replacement.previous_baseline_digest == initial.baseline_digest
    assert (
        replacement.activation_regression_digest
        == regression.report_digest
    )
    assert replacement.benchmark_report.report_digest == candidate.report_digest
    assert store.history(
        benchmark_fingerprint=candidate.benchmark_fingerprint,
        policy_id=selected.policy_id,
    ) == (replacement, initial)


def test_blocked_or_identity_mismatched_regression_is_refused(tmp_path):
    store = GraphRAGBaselineStore(tmp_path / "baselines.sqlite3")
    baseline = report()
    selected = policy()
    initial = store.activate(
        baseline,
        selected,
        expected_current_baseline_digest=None,
    )
    bad_candidate = report(selected=False)
    blocked = evaluate_graph_rag_regression(
        baseline, bad_candidate, selected
    )
    assert blocked.decision == "blocked"
    with pytest.raises(RuntimeError, match="not eligible"):
        store.activate(
            bad_candidate,
            selected,
            expected_current_baseline_digest=initial.baseline_digest,
            regression=blocked,
        )
    candidate = report(work=11)
    eligible = evaluate_graph_rag_regression(
        baseline, candidate, selected
    )
    forged = replace(eligible, candidate_report_digest="f" * 64)
    with pytest.raises(RuntimeError, match="identities differ"):
        store.activate(
            candidate,
            selected,
            expected_current_baseline_digest=initial.baseline_digest,
            regression=forged,
        )
    assert store.current(
        benchmark_fingerprint=baseline.benchmark_fingerprint,
        policy_id=selected.policy_id,
    ) == initial


def test_policy_scopes_have_independent_current_pointers(tmp_path):
    store = GraphRAGBaselineStore(tmp_path / "baselines.sqlite3")
    value = report()
    first = store.activate(
        value, policy("policy-a"), expected_current_baseline_digest=None
    )
    second = store.activate(
        value, policy("policy-b"), expected_current_baseline_digest=None
    )
    assert first.policy_id == "policy-a"
    assert second.policy_id == "policy-b"
    assert store.current(
        benchmark_fingerprint=value.benchmark_fingerprint,
        policy_id="policy-a",
    ) == first
    assert store.current(
        benchmark_fingerprint=value.benchmark_fingerprint,
        policy_id="policy-b",
    ) == second


def test_payload_pointer_and_database_tampering_fail_closed(tmp_path):
    path = tmp_path / "baselines.sqlite3"
    store = GraphRAGBaselineStore(path)
    value = report()
    selected = policy()
    record = store.activate(
        value, selected, expected_current_baseline_digest=None
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE graph_rag_baselines SET payload_json=? "
            "WHERE baseline_digest=?",
            ('{"benchmark_id":"broken"}', record.baseline_digest),
        )
    with pytest.raises(RuntimeError, match="corrupt"):
        store.current(
            benchmark_fingerprint=value.benchmark_fingerprint,
            policy_id=selected.policy_id,
        )

    guarded_path = tmp_path / "guarded.sqlite3"
    guarded = GraphRAGBaselineStore(guarded_path)
    guarded_path.rename(tmp_path / "old.sqlite3")
    guarded_path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="identity changed"):
        guarded.history(
            benchmark_fingerprint="f" * 64,
            policy_id="policy-v1",
        )
