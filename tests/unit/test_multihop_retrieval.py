import time

import pytest

from tools.multihop_retrieval import run_multihop_retrieval
from tools.query_decomposition import Subquestion, build_decomposition_plan


def plan():
    return build_decomposition_plan(
        "Compare two systems.",
        proposed_subquestions=[
            Subquestion("a", "Find system A evidence."),
            Subquestion("b", "Find system B evidence."),
            Subquestion(
                "compare", "Compare the systems.",
                depends_on=("a", "b"), relation="compare",
            ),
        ],
    )


def test_multihop_execution_preserves_lineage_and_builds_document_joins():
    calls = []

    def search(question, dependencies):
        calls.append((question.question_id, tuple(item.hop_id for item in dependencies)))
        if question.question_id == "a":
            return [{"source_id": "a-1", "doc_id": "shared", "text": "A evidence", "score": 0.8}]
        if question.question_id == "b":
            return [{"source_id": "b-1", "doc_id": "shared", "text": "B evidence", "score": 0.7}]
        assert {item.hop_id for item in dependencies} == {"a", "b"}
        return [{"source_id": "c-1", "doc_id": "shared", "text": "Comparison evidence", "score": 0.9}]

    result = run_multihop_retrieval(plan(), search=search, max_workers=2)
    assert result.abstain is False
    assert result.exhausted is False
    assert result.terminal_evidence_count == 1
    assert len(result.evidence) == 3
    assert result.joins[0].supporting_hops == ("a", "b", "compare")
    assert result.joins[0].source_ids == ("a-1", "b-1", "c-1")
    assert any(call[0] == "compare" and set(call[1]) == {"a", "b"} for call in calls)


def test_dependent_hop_is_skipped_when_required_evidence_is_missing():
    def search(question, dependencies):
        if question.question_id == "a":
            return []
        if question.question_id == "b":
            return [{"source_id": "b", "doc_id": "b", "text": "B"}]
        raise AssertionError("dependent search must not run")

    result = run_multihop_retrieval(plan(), search=search)
    trace = next(item for item in result.traces if item.hop_id == "compare")
    assert trace.status == "skipped_missing_dependency_evidence"
    assert result.abstain is True
    assert result.exhausted is True


def test_errors_are_contained_and_do_not_erase_other_parallel_hops():
    def search(question, dependencies):
        if question.question_id == "a":
            raise RuntimeError("backend down")
        return [{"source_id": question.question_id, "doc_id": question.question_id, "text": "ok"}]

    result = run_multihop_retrieval(
        plan(), search=search, require_dependency_evidence=False
    )
    a_trace = next(item for item in result.traces if item.hop_id == "a")
    assert a_trace.status == "error"
    assert a_trace.error_type == "RuntimeError"
    assert any(item.hop_id == "b" for item in result.evidence)


def test_timeout_returns_without_waiting_for_hung_worker():
    single = build_decomposition_plan("Slow question")

    def search(question, dependencies):
        time.sleep(0.25)
        return [{"source_id": "late", "text": "late"}]

    started = time.monotonic()
    result = run_multihop_retrieval(
        single, search=search, hop_timeout_seconds=0.01,
        global_timeout_seconds=0.02,
    )
    elapsed = time.monotonic() - started
    assert elapsed < 0.15
    assert result.traces[0].status == "timeout"
    assert result.evidence == ()
    assert result.abstain is True


def test_global_deadline_marks_later_batches_without_running_them():
    calls = []

    def search(question, dependencies):
        calls.append(question.question_id)
        if question.question_id in {"a", "b"}:
            time.sleep(0.1)
        return [{"source_id": question.question_id, "text": "ok"}]

    result = run_multihop_retrieval(
        plan(), search=search, hop_timeout_seconds=1.0,
        global_timeout_seconds=0.01,
    )
    compare = next(trace for trace in result.traces if trace.hop_id == "compare")
    assert compare.status in {"global_timeout", "skipped_global_timeout"}
    assert "compare" not in calls


def test_hostile_identifiers_and_controls_are_not_stringified_or_exposed():
    single = build_decomposition_plan("Question")

    class Hostile:
        called = False
        def __str__(self):
            self.called = True
            raise AssertionError("must not stringify")

    hostile = Hostile()
    result = run_multihop_retrieval(
        single,
        search=lambda *_: [{"source_id": "bad\x7f", "doc_id": hostile, "text": "safe"}],
    )
    assert hostile.called is False
    assert result.evidence[0].source_id.startswith("derived-")
    assert "\x7f" not in result.evidence[0].source_id


def test_invalid_callback_results_and_limits_fail_closed():
    single = build_decomposition_plan("Question")
    result = run_multihop_retrieval(single, search=lambda *_: "not evidence")
    assert result.traces[0].status == "error"
    assert result.traces[0].error_type == "RuntimeError"
    with pytest.raises(ValueError, match="max_workers"):
        run_multihop_retrieval(single, search=lambda *_: [], max_workers=True)
    with pytest.raises(ValueError, match="numeric"):
        run_multihop_retrieval(single, search=lambda *_: [], hop_timeout_seconds=True)
