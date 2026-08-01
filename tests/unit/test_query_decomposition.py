import pytest

from tools.query_decomposition import Subquestion, build_decomposition_plan


def test_comparison_plan_creates_parallel_lookups_and_dependent_comparison():
    plan = build_decomposition_plan("Compare E5 and BGE-M3 for scientific retrieval.")
    assert len(plan.subquestions) == 3
    assert plan.batches[0] == ("q1", "q2")
    assert plan.batches[1] == ("q3",)
    assert plan.subquestions[-1].relation == "compare"
    assert plan.subquestions[-1].depends_on == ("q1", "q2")
    assert plan.terminal_questions == ("q3",)
    assert len(plan.fingerprint) == 64


def test_explicit_dag_batches_parallel_and_serial_hops():
    plan = build_decomposition_plan(
        "Synthesize the evidence.",
        proposed_subquestions=[
            Subquestion("population", "Find population evidence."),
            Subquestion("outcome", "Find outcome evidence."),
            Subquestion(
                "synthesis",
                "Synthesize population and outcome evidence.",
                depends_on=("population", "outcome"),
                relation="synthesize",
            ),
        ],
    )
    assert plan.batches == (("outcome", "population"), ("synthesis",))
    assert plan.terminal_questions == ("synthesis",)


def test_plan_fingerprint_is_stable_and_changes_with_dependencies():
    first = build_decomposition_plan(
        "Question", proposed_subquestions=[{"question_id": "q1", "text": "Question"}]
    )
    second = build_decomposition_plan(
        "Question", proposed_subquestions=[{"question_id": "q1", "text": "Question"}]
    )
    changed = build_decomposition_plan(
        "Question",
        proposed_subquestions=[
            {"question_id": "q1", "text": "Question"},
            {"question_id": "q2", "text": "Follow up", "depends_on": ["q1"]},
        ],
    )
    assert first.fingerprint == second.fingerprint
    assert changed.fingerprint != first.fingerprint


def test_missing_dependencies_duplicates_and_cycles_fail_closed():
    with pytest.raises(ValueError, match="declared"):
        build_decomposition_plan(
            "Question",
            proposed_subquestions=[
                {"question_id": "q1", "text": "Question", "depends_on": ["q2"]}
            ],
        )
    with pytest.raises(ValueError, match="unique"):
        build_decomposition_plan(
            "Question",
            proposed_subquestions=[
                {"question_id": "q1", "text": "One"},
                {"question_id": "q1", "text": "Two"},
            ],
        )
    with pytest.raises(ValueError, match="acyclic"):
        build_decomposition_plan(
            "Question",
            proposed_subquestions=[
                {"question_id": "q1", "text": "One", "depends_on": ["q2"]},
                {"question_id": "q2", "text": "Two", "depends_on": ["q1"]},
            ],
        )


def test_limits_and_hostile_iterables_are_rejected_without_full_materialization():
    with pytest.raises(ValueError, match="max_subquestions"):
        build_decomposition_plan("Question", max_subquestions=True)
    with pytest.raises(ValueError, match="sequence"):
        build_decomposition_plan("Question", proposed_subquestions="bad")

    consumed = 0

    def oversized():
        nonlocal consumed
        while True:
            consumed += 1
            yield {"question_id": f"q{consumed}", "text": "Question"}

    with pytest.raises(ValueError, match="at most"):
        build_decomposition_plan(
            "Question", proposed_subquestions=oversized(), max_subquestions=3
        )
    assert consumed == 4

    class Hostile:
        def __iter__(self):
            yield {"question_id": "q1", "text": "Question"}
            raise RuntimeError("boom")

    with pytest.raises(ValueError, match="safely iterable"):
        build_decomposition_plan("Question", proposed_subquestions=Hostile())


def test_controls_unknown_fields_and_pre_normalization_size_fail_closed():
    with pytest.raises(ValueError, match="valid"):
        build_decomposition_plan("question\x7fhidden")
    with pytest.raises(ValueError, match="unknown fields"):
        build_decomposition_plan(
            "Question",
            proposed_subquestions=[
                {"question_id": "q1", "text": "Question", "ignored": True}
            ],
        )
    with pytest.raises(ValueError, match="1-4000"):
        Subquestion("q1", " " * 4_001)
