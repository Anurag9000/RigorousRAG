from __future__ import annotations

import pytest

import tools.adaptive_retrieval_runner as runner
from tools.adaptive_retrieval import CorrectivePlan, RetrievalAttempt, analyze_query


def evidence(index: int, score: float = 0.95):
    return {
        "doc_id": f"doc-{index}",
        "source_id": f"source-{index}",
        "score": score,
        "page_number": index + 1,
        "generation_sequence": 1,
        "source_kind": "dense",
    }


def test_success_on_last_planned_attempt_is_not_exhausted(monkeypatch):
    attempts = (
        RetrievalAttempt("dense", 5, 5),
        RetrievalAttempt("corpus-hybrid", 5, 10, reason="final"),
    )
    plan = CorrectivePlan(
        analyze_query("Explain the method"),
        None,
        attempts,
        sum(item.estimated_cost for item in attempts),
    )
    monkeypatch.setattr(runner, "build_corrective_plan", lambda *a, **k: plan)
    calls = 0

    def search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [] if calls == 1 else [evidence(index) for index in range(5)]

    result = runner.run_adaptive_retrieval(
        "Explain the method", search=search, owner_id="alice"
    )
    assert result.final_signals.decision == "sufficient"
    assert result.abstain is False
    assert result.exhausted is False
    assert len(result.traces) == len(attempts)


def test_fallback_evidence_ids_do_not_overwrite_distinct_content(monkeypatch):
    attempts = (
        RetrievalAttempt("dense", 2, 2),
        RetrievalAttempt("corpus-hybrid", 2, 4, reason="retry"),
    )
    plan = CorrectivePlan(
        analyze_query("general question"),
        None,
        attempts,
        sum(item.estimated_cost for item in attempts),
    )
    monkeypatch.setattr(runner, "build_corrective_plan", lambda *a, **k: plan)
    calls = 0

    def search(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [
            {
                "doc_id": "same-doc",
                "quote": "first distinct passage" if calls == 1 else "second distinct passage",
                "score": 0.1,
            }
        ]

    result = runner.run_adaptive_retrieval(
        "general question", search=search, owner_id="alice"
    )
    assert len(result.evidence) == 2
    assert result.abstain is True
    assert result.exhausted is True


def test_fallback_identifier_never_stringifies_hostile_objects(monkeypatch):
    attempts = (RetrievalAttempt("dense", 1, 1),)
    plan = CorrectivePlan(
        analyze_query("general question"),
        None,
        attempts,
        attempts[0].estimated_cost,
    )
    monkeypatch.setattr(runner, "build_corrective_plan", lambda *a, **k: plan)

    class Hostile:
        called = False

        def __str__(self):
            self.called = True
            raise AssertionError("hostile __str__ invoked")

    value = {"doc_id": Hostile(), "score": 0.1}
    result = runner.run_adaptive_retrieval(
        "general question", search=lambda *a, **k: [value], owner_id="alice"
    )
    assert len(result.evidence) == 1
    assert value["doc_id"].called is False


def test_adaptive_query_rejects_del_control_character():
    with pytest.raises(ValueError, match="invalid"):
        analyze_query("question\x7fhidden")
