import json
from types import SimpleNamespace

import pytest

from tools.decomposition_model import (
    parse_decomposition_response,
    propose_decomposition,
    score_decomposition_plan,
)
from tools.query_decomposition import build_decomposition_plan


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def client(content=None, error=None):
    completions = FakeCompletions(content, error)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_strict_model_response_builds_valid_parallel_plan():
    raw = json.dumps(
        {
            "subquestions": [
                {"question_id": "a", "text": "Find A."},
                {"question_id": "b", "text": "Find B."},
                {
                    "question_id": "compare",
                    "text": "Compare A and B.",
                    "depends_on": ["a", "b"],
                    "relation": "compare",
                },
            ]
        }
    )
    plan = parse_decomposition_response(raw, "Compare A and B.")
    assert plan.batches == (("a", "b"), ("compare",))
    assert plan.terminal_questions == ("compare",)


def test_closed_schema_rejects_root_and_node_extras():
    with pytest.raises(ValueError, match="only subquestions"):
        parse_decomposition_response(
            json.dumps({"subquestions": [{"text": "Q"}], "answer": "invented"}),
            "Q",
        )
    with pytest.raises(ValueError, match="unsupported"):
        parse_decomposition_response(
            json.dumps({"subquestions": [{"text": "Q", "citation": "fake"}]}),
            "Q",
        )


def test_provider_success_records_digest_but_not_raw_response():
    raw = json.dumps({"subquestions": [{"question_id": "q1", "text": "Question"}]})
    fake, calls = client(raw)
    decision = propose_decomposition(
        "Question",
        client=fake,
        model="planner-model",
    )
    assert decision.used_model is True
    assert decision.response_digest is not None
    assert len(decision.response_digest) == 64
    assert decision.fallback_reason is None
    assert calls.calls[0]["temperature"] == 0.0
    assert "never evidence" in calls.calls[0]["messages"][0]["content"]


def test_invalid_or_failed_provider_falls_back_deterministically():
    malformed, _ = client("not-json")
    failed, _ = client(error=RuntimeError("provider detail must not escape"))
    first = propose_decomposition("Compare E5 and BGE.", client=malformed, model="m")
    second = propose_decomposition("Compare E5 and BGE.", client=failed, model="m")
    expected = build_decomposition_plan("Compare E5 and BGE.")
    assert first.used_model is False
    assert second.used_model is False
    assert first.plan.fingerprint == expected.fingerprint
    assert second.plan.fingerprint == expected.fingerprint
    assert first.fallback_reason == "ValueError"
    assert second.fallback_reason == "RuntimeError"


def test_plan_quality_is_bounded_and_reports_parallelism_depth():
    plan = build_decomposition_plan("Compare A and B.")
    quality = score_decomposition_plan(plan)
    assert 0.0 <= quality.score <= 1.0
    assert quality.parallel_fraction > 0.0
    assert quality.maximum_depth == 2
    assert quality.token_coverage > 0.0


def test_boolean_and_oversized_limits_fail_closed():
    fake, _ = client("{}")
    with pytest.raises(ValueError, match="max_subquestions"):
        propose_decomposition("Question", client=fake, model="m", max_subquestions=True)
    with pytest.raises(ValueError, match="model"):
        propose_decomposition("Question", client=fake, model="\x00")
