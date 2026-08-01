from dataclasses import dataclass

import pytest

import tools.multihop_rag_tool as tool
from tools.decomposition_model import DecompositionDecision, score_decomposition_plan
from tools.multihop_budget import allocate_multihop_budget
from tools.multihop_retrieval import HopEvidence, MultiHopResult
from tools.query_decomposition import build_decomposition_plan


def test_schema_and_execution_forward_global_deadline(monkeypatch):
    properties = tool.MULTIHOP_RAG_SEARCH_TOOL_DEF["function"]["parameters"]["properties"]
    assert properties["global_timeout_seconds"]["maximum"] == 3600
    captured = {}
    plan = build_decomposition_plan("Question")

    def fake_run(received_plan, **kwargs):
        captured.update(kwargs)
        return MultiHopResult(
            received_plan.fingerprint, (), (), (), received_plan.terminal_questions,
            0, True, True,
        )

    monkeypatch.setattr(tool, "run_multihop_retrieval", fake_run)
    result = tool.search_uploaded_docs_multihop(
        "Question",
        global_timeout_seconds=17.5,
        max_total_estimated_cost=100,
        max_estimated_cost=100,
    )
    assert result.abstain is True
    assert captured["global_timeout_seconds"] == 17.5
    with pytest.raises(ValueError, match="global_timeout_seconds"):
        tool.search_uploaded_docs_multihop("Question", global_timeout_seconds=True)


def test_multihop_payload_removes_private_nested_fields_and_hostile_values():
    plan = build_decomposition_plan("Question")
    decision = DecompositionDecision(
        plan, False, None, None, score_decomposition_plan(plan)
    )
    budget = allocate_multihop_budget(
        plan, top_k=5, total_limit=100, per_hop_limit=100
    )
    raw = {
        "doc_id": "doc-1",
        "metadata": {
            "file_path": "/private/source.pdf",
            "token": "secret-token",
            "public": "retained",
        },
    }
    invalid = {"doc_id": "doc-2", "metadata": {"value": object()}}
    evidence = (
        HopEvidence("q1:s1", "q1", "s1", "doc-1", 1, "text", 0.9, raw),
        HopEvidence("q1:s2", "q1", "s2", "doc-2", 2, "text", 0.8, invalid),
    )
    retrieval = MultiHopResult(
        plan.fingerprint, evidence, (), (), plan.terminal_questions,
        2, False, False,
    )
    payload = tool.multihop_result_payload(
        tool.MultiHopRAGResult(retrieval, decision, budget)
    )
    rendered = repr(payload)
    assert "/private/source.pdf" not in rendered
    assert "secret-token" not in rendered
    assert "retained" in rendered
    assert len(payload["evidence"]) == 1


def test_public_payload_never_stringifies_hostile_model():
    class Hostile:
        called = False
        def __str__(self):
            self.called = True
            raise AssertionError("must not stringify")
    hostile = Hostile()
    assert tool.public_model_payload({"value": hostile}) is None
    assert hostile.called is False
