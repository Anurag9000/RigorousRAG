import importlib.util
import sys
import types

if importlib.util.find_spec("tools.adaptive_rag_tool") is None:
    stub = types.ModuleType("tools.adaptive_rag_tool")
    stub.search_uploaded_docs_adaptive = lambda *args, **kwargs: None
    sys.modules["tools.adaptive_rag_tool"] = stub

from tools import multihop_rag_tool
from tools.multihop_rag_tool import multihop_result_payload, search_uploaded_docs_multihop


class AdaptiveResult:
    def __init__(self, evidence):
        self.evidence = tuple(evidence)


def test_public_multihop_tool_propagates_bounded_terms_and_preserves_lineage(monkeypatch):
    queries = []

    def adaptive(query, **kwargs):
        queries.append(query)
        if query.startswith("Find evidence about E5"):
            return AdaptiveResult(
                [
                    {
                        "source_id": "e5-source",
                        "doc_id": "shared-doc",
                        "text": "E5 has asymmetric retrieval uniqueconcept.",
                        "score": 0.8,
                    }
                ]
            )
        if query.startswith("Find evidence about BGE-M3"):
            return AdaptiveResult(
                [
                    {
                        "source_id": "bge-source",
                        "doc_id": "shared-doc",
                        "text": "BGE-M3 supports multilingual retrieval otherconcept.",
                        "score": 0.7,
                    }
                ]
            )
        assert "Dependency-derived search terms:" in query
        assert "uniqueconcept" in query
        assert "otherconcept" in query
        return AdaptiveResult(
            [
                {
                    "source_id": "comparison-source",
                    "doc_id": "shared-doc",
                    "text": "Direct comparison evidence.",
                    "score": 0.9,
                }
            ]
        )

    monkeypatch.setattr(multihop_rag_tool, "search_uploaded_docs_adaptive", adaptive)
    result = search_uploaded_docs_multihop(
        "Compare E5 and BGE-M3 for retrieval.",
        owner_id="owner",
        max_workers=2,
    )
    assert result.abstain is False
    assert result.terminal_evidence_count == 1
    assert len(result.evidence) == 3
    assert result.joins[0].source_ids == (
        "bge-source",
        "comparison-source",
        "e5-source",
    )
    assert len(queries) == 3

    payload = multihop_result_payload(result)
    assert payload["abstain"] is False
    assert payload["evidence"][0]["citation"]["source_id"] in {
        "e5-source",
        "bge-source",
    }
    assert payload["evidence"][-1]["lineage"]["hop_id"] == "q3"
    assert payload["joins"][0]["supporting_hops"] == ("q1", "q2", "q3")


def test_public_multihop_tool_rejects_boolean_limits(monkeypatch):
    monkeypatch.setattr(
        multihop_rag_tool,
        "search_uploaded_docs_adaptive",
        lambda *args, **kwargs: AdaptiveResult(()),
    )
    try:
        search_uploaded_docs_multihop("Question", max_workers=True)
    except ValueError as exc:
        assert "max_workers" in str(exc)
    else:
        raise AssertionError("boolean worker limits must be rejected")
