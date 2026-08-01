from __future__ import annotations

import tools.adaptive_rag_tool as api
from tools.adaptive_trace_store import AdaptiveTraceStore


def evidence(index: int):
    return {
        "doc_id": f"doc-{index}",
        "source_id": f"source-{index}",
        "score": 0.95,
        "page_number": index + 1,
        "generation_sequence": 1,
        "source_kind": "hybrid",
        "metadata": {
            "file_path": "/private/source.pdf",
            "token": "secret",
            "public": "retained",
        },
    }


def test_public_adaptive_api_can_persist_privacy_safe_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "search_uploaded_docs", lambda *a, **k: [evidence(i) for i in range(5)])
    store = AdaptiveTraceStore(tmp_path / "traces.sqlite3")
    result = api.search_uploaded_docs_adaptive(
        "Explain the evidence",
        owner_id="alice",
        max_attempts=1,
        trace_store=store,
        trace_run_id="api-run",
    )
    assert result.abstain is False
    assert store.get_run(owner_id="alice", run_id="api-run") is not None


def test_payload_removes_private_keys_and_unsupported_evidence(monkeypatch):
    values = [evidence(i) for i in range(5)]
    values.append({"unsupported": object()})
    monkeypatch.setattr(api, "search_uploaded_docs", lambda *a, **k: values)
    result = api.search_uploaded_docs_adaptive(
        "Explain the evidence", owner_id="alice", max_attempts=1
    )
    payload = api.adaptive_result_payload(result)
    assert payload["citations"]
    rendered = repr(payload)
    assert "private/source.pdf" not in rendered
    assert "secret" not in rendered
    assert "retained" in rendered
    assert "object at" not in rendered


def test_payload_accepts_model_dump_citations_and_preserves_public_provenance():
    from tools.adaptive_retrieval import EvidenceSignals, RetrievalAttempt
    from tools.adaptive_retrieval_runner import AdaptiveAttemptTrace, AdaptiveRetrievalResult

    class CitationLike:
        def model_dump(self, **_kwargs):
            return {
                "doc_id": "doc-1",
                "source_id": "chunk-1",
                "page_number": 2,
                "metadata": {"fused_score": 0.9, "generation_sequence": 1},
            }

    attempt = RetrievalAttempt("corpus-hybrid", 5, 20, reason="test")
    signals = EvidenceSignals(1, 1, 0.9, 0.9, 1, 1.0, 1.0, 1, 0.75, "sufficient")
    result = AdaptiveRetrievalResult(
        evidence=(CitationLike(),),
        traces=(AdaptiveAttemptTrace(attempt, 1, 1, signals),),
        final_signals=signals,
        exhausted=False,
        abstain=False,
        estimated_cost=attempt.estimated_cost,
    )
    payload = api.adaptive_result_payload(result)
    assert payload["citations"][0]["doc_id"] == "doc-1"
    assert payload["citations"][0]["page_number"] == 2
