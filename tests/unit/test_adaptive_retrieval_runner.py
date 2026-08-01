from types import SimpleNamespace

import pytest

from tools.adaptive_retrieval_runner import run_adaptive_retrieval


def item(identifier, doc_id, score, *, generation=1, page=1):
    return SimpleNamespace(
        chunk_id=identifier,
        source_id=identifier,
        doc_id=doc_id,
        page_number=page,
        metadata={
            "fused_score": score,
            "generation_sequence": generation,
            "evidence_kind": "dense_chunk",
        },
    )


def test_sufficient_first_attempt_stops_early_and_forwards_route():
    calls = []

    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [
            item("a", "doc-1", 0.95),
            item("b", "doc-2", 0.85),
            item("c", "doc-3", 0.75),
        ]

    result = run_adaptive_retrieval(
        "Compare the effects of A versus B",
        search=search,
        owner_id="alice",
        top_k=3,
    )

    assert result.abstain is False
    assert len(result.traces) == 1
    assert result.final_signals.decision == "sufficient"
    assert calls[0][1]["retrieval_mode"] == "corpus-hybrid"
    assert calls[0][1]["use_multi_query"] is True
    assert calls[0][1]["reranker"] == "heuristic"


def test_weak_first_attempt_accumulates_and_deduplicates():
    calls = []

    def search(_query, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return [item("same", "doc-1", 0.2)]
        return [
            item("same", "doc-1", 0.7),
            item("new-1", "doc-2", 0.85),
            item("new-2", "doc-3", 0.8),
        ]

    result = run_adaptive_retrieval(
        "Explain why this mechanism changes",
        search=search,
        owner_id="alice",
        max_attempts=3,
    )

    assert len(result.traces) >= 2
    assert len(result.evidence) == 3
    assert result.final_signals.unique_documents == 3
    assert result.estimated_cost == sum(
        trace.attempt.estimated_cost for trace in result.traces
    )


def test_failed_attempt_records_only_error_type_and_continues():
    calls = 0

    def search(_query, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("failed at /private/vector.sqlite3")
        return [
            item("a", "doc-1", 0.9),
            item("b", "doc-2", 0.8),
            item("c", "doc-3", 0.7),
        ]

    result = run_adaptive_retrieval(
        "Find evidence for this claim",
        search=search,
        owner_id="alice",
    )

    assert result.traces[0].error_type == "RuntimeError"
    assert "/private" not in repr(result.traces)
    assert result.final_signals.decision == "sufficient"


def test_all_failed_attempts_exhaust_and_abstain():
    result = run_adaptive_retrieval(
        "Why is this true?",
        search=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("unavailable")
        ),
        owner_id="alice",
        max_attempts=2,
    )
    assert result.abstain is True
    assert result.exhausted is True
    assert result.evidence == ()
    assert len(result.traces) == 2
    assert all(trace.error_type == "RuntimeError" for trace in result.traces)


def test_owner_validation_precedes_search():
    calls = []
    with pytest.raises(ValueError):
        run_adaptive_retrieval(
            "question",
            search=lambda *_args, **_kwargs: calls.append(True),
            owner_id="../alice",
        )
    assert calls == []


def test_invalid_search_and_budget_controls_fail_closed():
    with pytest.raises(ValueError, match="callable"):
        run_adaptive_retrieval("question", search=None, owner_id="alice")
    with pytest.raises(ValueError, match="max_attempts"):
        run_adaptive_retrieval(
            "question",
            search=lambda *_args, **_kwargs: [],
            owner_id="alice",
            max_attempts=True,
        )
