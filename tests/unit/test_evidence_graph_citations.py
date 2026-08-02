from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.evidence_graph_citations import graph_evidence_to_citations
from tools.models import Citation


def _item(**overrides):
    values = {
        "owner_id": "alice",
        "origin": "cross_document",
        "doc_id": "doc-1",
        "generation": 3,
        "node_id": "b" * 64,
        "graph_digest": "c" * 64,
        "node_type": "claim",
        "label": "Primary result",
        "text": "Evidence text",
        "page_number": 2,
        "section": "Results",
        "score": 2.5,
        "provenance_digest": "d" * 64,
        "evidence_digest": "e" * 64,
        "text_sha256": "f" * 64,
        "matched_terms": ("result",),
        "lineage_step_digests": ("1" * 64,),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _selection(items, *, abstained: bool = False):
    return SimpleNamespace(
        owner_id="alice",
        graph_set_key="review",
        graph_set_id="2" * 64,
        graph_set_digest="3" * 64,
        authority_digest="4" * 64,
        query_digest="5" * 64,
        selection_digest="6" * 64,
        items=tuple(items),
        abstained=abstained,
        citation_conversion_performed=False,
        answer_generated=False,
    )


def test_conversion_uses_canonical_citation_and_preserves_lineage():
    citations = graph_evidence_to_citations(_selection((_item(),)))

    assert len(citations) == 1
    value = citations[0]
    assert isinstance(value, Citation)
    assert value.label == "[1]"
    assert value.source_type == "uploaded_document"
    assert value.url == "local://doc-1"
    assert value.doc_id == "doc-1"
    assert value.chunk_id == "b" * 64
    assert value.source_id == "e" * 64
    assert value.metadata["retrieval_strategy"] == "graph"
    assert value.metadata["graph_lineage_step_digests"] == ["1" * 64]
    assert value.metadata["graph_matched_term_count"] == 1
    assert len(value.metadata["graph_matched_terms_digest"]) == 64
    assert "owner_id" not in value.metadata
    assert "graph_matched_terms" not in value.metadata
    assert "query" not in value.metadata
    assert "source_path" not in value.metadata


def test_label_offset_origin_filter_limit_and_generation_scoped_deduplication():
    duplicate = _item()
    other = _item(
        doc_id="doc-2",
        node_id="7" * 64,
        evidence_digest="8" * 64,
    )

    values = graph_evidence_to_citations(
        _selection((duplicate, duplicate, other)),
        start_index=5,
        max_citations=1,
        allowed_origins=("cross_document",),
    )

    assert [value.label for value in values] == ["[5]"]


def test_abstention_returns_no_citations_and_rejects_inconsistent_evidence():
    assert graph_evidence_to_citations(_selection((), abstained=True)) == []

    with pytest.raises(ValueError, match="abstained"):
        graph_evidence_to_citations(_selection((_item(),), abstained=True))


def test_invalid_policy_and_item_fields_fail_closed():
    with pytest.raises(ValueError, match="max_citations"):
        graph_evidence_to_citations(_selection((_item(),)), max_citations=51)
    with pytest.raises(ValueError, match="allowed_origins"):
        graph_evidence_to_citations(
            _selection((_item(),)),
            allowed_origins=("invented",),
        )
    with pytest.raises(ValueError, match="score"):
        graph_evidence_to_citations(
            _selection((_item(score=float("nan")),))
        )


def test_owner_scope_and_prior_conversion_fail_closed():
    with pytest.raises(ValueError, match="owner scope"):
        graph_evidence_to_citations(
            _selection((_item(owner_id="bob"),))
        )

    selected = _selection((_item(),))
    selected.citation_conversion_performed = True
    with pytest.raises(ValueError, match="already contain"):
        graph_evidence_to_citations(selected)
