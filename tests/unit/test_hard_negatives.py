from __future__ import annotations

from evaluation.hard_negatives import mine_lexical_hard_negatives


def test_hard_negative_mining_excludes_relevant_and_ranks_overlap_deterministically():
    result = mine_lexical_hard_negatives(
        query="retrieval evidence graph",
        documents={
            "relevant": "retrieval evidence graph exact",
            "hard": "retrieval evidence unrelated",
            "medium": "retrieval other",
            "easy": "completely different words",
        },
        relevant_ids=["relevant"],
        limit=3,
    )
    assert [item.document_id for item in result] == ["hard", "medium", "easy"]
    assert [item.rank for item in result] == [1, 2, 3]
    assert result[0].lexical_overlap > result[1].lexical_overlap > result[2].lexical_overlap


def test_minimum_overlap_filters_easy_negatives():
    result = mine_lexical_hard_negatives(
        query="retrieval evidence graph",
        documents={"hard": "retrieval evidence", "easy": "different words"},
        relevant_ids=[],
        minimum_overlap=0.2,
    )
    assert [item.document_id for item in result] == ["hard"]


def test_ties_break_by_shared_terms_then_document_id():
    result = mine_lexical_hard_negatives(
        query="a b",
        documents={"z": "a c", "a": "a d"},
        relevant_ids=[],
    )
    assert [item.document_id for item in result] == ["a", "z"]
