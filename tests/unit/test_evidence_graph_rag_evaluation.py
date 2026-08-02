from __future__ import annotations

import pytest

from tools.evidence_graph_rag import (
    GraphEvidenceItem,
    GraphEvidenceSelection,
    GraphTraversalStep,
)
from tools.evidence_graph_rag_evaluation import (
    GraphNodeLocator,
    GraphRAGGoldCase,
    aggregate_graph_evaluations,
    evaluate_graph_selection,
    gold_case_from_mapping,
)

SET_ID = "a" * 64
SET_DIGEST = "b" * 64
AUTHORITY = "c" * 64
QUERY = "d" * 64
NODE_A = "1" * 64
NODE_B = "2" * 64
EDGE = "3" * 64
EDGE_PROV = "4" * 64
GRAPH_A = "5" * 64
GRAPH_B = "6" * 64
PROV_A = "7" * 64
PROV_B = "8" * 64


def traversal():
    return GraphTraversalStep(
        traversal_kind="cross_document",
        source_doc_id="doc-a",
        source_generation=1,
        source_node_id=NODE_A,
        edge_id=EDGE,
        edge_type="supports",
        edge_provenance_digest=EDGE_PROV,
        target_doc_id="doc-b",
        target_generation=2,
        target_node_id=NODE_B,
        depth=1,
        weight=0.8,
    )


def item_a():
    return GraphEvidenceItem(
        owner_id="alice",
        doc_id="doc-a",
        generation=1,
        graph_digest=GRAPH_A,
        node_id=NODE_A,
        node_type="claim",
        label="Claim A",
        text="alpha",
        page_number=1,
        section="Results",
        score=10.0,
        matched_terms=("alpha",),
        provenance_digest=PROV_A,
        origin="lexical",
        lineage_step_digests=(),
    )


def item_b(step):
    return GraphEvidenceItem(
        owner_id="alice",
        doc_id="doc-b",
        generation=2,
        graph_digest=GRAPH_B,
        node_id=NODE_B,
        node_type="claim",
        label="Claim B",
        text="beta",
        page_number=2,
        section="Discussion",
        score=8.0,
        matched_terms=("alpha",),
        provenance_digest=PROV_B,
        origin="cross_document",
        lineage_step_digests=(step.step_digest,),
    )


def selection(*, include_b=True, abstained=False):
    step = traversal()
    items = () if abstained else ((item_a(), item_b(step)) if include_b else (item_a(),))
    steps = () if abstained or not include_b else (step,)
    return GraphEvidenceSelection(
        owner_id="alice",
        graph_set_key="review",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        authority_digest=AUTHORITY,
        query_digest=QUERY,
        items=items,
        traversals=steps,
        lexical_seed_count=0 if abstained else 1,
        expanded_count=1 if include_b and not abstained else 0,
        estimated_work_units=20,
        abstained=abstained,
    )


def gold(*, abstain=False):
    return GraphRAGGoldCase(
        query_id="q1",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest=QUERY,
        relevant_nodes=(
            ()
            if abstain
            else (
                GraphNodeLocator("doc-a", 1, NODE_A),
                GraphNodeLocator("doc-b", 2, NODE_B),
            )
        ),
        required_edge_ids=() if abstain else (EDGE,),
        should_abstain=abstain,
    )


def test_perfect_selection_scores_nodes_documents_path_and_lineage():
    result = evaluate_graph_selection(selection(), gold())
    assert result.node_precision == result.node_recall == result.node_f1 == 1.0
    assert result.document_f1 == 1.0
    assert result.edge_f1 == 1.0
    assert result.complete_required_path is True
    assert result.lineage_completeness == 1.0
    assert result.abstention_correct is True
    assert len(result.evaluation_digest) == 64


def test_missing_node_and_path_reduce_recall_without_false_precision_penalty():
    result = evaluate_graph_selection(selection(include_b=False), gold())
    assert result.node_precision == 1.0
    assert result.node_recall == 0.5
    assert result.document_recall == 0.5
    assert result.edge_recall == 0.0
    assert result.complete_required_path is False
    assert result.lineage_completeness == 1.0


def test_correct_and_incorrect_abstention_are_explicit():
    correct = evaluate_graph_selection(selection(abstained=True), gold(abstain=True))
    assert correct.abstention_correct is True
    assert correct.node_f1 == 1.0
    incorrect = evaluate_graph_selection(selection(include_b=False), gold(abstain=True))
    assert incorrect.abstention_correct is False
    assert incorrect.node_precision == 0.0
    assert incorrect.node_recall == 0.0


def test_identity_mismatch_is_refused():
    wrong = GraphRAGGoldCase(
        query_id="q1",
        graph_set_id=SET_ID,
        graph_set_digest=SET_DIGEST,
        query_digest="e" * 64,
        relevant_nodes=(),
        required_edge_ids=(),
        should_abstain=True,
    )
    with pytest.raises(ValueError, match="identities differ"):
        evaluate_graph_selection(selection(), wrong)


def test_aggregate_is_macro_and_deterministic():
    first = evaluate_graph_selection(selection(), gold())
    second = evaluate_graph_selection(selection(include_b=False), gold())
    report = aggregate_graph_evaluations((first, second))
    assert report.case_count == 2
    assert report.macro_node_recall == 0.75
    assert report.complete_required_path_rate == 0.5
    assert report.abstention_accuracy == 1.0
    assert len(report.aggregate_digest) == 64


def test_strict_mapping_adapter_rejects_unknown_fields_and_abstention_support():
    payload = {
        "query_id": "q1",
        "graph_set_id": SET_ID,
        "graph_set_digest": SET_DIGEST,
        "query_digest": QUERY,
        "relevant_nodes": [
            {"doc_id": "doc-a", "generation": 1, "node_id": NODE_A}
        ],
        "required_edge_ids": [EDGE],
        "should_abstain": False,
    }
    assert gold_case_from_mapping(payload).relevant_nodes[0].doc_id == "doc-a"
    with pytest.raises(ValueError, match="schema"):
        gold_case_from_mapping({**payload, "raw_query": "private"})
    with pytest.raises(ValueError, match="may not contain"):
        GraphRAGGoldCase(
            query_id="q1",
            graph_set_id=SET_ID,
            graph_set_digest=SET_DIGEST,
            query_digest=QUERY,
            relevant_nodes=(GraphNodeLocator("doc-a", 1, NODE_A),),
            required_edge_ids=(),
            should_abstain=True,
        )
