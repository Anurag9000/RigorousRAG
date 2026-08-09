from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tools.evidence_graph_builder import (
    ExplicitGraphRelation,
    GraphAnnotation,
    build_evidence_graph,
)
from tools.evidence_graph_retrieval import search_nodes, search_nodes_governed

PROFILE = "b" * 64


def document():
    text = "Finalized temporal evidence document."
    return SimpleNamespace(
        id="doc-temporal",
        title="Temporal evidence",
        filename="temporal.pdf",
        text=text,
        metadata={"content_sha256": hashlib.sha256(text.encode()).hexdigest()},
        sections=[
            SimpleNamespace(
                title="Results",
                content="Several outcome claims are compared.",
                page_number=1,
                metadata={},
            )
        ],
    )


def governed_graph():
    return build_evidence_graph(
        document(),
        owner_id="alice",
        generation=7,
        profile_fingerprint=PROFILE,
        annotations=(
            GraphAnnotation(
                "dataset-retracted",
                "dataset",
                "Retracted registry",
                section_index=0,
                metadata={"retracted_at": 5.0},
            ),
            GraphAnnotation(
                "claim-dependent",
                "claim",
                "Dependent outcome improved",
                section_index=0,
            ),
            GraphAnnotation(
                "claim-derived",
                "claim",
                "Derived outcome improved",
                section_index=0,
            ),
            GraphAnnotation(
                "claim-independent",
                "claim",
                "Independent outcome improved",
                section_index=0,
            ),
            GraphAnnotation(
                "claim-future",
                "claim",
                "Future outcome improved",
                section_index=0,
                metadata={"valid_from": 20.0},
            ),
            GraphAnnotation(
                "claim-expired",
                "claim",
                "Expired outcome improved",
                section_index=0,
                metadata={"valid_to": 8.0},
            ),
        ),
        relations=(
            ExplicitGraphRelation(
                "dependent-uses-retracted-dataset",
                "claim-dependent",
                "dataset-retracted",
                "uses_dataset",
            ),
            ExplicitGraphRelation(
                "derived-from-dependent",
                "claim-derived",
                "claim-dependent",
                "derived_from",
            ),
        ),
        now=1.0,
    )


def by_key(batch, key):
    return next(node for node in batch.nodes if node.natural_key == f"annotation:{key}")


def test_governed_search_excludes_inactive_and_any_retraction_risk_by_default():
    batch = governed_graph()
    legacy = search_nodes(
        batch,
        "outcome improved",
        node_types=("claim",),
        limit=20,
    )
    assert {item.node.natural_key for item in legacy} == {
        "annotation:claim-dependent",
        "annotation:claim-derived",
        "annotation:claim-independent",
        "annotation:claim-future",
        "annotation:claim-expired",
    }

    governed = search_nodes_governed(
        batch,
        "outcome improved",
        as_of=10.0,
        node_types=("claim",),
    )
    assert [item.node.natural_key for item in governed] == [
        "annotation:claim-independent"
    ]
    assert governed[0].temporal_status == "active"
    assert governed[0].retraction_risk == 0.0


def test_exclusion_threshold_uses_conservative_propagated_dependency_risk():
    batch = governed_graph()
    governed = search_nodes_governed(
        batch,
        "outcome improved",
        as_of=10.0,
        node_types=("claim",),
        max_retraction_risk=0.50,
    )
    by_key_result = {item.node.natural_key: item for item in governed}
    assert set(by_key_result) == {
        "annotation:claim-derived",
        "annotation:claim-independent",
    }
    derived = by_key_result["annotation:claim-derived"]
    assert derived.retraction_risk == pytest.approx(0.49)
    assert derived.retracted_source_ids == (
        by_key(batch, "dataset-retracted").node_id,
    )
    assert "annotation:claim-dependent" not in by_key_result


def test_penalty_policy_retains_active_risky_nodes_but_reranks_them():
    batch = governed_graph()
    governed = search_nodes_governed(
        batch,
        "outcome improved",
        as_of=10.0,
        node_types=("claim",),
        retraction_policy="penalize",
        risk_penalty=1.0,
    )
    names = [item.node.natural_key for item in governed]
    assert names == [
        "annotation:claim-independent",
        "annotation:claim-derived",
        "annotation:claim-dependent",
    ]
    values = {item.node.natural_key: item for item in governed}
    assert values["annotation:claim-dependent"].retraction_risk == pytest.approx(0.70)
    assert values["annotation:claim-derived"].retraction_risk == pytest.approx(0.49)
    assert values["annotation:claim-dependent"].adjusted_score < values[
        "annotation:claim-dependent"
    ].lexical_score
    assert values["annotation:claim-derived"].adjusted_score < values[
        "annotation:claim-derived"
    ].lexical_score
    assert values["annotation:claim-independent"].adjusted_score == values[
        "annotation:claim-independent"
    ].lexical_score


def test_governed_search_is_deterministic_and_requires_explicit_valid_controls():
    batch = governed_graph()
    first = search_nodes_governed(
        batch,
        "outcome improved",
        as_of=10.0,
        node_types=("claim",),
        retraction_policy="penalize",
        risk_penalty=0.5,
    )
    second = search_nodes_governed(
        batch,
        "outcome improved",
        as_of=10.0,
        node_types=("claim",),
        retraction_policy="penalize",
        risk_penalty=0.5,
    )
    assert first == second
    with pytest.raises(ValueError, match="retraction_policy"):
        search_nodes_governed(batch, "outcome", as_of=10.0, retraction_policy="ignore")
    with pytest.raises(ValueError, match="between 0 and 1"):
        search_nodes_governed(batch, "outcome", as_of=10.0, max_retraction_risk=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        search_nodes_governed(batch, "outcome", as_of=-1.0)
