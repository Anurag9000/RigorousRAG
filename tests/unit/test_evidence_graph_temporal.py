from __future__ import annotations

import pytest

from tools.evidence_graph_temporal import (
    build_scientific_hyperedge,
    project_hyperedges_for_gnn,
    propagate_retraction_risk,
    temporal_evidence_status,
)
from tools.evidence_graph_types import (
    EvidenceEdge,
    EvidenceNode,
    deterministic_edge_id,
    deterministic_node_id,
)


def node(kind: str, key: str, *, metadata=None):
    return EvidenceNode(
        node_id=deterministic_node_id(
            owner_id="alice",
            doc_id="doc-1",
            generation=1,
            node_type=kind,
            natural_key=key,
        ),
        owner_id="alice",
        doc_id="doc-1",
        generation=1,
        node_type=kind,
        natural_key=key,
        label=key,
        metadata=metadata or {},
    )


def edge(source, target, kind: str, key: str):
    return EvidenceEdge(
        edge_id=deterministic_edge_id(
            owner_id="alice",
            doc_id="doc-1",
            generation=1,
            source_node_id=source.node_id,
            target_node_id=target.node_id,
            edge_type=kind,
            relation_key=key,
        ),
        owner_id="alice",
        doc_id="doc-1",
        generation=1,
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_type=kind,
        relation_key=key,
    )


def test_temporal_status_uses_only_explicit_validity_and_retraction_metadata():
    future = node("claim", "future", metadata={"valid_from": 20.0})
    expired = node("claim", "expired", metadata={"valid_from": 1.0, "valid_to": 5.0})
    retracted = node("claim", "retracted", metadata={"retracted_at": 7.0})
    active = node("claim", "active")

    assert temporal_evidence_status(future, as_of=10).status == "not_yet_valid"
    assert temporal_evidence_status(expired, as_of=10).status == "expired"
    assert temporal_evidence_status(retracted, as_of=10).status == "retracted"
    assert temporal_evidence_status(active, as_of=10).status == "active"
    assert len(temporal_evidence_status(active, as_of=10).status_digest) == 64


def test_invalid_temporal_interval_fails_closed():
    invalid = node("claim", "invalid", metadata={"valid_from": 10.0, "valid_to": 5.0})
    with pytest.raises(ValueError, match="precede"):
        temporal_evidence_status(invalid, as_of=7.0)


def test_retraction_risk_propagates_backwards_only_across_dependencies():
    dataset = node("dataset", "dataset", metadata={"retracted_at": 5.0})
    claim = node("claim", "claim")
    derived = node("claim", "derived")
    unrelated = node("claim", "unrelated")
    support = node("claim", "support")
    edges = [
        edge(claim, dataset, "uses_dataset", "claim-uses-dataset"),
        edge(derived, claim, "derived_from", "derived-from-claim"),
        edge(support, dataset, "supports", "explicit-support"),
    ]

    risks = propagate_retraction_risk(
        [dataset, claim, derived, unrelated, support],
        edges,
        as_of=10.0,
        max_hops=4,
        decay=0.5,
    )

    assert risks[dataset.node_id].distance == 0
    assert risks[dataset.node_id].risk == 1.0
    assert risks[claim.node_id].distance == 1
    assert risks[claim.node_id].risk == pytest.approx(0.5)
    assert risks[derived.node_id].distance == 2
    assert risks[derived.node_id].risk == pytest.approx(0.25)
    assert unrelated.node_id not in risks
    assert support.node_id not in risks


def test_scientific_hyperedge_is_deterministic_and_projects_for_gnn():
    claim = node("claim", "claim")
    method = node("method", "method")
    dataset = node("dataset", "dataset")
    result = node("claim", "result")
    roles = {
        "claim": [claim.node_id],
        "method": [method.node_id],
        "dataset": [dataset.node_id],
        "result": [result.node_id],
    }
    first = build_scientific_hyperedge(
        relation_type="experiment",
        roles=roles,
        weight=0.9,
    )
    second = build_scientific_hyperedge(
        relation_type="experiment",
        roles={key: list(reversed(value)) for key, value in roles.items()},
        weight=0.9,
    )
    assert first.hyperedge_id == second.hyperedge_id
    assert first.members == tuple(sorted([claim.node_id, method.node_id, dataset.node_id, result.node_id]))
    assert len(first.provenance_digest) == 64

    projected = project_hyperedges_for_gnn([first])
    assert len(projected) == 6
    assert all(item.hyperedge_id == first.hyperedge_id for item in projected)
    assert all(item.weight == pytest.approx(0.3) for item in projected)
    assert len({(item.left_node_id, item.right_node_id) for item in projected}) == 6


def test_hyperedge_rejects_unknown_roles_relations_and_self_only_membership():
    claim = node("claim", "claim")
    with pytest.raises(ValueError, match="unsupported"):
        build_scientific_hyperedge(
            relation_type="invented",
            roles={"claim": [claim.node_id], "result": [claim.node_id]},
        )
    with pytest.raises(ValueError, match="unsupported"):
        build_scientific_hyperedge(
            relation_type="experiment",
            roles={"unknown": [claim.node_id], "result": [claim.node_id]},
        )
    with pytest.raises(ValueError, match="member count"):
        build_scientific_hyperedge(
            relation_type="experiment",
            roles={"claim": [claim.node_id]},
        )
