from __future__ import annotations

import pytest

from tools.evidence_graph_temporal_replay import replay_temporal_evidence
from tools.evidence_graph_types import EvidenceNode, deterministic_node_id


def node(key: str, metadata):
    return EvidenceNode(
        node_id=deterministic_node_id(
            owner_id="alice",
            doc_id="doc-1",
            generation=1,
            node_type="claim",
            natural_key=key,
        ),
        owner_id="alice",
        doc_id="doc-1",
        generation=1,
        node_type="claim",
        natural_key=key,
        label=key,
        metadata=metadata,
    )


def test_temporal_replay_records_explicit_state_transitions_and_digests():
    future = node("future", {"valid_from": 5.0, "valid_to": 9.0})
    retracted = node("retracted", {"retracted_at": 7.0})
    report = replay_temporal_evidence(
        [future, retracted],
        as_of_points=[1.0, 6.0, 8.0, 10.0],
    )
    assert len(report.frames) == 4
    assert future.node_id in report.frames[0].not_yet_valid_node_ids
    assert future.node_id in report.frames[1].active_node_ids
    assert retracted.node_id in report.frames[2].retracted_node_ids
    assert future.node_id in report.frames[3].expired_node_ids
    transitions = {
        (item.node_id, item.from_status, item.to_status, item.at)
        for item in report.transitions
    }
    assert (future.node_id, "not_yet_valid", "active", 6.0) in transitions
    assert (retracted.node_id, "active", "retracted", 8.0) in transitions
    assert (future.node_id, "active", "expired", 10.0) in transitions
    assert all(len(frame.frame_digest) == 64 for frame in report.frames)
    assert len(report.report_digest) == 64


def test_temporal_replay_requires_strictly_increasing_unique_points():
    value = node("claim", {})
    with pytest.raises(ValueError, match="strictly increasing"):
        replay_temporal_evidence([value], as_of_points=[2.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        replay_temporal_evidence([value], as_of_points=[1.0, 1.0])
