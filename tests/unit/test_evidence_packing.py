from __future__ import annotations

import pytest

from tools.evidence_packing import (
    EvidencePackingCandidate,
    QualityCostPoint,
    choose_pareto_candidate,
    pack_evidence,
    pareto_frontier,
)


def test_packing_respects_budget_source_cap_risk_and_diversity():
    values = [
        EvidencePackingCandidate("a1", "a", 100, 0.95, redundancy_group="same"),
        EvidencePackingCandidate("a2", "a", 100, 0.94, redundancy_group="same"),
        EvidencePackingCandidate("b1", "b", 100, 0.80),
        EvidencePackingCandidate("risky", "c", 10, 1.0, retraction_risk=0.9),
    ]
    plan = pack_evidence(
        values,
        token_budget=200,
        max_per_source=1,
        max_retraction_risk=0.5,
        source_diversity_bonus=0.2,
    )
    assert plan.tokens_used == 200
    assert plan.source_count == 2
    assert {item.evidence_id for item in plan.selected} == {"a1", "b1"}
    assert plan.excluded_high_risk == ("risky",)
    assert plan.objective > 0.0


def test_redundancy_penalty_prefers_nonredundant_evidence_when_density_is_close():
    values = [
        EvidencePackingCandidate("a", "s1", 100, 0.9, redundancy_group="g"),
        EvidencePackingCandidate("b", "s2", 100, 0.85, redundancy_group="g"),
        EvidencePackingCandidate("c", "s3", 100, 0.80, redundancy_group="h"),
    ]
    plan = pack_evidence(
        values,
        token_budget=200,
        redundancy_penalty=0.5,
        source_diversity_bonus=0.0,
    )
    assert {item.evidence_id for item in plan.selected} == {"a", "c"}


def test_packing_rejects_duplicate_ids_and_invalid_candidates():
    duplicate = EvidencePackingCandidate("same", "a", 10, 0.5)
    with pytest.raises(ValueError, match="unique"):
        pack_evidence([duplicate, duplicate], token_budget=100)
    with pytest.raises(ValueError, match="EvidencePackingCandidate"):
        pack_evidence([object()], token_budget=100)


def test_pareto_frontier_removes_dominated_points_and_budget_selects_best():
    points = [
        QualityCostPoint("fast-good", 0.90, 5.0, 50.0),
        QualityCostPoint("dominated", 0.80, 6.0, 60.0),
        QualityCostPoint("expensive-best", 0.95, 20.0, 100.0),
        QualityCostPoint("cheap", 0.70, 1.0, 20.0),
    ]
    frontier = pareto_frontier(points)
    assert {item.candidate_id for item in frontier} == {
        "fast-good",
        "expensive-best",
        "cheap",
    }
    selected = choose_pareto_candidate(
        points,
        max_cost=10.0,
        max_latency_ms=80.0,
        min_quality=0.8,
    )
    assert selected is not None
    assert selected.candidate_id == "fast-good"
    assert choose_pareto_candidate(
        points,
        max_cost=0.5,
        max_latency_ms=10.0,
        min_quality=0.9,
    ) is None
