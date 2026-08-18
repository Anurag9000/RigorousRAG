from __future__ import annotations

import hashlib

import pytest

from evaluation.retrieval_interleaving import InterleavingOutcome, InterleavingSpec, RankedIdentity, build_team_draft_interleaving
from evaluation.retrieval_interleaving_promotion import InterleavingPromotionPolicy, build_interleaving_evidence, qualify_interleaving_experiment


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec() -> InterleavingSpec:
    return InterleavingSpec(sha("experiment"), sha("baseline"), sha("candidate"), max_positions=4)


def ranking(prefix: str):
    return tuple(RankedIdentity(f"{prefix}-{index}", f"source-{prefix}-{index}") for index in range(4))


def evidence_with_candidate_wins(count: int, *, ties: int = 0):
    experiment = spec()
    impressions = []
    outcomes = []
    for index in range(count + ties):
        impression = build_team_draft_interleaving(
            experiment,
            query_sha256=sha(f"query-{index}"),
            impression_index=index,
            ranking_a=ranking(f"a{index}"),
            ranking_b=ranking(f"b{index}"),
        )
        impressions.append(impression)
        if index < count:
            b_position = next(item.position for item in impression.items if item.contributed_by == "b")
            outcomes.append(InterleavingOutcome.build(impression, (b_position,)))
        else:
            outcomes.append(InterleavingOutcome.build(impression, ()))
    return experiment, tuple(impressions), tuple(outcomes)


def test_evidence_requires_one_outcome_for_every_exact_impression() -> None:
    experiment, impressions, outcomes = evidence_with_candidate_wins(2)
    with pytest.raises(ValueError, match="exactly one outcome"):
        build_interleaving_evidence(experiment, impressions, outcomes[:-1])


def test_candidate_promotion_requires_minimum_traffic_and_decisive_evidence() -> None:
    experiment, impressions, outcomes = evidence_with_candidate_wins(3)
    evidence = build_interleaving_evidence(experiment, impressions, outcomes)
    receipt = qualify_interleaving_experiment(
        experiment,
        evidence,
        policy=InterleavingPromotionPolicy(
            candidate_team="b",
            min_impressions=10,
            min_decisive=5,
            min_candidate_preference_rate=0.5,
            min_candidate_wilson_low=0.0,
            max_sign_test_p_value=1.0,
            max_tie_fraction=1.0,
        ),
    )
    assert not receipt.eligible
    assert "insufficient_impressions" in receipt.reason_codes
    assert "insufficient_decisive_comparisons" in receipt.reason_codes


def test_candidate_can_pass_when_randomized_evidence_is_decisive() -> None:
    experiment, impressions, outcomes = evidence_with_candidate_wins(12)
    evidence = build_interleaving_evidence(experiment, impressions, outcomes)
    receipt = qualify_interleaving_experiment(
        experiment,
        evidence,
        policy=InterleavingPromotionPolicy(
            candidate_team="b",
            min_impressions=10,
            min_decisive=10,
            min_candidate_preference_rate=0.6,
            min_candidate_wilson_low=0.5,
            max_sign_test_p_value=0.01,
            max_tie_fraction=0.2,
        ),
    )
    assert receipt.eligible
    assert receipt.candidate_policy_sha256 == experiment.policy_b_sha256
    assert receipt.candidate_wins == 12
    assert receipt.baseline_wins == 0


def test_tie_heavy_experiment_is_blocked_even_with_candidate_wins() -> None:
    experiment, impressions, outcomes = evidence_with_candidate_wins(2, ties=8)
    evidence = build_interleaving_evidence(experiment, impressions, outcomes)
    receipt = qualify_interleaving_experiment(
        experiment,
        evidence,
        policy=InterleavingPromotionPolicy(
            candidate_team="b",
            min_impressions=10,
            min_decisive=1,
            min_candidate_preference_rate=0.5,
            min_candidate_wilson_low=0.0,
            max_sign_test_p_value=1.0,
            max_tie_fraction=0.5,
        ),
    )
    assert not receipt.eligible
    assert "tie_fraction_exceeded" in receipt.reason_codes


def test_evidence_rejects_outcome_from_another_impression() -> None:
    experiment, impressions, outcomes = evidence_with_candidate_wins(2)
    other = build_team_draft_interleaving(
        experiment,
        query_sha256=sha("other-query"),
        impression_index=99,
        ranking_a=ranking("other-a"),
        ranking_b=ranking("other-b"),
    )
    mismatched = (InterleavingOutcome.build(other, (1,)), outcomes[1])
    with pytest.raises(ValueError, match="missing outcome|outside"):
        build_interleaving_evidence(experiment, impressions, mismatched)
