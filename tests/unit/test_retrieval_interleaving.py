from __future__ import annotations

import hashlib

import pytest

from evaluation.retrieval_interleaving import (
    InterleavingOutcome,
    InterleavingSpec,
    RankedIdentity,
    aggregate_interleaving_preferences,
    build_team_draft_interleaving,
    preference_from_outcome,
)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def spec() -> InterleavingSpec:
    return InterleavingSpec(sha("experiment"), sha("policy-a"), sha("policy-b"), max_positions=6)


def ranking(prefix: str, values: tuple[str, ...]):
    return tuple(RankedIdentity(item_id=value, source_id=f"source:{value}") for value in values)


def test_team_draft_is_deterministically_replayable_for_same_impression_identity() -> None:
    left = ranking("a", ("a", "shared", "c", "d"))
    right = ranking("b", ("b", "shared", "e", "f"))
    first = build_team_draft_interleaving(spec(), query_sha256=sha("query"), impression_index=7, ranking_a=left, ranking_b=right)
    second = build_team_draft_interleaving(spec(), query_sha256=sha("query"), impression_index=7, ranking_a=left, ranking_b=right)
    assert first == second
    assert first.impression_sha256 == second.impression_sha256
    assert len({item.item.item_id for item in first.items}) == len(first.items)


def test_team_draft_keeps_team_contribution_counts_balanced_when_both_have_inventory() -> None:
    left = ranking("a", tuple(f"a-{index}" for index in range(10)))
    right = ranking("b", tuple(f"b-{index}" for index in range(10)))
    impression = build_team_draft_interleaving(spec(), query_sha256=sha("balanced-query"), impression_index=0, ranking_a=left, ranking_b=right)
    counts = {"a": 0, "b": 0}
    for item in impression.items:
        counts[item.contributed_by] += 1
    assert abs(counts["a"] - counts["b"]) <= 1


def test_shared_items_appear_once_even_when_both_policies_rank_them() -> None:
    left = ranking("a", ("shared", "a-only", "tail-a"))
    right = ranking("b", ("shared", "b-only", "tail-b"))
    impression = build_team_draft_interleaving(spec(), query_sha256=sha("shared-query"), impression_index=1, ranking_a=left, ranking_b=right)
    ids = [item.item.item_id for item in impression.items]
    assert ids.count("shared") == 1


def test_outcome_credit_is_bound_to_exact_impression_and_contributor() -> None:
    left = ranking("a", ("a-1", "a-2", "a-3"))
    right = ranking("b", ("b-1", "b-2", "b-3"))
    impression = build_team_draft_interleaving(spec(), query_sha256=sha("credit-query"), impression_index=2, ranking_a=left, ranking_b=right)
    a_position = next(item.position for item in impression.items if item.contributed_by == "a")
    outcome = InterleavingOutcome.build(impression, (a_position,))
    assert preference_from_outcome(impression, outcome) == 1

    other = build_team_draft_interleaving(spec(), query_sha256=sha("credit-query"), impression_index=3, ranking_a=left, ranking_b=right)
    with pytest.raises(ValueError, match="does not belong"):
        preference_from_outcome(other, outcome)


def test_outcome_rejects_position_outside_displayed_impression() -> None:
    impression = build_team_draft_interleaving(
        spec(),
        query_sha256=sha("bounds-query"),
        impression_index=0,
        ranking_a=ranking("a", ("a",)),
        ranking_b=ranking("b", ("b",)),
    )
    with pytest.raises(ValueError, match="outside"):
        InterleavingOutcome.build(impression, (99,))


def test_aggregate_preference_reports_ties_interval_and_exact_sign_test() -> None:
    result = aggregate_interleaving_preferences((1, 1, 1, 1, 1, -1, 0, 0))
    assert result.impression_count == 8
    assert result.wins_a == 5
    assert result.wins_b == 1
    assert result.ties == 2
    assert result.decisive_count == 6
    assert 0.0 <= result.wilson_low <= result.preference_rate_a <= result.wilson_high <= 1.0
    assert 0.0 <= result.sign_test_p_value <= 1.0


def test_all_ties_are_represented_as_no_preference_evidence() -> None:
    result = aggregate_interleaving_preferences((0, 0, 0))
    assert result.decisive_count == 0
    assert result.preference_rate_a == 0.5
    assert result.sign_test_p_value == 1.0
    assert (result.wilson_low, result.wilson_high) == (0.0, 1.0)
