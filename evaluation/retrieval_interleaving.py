"""Randomized team-draft interleaving for retrieval-policy comparison.

Counterfactual snapshot diffs are descriptive.  This module adds a complementary
randomized comparison design that mixes two ranked lists for the *same* privacy-safe query
identity and records which policy contributed each displayed item.  Engagement can then
be credited to contributor teams and aggregated into preference rates with an exact
binomial sign test and Wilson interval.

The estimator does not assert causality beyond the stated randomization/measurement
assumptions; downstream code must still control experiment eligibility, traffic and user
safety.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 digest")
    selected = value.strip().lower()
    if len(selected) != 64 or any(ch not in "0123456789abcdef" for ch in selected):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return selected


def _identifier(value: str, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


@dataclass(frozen=True)
class InterleavingSpec:
    experiment_sha256: str
    policy_a_sha256: str
    policy_b_sha256: str
    max_positions: int = 10
    randomization_version: str = "team-draft-v1"

    def __post_init__(self) -> None:
        for name in ("experiment_sha256", "policy_a_sha256", "policy_b_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if self.policy_a_sha256 == self.policy_b_sha256:
            raise ValueError("interleaving policies must be distinct")
        if isinstance(self.max_positions, bool) or not isinstance(self.max_positions, int) or not 1 <= self.max_positions <= 1000:
            raise ValueError("max_positions must be in [1, 1000]")
        object.__setattr__(self, "randomization_version", _identifier(self.randomization_version, "randomization_version", 100))

    @property
    def spec_sha256(self) -> str:
        return _digest({"schema": "rigorousrag-interleaving-spec/v1", **asdict(self)})


@dataclass(frozen=True)
class RankedIdentity:
    item_id: str
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_id", _identifier(self.item_id, "item_id", 1000))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id", 1000))


@dataclass(frozen=True)
class InterleavedItem:
    position: int
    item: RankedIdentity
    contributed_by: str
    source_rank: int

    def __post_init__(self) -> None:
        if isinstance(self.position, bool) or not isinstance(self.position, int) or self.position < 1:
            raise ValueError("position must be positive")
        if not isinstance(self.item, RankedIdentity):
            raise ValueError("item must be RankedIdentity")
        if self.contributed_by not in {"a", "b"}:
            raise ValueError("contributed_by must be a or b")
        if isinstance(self.source_rank, bool) or not isinstance(self.source_rank, int) or self.source_rank < 1:
            raise ValueError("source_rank must be positive")


@dataclass(frozen=True)
class InterleavingImpression:
    spec_sha256: str
    query_sha256: str
    impression_index: int
    ranking_a_sha256: str
    ranking_b_sha256: str
    items: tuple[InterleavedItem, ...]
    impression_sha256: str

    def __post_init__(self) -> None:
        for name in ("spec_sha256", "query_sha256", "ranking_a_sha256", "ranking_b_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        if isinstance(self.impression_index, bool) or not isinstance(self.impression_index, int) or self.impression_index < 0:
            raise ValueError("impression_index must be non-negative")
        items = tuple(self.items)
        if not items:
            raise ValueError("interleaving impression must contain at least one item")
        if [item.position for item in items] != list(range(1, len(items) + 1)):
            raise ValueError("interleaved positions must be contiguous")
        if len({item.item.item_id for item in items}) != len(items):
            raise ValueError("interleaved impression may not contain duplicate item ids")
        object.__setattr__(self, "items", items)
        expected = _digest(self._payload())
        provided = _sha(self.impression_sha256, "impression_sha256")
        if expected != provided:
            raise ValueError("impression_sha256 does not match impression content")
        object.__setattr__(self, "impression_sha256", provided)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "rigorousrag-team-draft-impression/v1",
            "spec_sha256": self.spec_sha256,
            "query_sha256": self.query_sha256,
            "impression_index": self.impression_index,
            "ranking_a_sha256": self.ranking_a_sha256,
            "ranking_b_sha256": self.ranking_b_sha256,
            "items": [asdict(item) for item in self.items],
        }


def _ranking_digest(values: Sequence[RankedIdentity]) -> str:
    return _digest({"schema": "rigorousrag-interleaving-ranking/v1", "items": [asdict(value) for value in values]})


def _validate_ranking(values: Sequence[RankedIdentity], label: str) -> tuple[RankedIdentity, ...]:
    ranking = tuple(values)
    if not ranking or len(ranking) > 100_000:
        raise ValueError(f"{label} must be a non-empty bounded ranking")
    if any(not isinstance(item, RankedIdentity) for item in ranking):
        raise ValueError(f"{label} contains an invalid item")
    if len({item.item_id for item in ranking}) != len(ranking):
        raise ValueError(f"{label} contains duplicate item ids")
    return ranking


def build_team_draft_interleaving(
    spec: InterleavingSpec,
    *,
    query_sha256: str,
    impression_index: int,
    ranking_a: Sequence[RankedIdentity],
    ranking_b: Sequence[RankedIdentity],
) -> InterleavingImpression:
    """Create one deterministic replayable randomized team-draft impression."""

    if not isinstance(spec, InterleavingSpec):
        raise ValueError("spec must be InterleavingSpec")
    query_digest = _sha(query_sha256, "query_sha256")
    if isinstance(impression_index, bool) or not isinstance(impression_index, int) or impression_index < 0:
        raise ValueError("impression_index must be non-negative")
    left = _validate_ranking(ranking_a, "ranking_a")
    right = _validate_ranking(ranking_b, "ranking_b")
    ranking_a_digest = _ranking_digest(left)
    ranking_b_digest = _ranking_digest(right)
    seed = int(hashlib.sha256(f"{spec.spec_sha256}:{query_digest}:{impression_index}".encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    cursors = {"a": 0, "b": 0}
    rankings = {"a": left, "b": right}
    counts = {"a": 0, "b": 0}
    used: set[str] = set()
    output: list[InterleavedItem] = []

    def next_available(team: str) -> tuple[RankedIdentity, int] | None:
        ranking = rankings[team]
        while cursors[team] < len(ranking):
            index = cursors[team]
            cursors[team] += 1
            candidate = ranking[index]
            if candidate.item_id not in used:
                return candidate, index + 1
        return None

    while len(output) < spec.max_positions:
        available = [team for team in ("a", "b") if any(item.item_id not in used for item in rankings[team][cursors[team] :])]
        if not available:
            break
        if len(available) == 1:
            team = available[0]
        elif counts["a"] < counts["b"]:
            team = "a"
        elif counts["b"] < counts["a"]:
            team = "b"
        else:
            team = "a" if rng.random() < 0.5 else "b"
        selected = next_available(team)
        if selected is None:
            other = "b" if team == "a" else "a"
            selected = next_available(other)
            if selected is None:
                break
            team = other
        item, source_rank = selected
        used.add(item.item_id)
        counts[team] += 1
        output.append(InterleavedItem(len(output) + 1, item, team, source_rank))

    payload = {
        "schema": "rigorousrag-team-draft-impression/v1",
        "spec_sha256": spec.spec_sha256,
        "query_sha256": query_digest,
        "impression_index": impression_index,
        "ranking_a_sha256": ranking_a_digest,
        "ranking_b_sha256": ranking_b_digest,
        "items": [asdict(item) for item in output],
    }
    return InterleavingImpression(
        spec_sha256=spec.spec_sha256,
        query_sha256=query_digest,
        impression_index=impression_index,
        ranking_a_sha256=ranking_a_digest,
        ranking_b_sha256=ranking_b_digest,
        items=tuple(output),
        impression_sha256=_digest(payload),
    )


@dataclass(frozen=True)
class InterleavingOutcome:
    impression_sha256: str
    engaged_positions: tuple[int, ...]
    outcome_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "impression_sha256", _sha(self.impression_sha256, "impression_sha256"))
        positions = tuple(sorted(set(self.engaged_positions)))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > 1000 for value in positions):
            raise ValueError("engaged_positions contain an invalid position")
        object.__setattr__(self, "engaged_positions", positions)
        expected = _digest({"schema": "rigorousrag-interleaving-outcome/v1", "impression_sha256": self.impression_sha256, "engaged_positions": positions})
        provided = _sha(self.outcome_sha256, "outcome_sha256")
        if expected != provided:
            raise ValueError("outcome_sha256 does not match outcome content")
        object.__setattr__(self, "outcome_sha256", provided)

    @classmethod
    def build(cls, impression: InterleavingImpression, engaged_positions: Iterable[int]) -> "InterleavingOutcome":
        positions = tuple(sorted(set(engaged_positions)))
        if any(position > len(impression.items) for position in positions):
            raise ValueError("engagement references a position outside the impression")
        payload = {"schema": "rigorousrag-interleaving-outcome/v1", "impression_sha256": impression.impression_sha256, "engaged_positions": positions}
        return cls(impression.impression_sha256, positions, _digest(payload))


def preference_from_outcome(impression: InterleavingImpression, outcome: InterleavingOutcome) -> int:
    """Return +1 for A, -1 for B, and 0 for tied/no-credit outcomes."""

    if outcome.impression_sha256 != impression.impression_sha256:
        raise ValueError("outcome does not belong to impression")
    credits = {"a": 0, "b": 0}
    by_position = {item.position: item for item in impression.items}
    for position in outcome.engaged_positions:
        item = by_position.get(position)
        if item is None:
            raise ValueError("outcome contains an unavailable position")
        credits[item.contributed_by] += 1
    return 1 if credits["a"] > credits["b"] else -1 if credits["b"] > credits["a"] else 0


def _binomial_probability(n: int, k: int) -> float:
    return math.comb(n, k) * (0.5 ** n)


def _two_sided_sign_p_value(wins_a: int, wins_b: int) -> float:
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    observed = min(wins_a, wins_b)
    tail = sum(_binomial_probability(n, k) for k in range(0, observed + 1))
    return min(1.0, 2.0 * tail)


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(frozen=True)
class InterleavingAggregate:
    impression_count: int
    wins_a: int
    wins_b: int
    ties: int
    decisive_count: int
    preference_rate_a: float
    wilson_low: float
    wilson_high: float
    sign_test_p_value: float
    aggregate_sha256: str


def aggregate_interleaving_preferences(preferences: Iterable[int]) -> InterleavingAggregate:
    values = tuple(preferences)
    if not values or any(value not in {-1, 0, 1} for value in values):
        raise ValueError("preferences must be a non-empty sequence containing only -1, 0, 1")
    wins_a = sum(value == 1 for value in values)
    wins_b = sum(value == -1 for value in values)
    ties = len(values) - wins_a - wins_b
    decisive = wins_a + wins_b
    rate = wins_a / decisive if decisive else 0.5
    low, high = _wilson_interval(wins_a, decisive)
    p_value = _two_sided_sign_p_value(wins_a, wins_b)
    payload = {
        "schema": "rigorousrag-interleaving-aggregate/v1",
        "impression_count": len(values),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "decisive_count": decisive,
        "preference_rate_a": rate,
        "wilson_low": low,
        "wilson_high": high,
        "sign_test_p_value": p_value,
    }
    return InterleavingAggregate(len(values), wins_a, wins_b, ties, decisive, rate, low, high, p_value, _digest(payload))


__all__ = [
    "InterleavedItem",
    "InterleavingAggregate",
    "InterleavingImpression",
    "InterleavingOutcome",
    "InterleavingSpec",
    "RankedIdentity",
    "aggregate_interleaving_preferences",
    "build_team_draft_interleaving",
    "preference_from_outcome",
]
