"""Executable owner-isolated entity and interval/evidence contracts.

These tests exercise deterministic source behavior; no external resolver,
production entity corpus, or timestamp authority is assumed.
"""
from __future__ import annotations

import datetime as dt

import pytest

from tools.entity_temporal import (
    Entity, EntityMention, EntityRegistry, TemporalEvidence, TemporalInterval,
    TemporalPoint, filter_as_of, freshness_score, normalize_entity_text,
)


def _mention(text="Atlas", *, owner="owner-a", hint="other", identifier="mention-1"):
    return EntityMention(
        mention_id=identifier,
        owner_id=owner,
        observed_text=text,
        source_id="source-1",
        start=0,
        end=len(text),
        kind_hint=hint,
    )


def _date(value):
    return TemporalPoint.parse(value)


def _interval(start, end, *, role="validity", inclusive_start=True, inclusive_end=True):
    return TemporalInterval(
        _date(start), _date(end), role,
        inclusive_start=inclusive_start, inclusive_end=inclusive_end,
    )


def test_entity_normalization_and_identity_are_stable_under_alias_order():
    assert normalize_entity_text("  AＴLAS  Project! ") == "atlas project"
    first = Entity(
        "owner-a", "Atlas", "organization", aliases=("Lab", "Research Centre"),
        external_ids={"doi": "xyz", "project": "007"},
    )
    second = Entity(
        "owner-a", "ATLAS", "organization", aliases=("Research Centre", "Lab"),
        external_ids={"project": "007", "doi": "xyz"},
    )
    assert first.entity_id == second.entity_id
    assert first.aliases != second.aliases


def test_entities_are_owner_isolated_even_when_same_canonical_name():
    own = Entity("owner-a", "Atlas", "organization")
    other = Entity("owner-b", "Atlas", "organization")
    reg = EntityRegistry()
    assert reg.register(own) != reg.register(other)
    assert reg.get("owner-a", own.entity_id) == own
    with pytest.raises(PermissionError):
        reg.get("owner-a", other.entity_id)
    assert reg.resolve(_mention(owner="owner-b")).selected_entity_id == other.entity_id


def test_exact_alias_resolution_respects_explicit_kind_hint():
    reg = EntityRegistry()
    method = Entity("owner-a", "Atlas Algorithm", "method", aliases=("Atlas",))
    organization = Entity("owner-a", "Atlas Institute", "organization", aliases=("Atlas",))
    reg.register(method)
    reg.register(organization)
    hit = reg.resolve(_mention(hint="organization"))
    assert hit.selected_entity_id == organization.entity_id
    assert tuple(item.entity_id for item in hit.candidates) == (organization.entity_id,)
    none = reg.resolve(_mention(hint="person"))
    assert none.selected_entity_id is None


def test_ambiguous_exact_alias_does_not_force_false_canonical_identity():
    reg = EntityRegistry()
    a = Entity("owner-a", "Atlas Research", "organization", aliases=("Atlas",))
    b = Entity("owner-a", "Atlas Lab", "organization", aliases=("Atlas",))
    reg.register(a)
    reg.register(b)
    result = reg.resolve(_mention(), threshold=0.85, ambiguity_margin=0.01)
    assert result.ambiguous
    assert result.selected_entity_id is None
    assert all(c.score == 1 for c in result.candidates)
    assert {c.entity_id for c in result.candidates} == {a.entity_id, b.entity_id}


def test_fuzzy_candidate_retrieval_is_deterministic_and_thresholded():
    reg = EntityRegistry()
    organization = Entity("owner-a", "Atlas Alpine Institute", "organization")
    reg.register(organization)
    low = reg.resolve(_mention("Atlas Institute", hint="organization"), threshold=0.9)
    assert low.selected_entity_id is None
    assert low.candidates[0].score == pytest.approx(2 / 3)
    accepted = reg.resolve(_mention("Atlas Institute", hint="organization"), threshold=0.5)
    assert accepted.selected_entity_id == organization.entity_id


@pytest.mark.parametrize("behavior", ("raises", "wrong-length"))
def test_provider_failure_or_cardinality_mismatch_is_truthfully_deterministic(behavior):
    reg = EntityRegistry()
    entity = Entity("owner-a", "Atlas", "organization")
    reg.register(entity)

    class BrokenProvider:
        def rank(self, mention, candidates):
            if behavior == "raises":
                raise RuntimeError("provider offline")
            return (0.4, 0.5)

    result = reg.resolve(_mention(), provider=BrokenProvider())
    assert result.selected_entity_id == entity.entity_id
    assert result.candidates[0].method == "deterministic"
    assert result.candidates[0].score == 1.0


def test_provider_valid_scores_are_labeled_provider_and_used_for_threshold():
    reg = EntityRegistry()
    entity = Entity("owner-a", "Atlas", "organization")
    reg.register(entity)

    class ConservativeProvider:
        def rank(self, mention, candidates):
            return (0.4,)

    result = reg.resolve(_mention(), provider=ConservativeProvider())
    assert result.selected_entity_id is None
    assert result.candidates[0].method == "provider"
    assert result.candidates[0].score == pytest.approx(0.4)


def test_provider_invalid_probability_is_rejected_not_used_for_promotion():
    reg = EntityRegistry()
    reg.register(Entity("owner-a", "Atlas", "organization"))

    class BadProvider:
        def rank(self, mention, candidates):
            return (1.2,)

    with pytest.raises(ValueError, match="provider score"):
        reg.resolve(_mention(), provider=BadProvider())


def test_reusing_entity_id_with_conflicting_observed_aliases_fails_closed():
    reg = EntityRegistry()
    reg.register(Entity("owner-a", "Atlas", "organization", aliases=("One",)))
    with pytest.raises(ValueError, match="identity collision"):
        reg.register(Entity("owner-a", "Atlas", "organization", aliases=("Two",)))


@pytest.mark.parametrize("kwargs", (
    {"kind": "unsupported"},
    {"canonical_name": ""},
    {"aliases": tuple("a" for _ in range(257))},
))
def test_invalid_entities_are_rejected(kwargs):
    defaults = {"owner_id": "owner-a", "canonical_name": "Atlas", "kind": "organization"}
    with pytest.raises(ValueError):
        Entity(**(defaults | kwargs))


@pytest.mark.parametrize("start,end", ((-1, 2), (3, 3), (4, 3), (True, 1)))
def test_invalid_mention_offsets_are_rejected(start, end):
    with pytest.raises(ValueError, match="offsets"):
        EntityMention("m", "owner-a", "atlas", "source", start, end)


@pytest.mark.parametrize("literal,precision", (
    ("2025", "year"),
    ("2025-02", "month"),
    ("2025-02-03", "day"),
    ("2025-02-03T09:10:11Z", "second"),
))
def test_temporal_precision_is_explicit_and_parsing_is_utc(literal, precision):
    point = _date(literal)
    assert point.precision == precision
    assert point.value.tzinfo == dt.timezone.utc


def test_temporal_offsets_normalize_to_absolute_instant():
    from_offset = _date("2025-02-03T11:10:11+02:00")
    from_utc = _date("2025-02-03T09:10:11Z")
    assert from_offset == from_utc


@pytest.mark.parametrize("literal", ("bad", "2025-13", "2025-02-30"))
def test_invalid_temporal_points_rejected(literal):
    with pytest.raises(ValueError):
        _date(literal)


def test_interval_contains_respects_open_and_closed_endpoints():
    value = _interval("2025-01-01", "2025-01-03", inclusive_end=False)
    assert value.contains(_date("2025-01-01"))
    assert value.contains(_date("2025-01-02"))
    assert not value.contains(_date("2025-01-03"))
    assert not value.contains(_date("2024-12-31"))


def test_adjacent_intervals_disjoint_when_either_touching_boundary_is_open():
    left = _interval("2025-01-01", "2025-01-02", inclusive_end=False)
    right = _interval("2025-01-02", "2025-01-04")
    assert not left.overlaps(right)
    assert not right.overlaps(left)
    assert left.relation(right) == "before"
    assert right.relation(left) == "after"

    closed = _interval("2025-01-01", "2025-01-02")
    assert closed.overlaps(right) and right.overlaps(closed)
    assert closed.relation(right) == "overlaps"


@pytest.mark.parametrize("left,right,expected", (
    (("2025-01-01", "2025-01-02"), ("2025-01-04", "2025-01-05"), "before"),
    (("2025-01-05", "2025-01-06"), ("2025-01-01", "2025-01-04"), "after"),
    (("2025-01-01", "2025-01-05"), ("2025-01-01", "2025-01-05"), "equal"),
    (("2025-01-02", "2025-01-04"), ("2025-01-01", "2025-01-05"), "during"),
    (("2025-01-01", "2025-01-05"), ("2025-01-02", "2025-01-04"), "contains"),
    (("2025-01-01", "2025-01-04"), ("2025-01-03", "2025-01-05"), "overlaps"),
))
def test_interval_relation_covers_all_six_temporal_categories(left, right, expected):
    assert _interval(*left).relation(_interval(*right)) == expected


def test_evidence_as_of_filters_future_retracted_and_superseded_records():
    interval = _interval("2024-01-01", "2026-12-31")
    records = (
        TemporalEvidence("current", interval, publication=_date("2025-01-01")),
        TemporalEvidence("future", interval, publication=_date("2027-01-01")),
        TemporalEvidence("retracted", interval, retracted_at=_date("2025-06-01")),
        TemporalEvidence("superseded", interval, superseded_at=_date("2025-05-01")),
    )
    valid = filter_as_of(records, _date("2025-10-01"))
    assert tuple(item.evidence_id for item in valid) == ("current",)


def test_evidence_retraction_inclusive_boundary_invalidates_record():
    record = TemporalEvidence(
        "retracted", _interval("2025-01-01", "2025-12-31"),
        retracted_at=_date("2025-07-01"),
    )
    assert record.valid_as_of(_date("2025-06-30"))
    assert not record.valid_as_of(_date("2025-07-01"))


def test_freshness_half_life_and_future_publications_are_bounded():
    publication = _date("2025-01-01")
    assert freshness_score(publication, as_of=publication) == pytest.approx(1)
    assert freshness_score(publication, as_of=_date("2026-01-01")) == pytest.approx(0.5)
    assert freshness_score(_date("2027-01-01"), as_of=publication) == pytest.approx(1)


@pytest.mark.parametrize("half_life", (True, 0.0, -3, float("nan"), float("inf")))
def test_bad_half_life_fails_closed(half_life):
    with pytest.raises(ValueError, match="half_life_days"):
        freshness_score(_date("2025-01-01"), as_of=_date("2025-03-01"), half_life_days=half_life)
