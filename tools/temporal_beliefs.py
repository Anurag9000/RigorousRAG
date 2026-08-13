"""Temporal evidence revision with retractions, conflict detection, and source caps."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

_MAX_EVENTS = 100_000
_STANCES = {"support", "contradict", "retract"}


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    text = value.strip()
    if not text or len(text) > 500 or any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"{label} is invalid.")
    return text


def _unit(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be between 0 and 1.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return parsed


def _timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a timestamp or ISO-8601 datetime.")
    if isinstance(value, (int, float)):
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = float(text)
            if math.isfinite(parsed):
                return parsed
        except ValueError:
            pass
        try:
            parsed_dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} must be a timestamp or ISO-8601 datetime.") from exc
        if parsed_dt.tzinfo is None:
            parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
        return parsed_dt.timestamp()
    raise ValueError(f"{label} must be a timestamp or ISO-8601 datetime.")


@dataclass(frozen=True)
class BeliefEvent:
    claim_id: str
    evidence_id: str
    source_id: str
    stance: Literal["support", "contradict", "retract"]
    confidence: float
    observed_at: float | str
    retracts_evidence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _identifier(self.claim_id, "claim_id"))
        object.__setattr__(self, "evidence_id", _identifier(self.evidence_id, "evidence_id"))
        object.__setattr__(self, "source_id", _identifier(self.source_id, "source_id"))
        if self.stance not in _STANCES:
            raise ValueError("stance must be support, contradict, or retract.")
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed_at"))
        if self.stance == "retract":
            if self.retracts_evidence_id is None:
                raise ValueError("retract events require retracts_evidence_id.")
            object.__setattr__(self, "retracts_evidence_id", _identifier(self.retracts_evidence_id, "retracts_evidence_id"))
        elif self.retracts_evidence_id is not None:
            raise ValueError("only retract events may set retracts_evidence_id.")


@dataclass(frozen=True)
class BeliefState:
    claim_id: str
    status: Literal["supported", "contradicted", "conflicted", "unknown"]
    support_score: float
    contradiction_score: float
    confidence: float
    active_evidence_ids: tuple[str, ...]
    retracted_evidence_ids: tuple[str, ...]
    independent_sources: tuple[str, ...]
    as_of: float


def revise_belief(
    claim_id: str,
    events: Iterable[BeliefEvent],
    *,
    as_of: float | str | None = None,
    half_life_seconds: float | None = None,
    conflict_margin: float = 0.15,
) -> BeliefState:
    """Recompute a claim from active evidence with optional recency decay.

    Repeated evidence from one source is capped at that source's strongest active
    support and contradiction contribution so mirrors/duplicates cannot dominate.
    """

    claim = _identifier(claim_id, "claim_id")
    selected_as_of = _timestamp(as_of, "as_of") if as_of is not None else datetime.now(timezone.utc).timestamp()
    margin = _unit(conflict_margin, "conflict_margin")
    if half_life_seconds is not None:
        if isinstance(half_life_seconds, bool):
            raise ValueError("half_life_seconds must be finite and positive.")
        try:
            half_life = float(half_life_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("half_life_seconds must be finite and positive.") from exc
        if not math.isfinite(half_life) or half_life <= 0.0:
            raise ValueError("half_life_seconds must be finite and positive.")
    else:
        half_life = None
    rows: list[BeliefEvent] = []
    try:
        iterator = iter(events)
    except Exception as exc:
        raise ValueError("events must be safely iterable.") from exc
    for event in iterator:
        if len(rows) >= _MAX_EVENTS:
            raise ValueError("events exceeds the event limit.")
        if not isinstance(event, BeliefEvent):
            raise ValueError("events contains an invalid value.")
        if event.claim_id == claim and event.observed_at <= selected_as_of:
            rows.append(event)
    rows.sort(key=lambda event: (event.observed_at, event.evidence_id))
    retracted: set[str] = set()
    for event in rows:
        if event.stance == "retract" and event.retracts_evidence_id is not None:
            retracted.add(event.retracts_evidence_id)
    active = [event for event in rows if event.stance != "retract" and event.evidence_id not in retracted]
    per_source_support: dict[str, float] = {}
    per_source_contradiction: dict[str, float] = {}
    for event in active:
        weight = event.confidence
        if half_life is not None:
            age = max(0.0, selected_as_of - event.observed_at)
            weight *= math.exp(-math.log(2.0) * age / half_life)
        target = per_source_support if event.stance == "support" else per_source_contradiction
        target[event.source_id] = max(target.get(event.source_id, 0.0), weight)
    support = 1.0 - math.prod(1.0 - value for value in per_source_support.values()) if per_source_support else 0.0
    contradiction = 1.0 - math.prod(1.0 - value for value in per_source_contradiction.values()) if per_source_contradiction else 0.0
    total = support + contradiction
    confidence = abs(support - contradiction) / total if total > 0.0 else 0.0
    if total == 0.0:
        status: Literal["supported", "contradicted", "conflicted", "unknown"] = "unknown"
    elif support > 0.0 and contradiction > 0.0 and abs(support - contradiction) <= margin:
        status = "conflicted"
    elif support >= contradiction:
        status = "supported"
    else:
        status = "contradicted"
    return BeliefState(
        claim_id=claim,
        status=status,
        support_score=support,
        contradiction_score=contradiction,
        confidence=confidence,
        active_evidence_ids=tuple(sorted(event.evidence_id for event in active)),
        retracted_evidence_ids=tuple(sorted(retracted)),
        independent_sources=tuple(sorted({event.source_id for event in active})),
        as_of=selected_as_of,
    )


__all__ = ["BeliefEvent", "BeliefState", "revise_belief"]
