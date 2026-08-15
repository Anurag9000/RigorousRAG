"""Owner-scoped entity normalization and temporal evidence reasoning.

The module preserves observed aliases and timestamps while attaching canonical IDs and
explicit temporal roles.  Matching is deterministic by default; learned/entity-linker
implementations can be injected without mutating source evidence.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from tools.security import normalize_owner_id

_MAX_ENTITIES = 100_000
_MAX_ALIASES = 256
_MAX_CANDIDATES = 32
_ENTITY_KINDS = frozenset({
    "person", "organization", "institution", "dataset", "method", "model", "condition",
    "intervention", "outcome", "location", "material", "chemical", "quantity", "unit",
    "basin", "river", "reservoir", "study", "other",
})
_TEMPORAL_ROLES = frozenset({"publication", "study_period", "event", "version", "retraction", "observation", "validity", "query_as_of"})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _probability(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a probability")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a probability") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} is outside [0,1]")
    return result


def normalize_entity_text(value: str) -> str:
    text = _text(value, "entity text", 1000)
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"[^\w\s.+/-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


@dataclass(frozen=True)
class Entity:
    owner_id: str
    canonical_name: str
    kind: str
    aliases: tuple[str, ...] = ()
    external_ids: Mapping[str, str] = field(default_factory=dict)
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "canonical_name", _text(self.canonical_name, "canonical_name"))
        kind = _text(self.kind, "kind", 64).lower()
        if kind not in _ENTITY_KINDS:
            raise ValueError("unsupported entity kind")
        object.__setattr__(self, "kind", kind)
        if len(self.aliases) > _MAX_ALIASES:
            raise ValueError("aliases exceed the item limit")
        aliases = tuple(dict.fromkeys(_text(item, "alias", 1000) for item in self.aliases if item.strip()))
        object.__setattr__(self, "aliases", aliases)
        for name in ("external_ids", "attributes"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping) or len(mapping) > 64:
                raise ValueError(f"{name} must be a bounded mapping")
            safe = {_text(str(key), f"{name} key", 100): _text(str(value), f"{name} value", 500) for key, value in mapping.items()}
            object.__setattr__(self, name, safe)

    @property
    def entity_id(self) -> str:
        payload = {
            "owner_id": self.owner_id,
            "kind": self.kind,
            "canonical_name": normalize_entity_text(self.canonical_name),
            "external_ids": dict(sorted(self.external_ids.items())),
        }
        return hashlib.sha256(_canonical(payload)).hexdigest()


@dataclass(frozen=True)
class EntityMention:
    mention_id: str
    owner_id: str
    observed_text: str
    source_id: str
    start: int
    end: int
    kind_hint: str = "other"

    def __post_init__(self) -> None:
        object.__setattr__(self, "mention_id", _text(self.mention_id, "mention_id", 256))
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "observed_text", _text(self.observed_text, "observed_text", 1000))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (self.start, self.end)) or self.end <= self.start:
            raise ValueError("mention offsets are invalid")
        hint = _text(self.kind_hint, "kind_hint", 64).lower()
        if hint not in _ENTITY_KINDS:
            raise ValueError("unsupported entity kind hint")
        object.__setattr__(self, "kind_hint", hint)


@dataclass(frozen=True)
class EntityCandidate:
    entity_id: str
    score: float
    method: str

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or len(self.entity_id) != 64:
            raise ValueError("entity_id is invalid")
        object.__setattr__(self, "score", _probability(self.score, "score"))
        object.__setattr__(self, "method", _text(self.method, "method", 100))


class EntityResolverProvider(Protocol):
    def rank(self, mention: EntityMention, candidates: Sequence[Entity]) -> Sequence[float]: ...


@dataclass(frozen=True)
class EntityResolution:
    mention_id: str
    selected_entity_id: str | None
    confidence: float
    ambiguous: bool
    candidates: tuple[EntityCandidate, ...]


class EntityRegistry:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._alias_index: dict[tuple[str, str], set[str]] = {}

    def register(self, entity: Entity) -> str:
        if not isinstance(entity, Entity):
            raise TypeError("entity must be Entity")
        existing = self._entities.get(entity.entity_id)
        if existing is not None and existing != entity:
            raise ValueError("entity identity collision")
        if existing is None and len(self._entities) >= _MAX_ENTITIES:
            raise RuntimeError("entity registry capacity reached")
        self._entities[entity.entity_id] = entity
        for alias in (entity.canonical_name, *entity.aliases):
            key = (entity.owner_id, normalize_entity_text(alias))
            self._alias_index.setdefault(key, set()).add(entity.entity_id)
        return entity.entity_id

    def get(self, owner_id: str, entity_id: str) -> Entity:
        owner = normalize_owner_id(owner_id)
        entity = self._entities[entity_id]
        if entity.owner_id != owner:
            raise PermissionError("entity belongs to another owner")
        return entity

    def resolve(
        self,
        mention: EntityMention,
        *,
        provider: EntityResolverProvider | None = None,
        threshold: float = 0.85,
        ambiguity_margin: float = 0.05,
    ) -> EntityResolution:
        threshold_value = _probability(threshold, "threshold")
        margin = _probability(ambiguity_margin, "ambiguity_margin")
        key = (mention.owner_id, normalize_entity_text(mention.observed_text))
        exact_ids = sorted(self._alias_index.get(key, ()))
        candidate_entities = [self._entities[item] for item in exact_ids]
        if not candidate_entities:
            normalized = key[1]
            tokens = set(normalized.split())
            scored: list[tuple[float, Entity]] = []
            for entity in self._entities.values():
                if entity.owner_id != mention.owner_id or (mention.kind_hint != "other" and entity.kind != mention.kind_hint):
                    continue
                names = (entity.canonical_name, *entity.aliases)
                best = 0.0
                for name in names:
                    candidate_tokens = set(normalize_entity_text(name).split())
                    union = tokens | candidate_tokens
                    score = len(tokens & candidate_tokens) / len(union) if union else 0.0
                    best = max(best, score)
                if best > 0.0:
                    scored.append((best, entity))
            scored.sort(key=lambda item: (-item[0], item[1].entity_id))
            candidate_entities = [entity for _, entity in scored[:_MAX_CANDIDATES]]
            base_scores = [score for score, _ in scored[:_MAX_CANDIDATES]]
        else:
            base_scores = [1.0 for _ in candidate_entities]
        if not candidate_entities:
            return EntityResolution(mention.mention_id, None, 0.0, False, ())
        if provider is not None:
            try:
                provider_scores = list(provider.rank(mention, candidate_entities))
            except Exception:
                provider_scores = []
            if len(provider_scores) == len(candidate_entities):
                base_scores = [_probability(value, "provider score") for value in provider_scores]
        ranked = sorted(zip(base_scores, candidate_entities), key=lambda item: (-item[0], item[1].entity_id))
        candidates = tuple(EntityCandidate(entity.entity_id, score, "provider" if provider is not None else "deterministic") for score, entity in ranked)
        top_score, top_entity = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        ambiguous = len(ranked) > 1 and (top_score - second_score) < margin
        selected = top_entity.entity_id if top_score >= threshold_value and not ambiguous else None
        return EntityResolution(mention.mention_id, selected, top_score, ambiguous, candidates)


@dataclass(frozen=True, order=True)
class TemporalPoint:
    value: dt.datetime
    precision: str = "second"

    def __post_init__(self) -> None:
        if not isinstance(self.value, dt.datetime):
            raise ValueError("value must be datetime")
        value = self.value
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        object.__setattr__(self, "value", value.astimezone(dt.timezone.utc))
        precision = _text(self.precision, "precision", 16).lower()
        if precision not in {"year", "month", "day", "second"}:
            raise ValueError("unsupported temporal precision")
        object.__setattr__(self, "precision", precision)

    @classmethod
    def parse(cls, value: str, *, precision: str = "second") -> "TemporalPoint":
        raw = _text(value, "temporal value", 64)
        try:
            if re.fullmatch(r"\d{4}", raw):
                parsed = dt.datetime(int(raw), 1, 1, tzinfo=dt.timezone.utc)
                precision = "year"
            elif re.fullmatch(r"\d{4}-\d{2}", raw):
                parsed = dt.datetime.fromisoformat(raw + "-01").replace(tzinfo=dt.timezone.utc)
                precision = "month"
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
                parsed = dt.datetime.fromisoformat(raw).replace(tzinfo=dt.timezone.utc)
                precision = "day"
            else:
                parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except (ValueError, OverflowError) as exc:
            raise ValueError("invalid ISO temporal value") from exc
        return cls(parsed, precision)


@dataclass(frozen=True)
class TemporalInterval:
    start: TemporalPoint
    end: TemporalPoint
    role: str
    inclusive_start: bool = True
    inclusive_end: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.start, TemporalPoint) or not isinstance(self.end, TemporalPoint):
            raise ValueError("start/end must be TemporalPoint")
        if self.end.value < self.start.value:
            raise ValueError("temporal interval end precedes start")
        role = _text(self.role, "temporal role", 32).lower()
        if role not in _TEMPORAL_ROLES:
            raise ValueError("unsupported temporal role")
        object.__setattr__(self, "role", role)

    def contains(self, point: TemporalPoint) -> bool:
        left = point.value > self.start.value or (self.inclusive_start and point.value == self.start.value)
        right = point.value < self.end.value or (self.inclusive_end and point.value == self.end.value)
        return left and right

    def overlaps(self, other: "TemporalInterval") -> bool:
        return max(self.start.value, other.start.value) <= min(self.end.value, other.end.value)

    def relation(self, other: "TemporalInterval") -> str:
        if self.end.value < other.start.value:
            return "before"
        if self.start.value > other.end.value:
            return "after"
        if self.start.value == other.start.value and self.end.value == other.end.value:
            return "equal"
        if self.start.value >= other.start.value and self.end.value <= other.end.value:
            return "during"
        if other.start.value >= self.start.value and other.end.value <= self.end.value:
            return "contains"
        return "overlaps"


@dataclass(frozen=True)
class TemporalEvidence:
    evidence_id: str
    interval: TemporalInterval
    publication: TemporalPoint | None = None
    retracted_at: TemporalPoint | None = None
    superseded_at: TemporalPoint | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, "evidence_id", 500))
        if not isinstance(self.interval, TemporalInterval):
            raise ValueError("interval must be TemporalInterval")
        for name in ("publication", "retracted_at", "superseded_at"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, TemporalPoint):
                raise ValueError(f"{name} must be TemporalPoint")

    def valid_as_of(self, as_of: TemporalPoint) -> bool:
        if self.publication is not None and self.publication.value > as_of.value:
            return False
        if self.retracted_at is not None and self.retracted_at.value <= as_of.value:
            return False
        if self.superseded_at is not None and self.superseded_at.value <= as_of.value:
            return False
        return True


def freshness_score(publication: TemporalPoint, *, as_of: TemporalPoint, half_life_days: float = 365.0) -> float:
    if isinstance(half_life_days, bool):
        raise ValueError("half_life_days is invalid")
    half_life = float(half_life_days)
    if not math.isfinite(half_life) or half_life <= 0:
        raise ValueError("half_life_days must be positive")
    age_days = max(0.0, (as_of.value - publication.value).total_seconds() / 86400.0)
    return math.exp(-math.log(2.0) * age_days / half_life)


def filter_as_of(evidence: Sequence[TemporalEvidence], as_of: TemporalPoint) -> tuple[TemporalEvidence, ...]:
    return tuple(item for item in evidence if item.valid_as_of(as_of))


__all__ = [
    "Entity",
    "EntityCandidate",
    "EntityMention",
    "EntityRegistry",
    "EntityResolution",
    "EntityResolverProvider",
    "TemporalEvidence",
    "TemporalInterval",
    "TemporalPoint",
    "filter_as_of",
    "freshness_score",
    "normalize_entity_text",
]
