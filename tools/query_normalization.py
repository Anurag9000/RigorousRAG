"""Governed entity and temporal normalization for adaptive query routing."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_MAX_QUERY = 20_000
_MAX_ENTITIES = 128
_MAX_RANGES = 128
_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20\d{2}|2100)(?!\d)")


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if not rendered or len(rendered) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered):
        raise ValueError(f"{label} is invalid.")
    return rendered


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
        raise ValueError(f"{label} must be a finite timestamp.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite timestamp.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite timestamp.")
    return parsed


@dataclass(frozen=True)
class CanonicalEntity:
    surface: str
    canonical_id: str
    canonical_name: str
    confidence: float = 1.0
    resolver_version: str = "fallback-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", _text(self.surface, "surface", 1_000))
        object.__setattr__(self, "canonical_id", _text(self.canonical_id, "canonical_id", 500))
        object.__setattr__(self, "canonical_name", _text(self.canonical_name, "canonical_name", 1_000))
        object.__setattr__(self, "confidence", _unit(self.confidence, "confidence"))
        object.__setattr__(self, "resolver_version", _text(self.resolver_version, "resolver_version", 200))


@dataclass(frozen=True)
class TemporalRange:
    start_utc: float
    end_utc: float
    precision: str
    surface: str
    parser_version: str = "fallback-v1"

    def __post_init__(self) -> None:
        start = _timestamp(self.start_utc, "start_utc")
        end = _timestamp(self.end_utc, "end_utc")
        if end < start:
            raise ValueError("end_utc may not precede start_utc.")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)
        object.__setattr__(self, "precision", _text(self.precision, "precision", 50))
        object.__setattr__(self, "surface", _text(self.surface, "surface", 200))
        object.__setattr__(self, "parser_version", _text(self.parser_version, "parser_version", 200))


@dataclass(frozen=True)
class NormalizedQueryContext:
    query: str
    entities: tuple[CanonicalEntity, ...]
    temporal_ranges: tuple[TemporalRange, ...]
    entity_source: str
    temporal_source: str


EntityResolver = Callable[[str], Iterable[CanonicalEntity]]
TemporalParser = Callable[[str], Iterable[TemporalRange]]


def _fallback_entities(query: str, aliases: Mapping[str, tuple[str, str]] | None) -> tuple[CanonicalEntity, ...]:
    if aliases is None:
        return ()
    if not isinstance(aliases, Mapping) or len(aliases) > 10_000:
        raise ValueError("entity_aliases must be a bounded mapping.")
    rows: list[CanonicalEntity] = []
    for raw_alias, target in aliases.items():
        alias = _text(raw_alias, "entity alias", 500)
        if not isinstance(target, tuple) or len(target) != 2:
            raise ValueError("entity alias targets must be (canonical_id, canonical_name) tuples.")
        canonical_id = _text(target[0], "canonical_id", 500)
        canonical_name = _text(target[1], "canonical_name", 1_000)
        match = re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", query, flags=re.IGNORECASE)
        if match is None:
            continue
        rows.append(CanonicalEntity(match.group(0), canonical_id, canonical_name))
        if len(rows) >= _MAX_ENTITIES:
            break
    rows.sort(key=lambda item: (item.canonical_id, item.surface.lower()))
    return tuple(rows)


def _fallback_temporal(query: str) -> tuple[TemporalRange, ...]:
    rows: list[TemporalRange] = []
    occupied: list[tuple[int, int]] = []
    for match in _DATE_RE.finditer(query):
        try:
            day = datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)), tzinfo=timezone.utc)
        except ValueError:
            continue
        start = day.timestamp()
        rows.append(TemporalRange(start, start + 86_400.0 - 1e-6, "day", match.group(0)))
        occupied.append(match.span())
        if len(rows) >= _MAX_RANGES:
            return tuple(rows)
    for match in _YEAR_RE.finditer(query):
        if any(start <= match.start() and match.end() <= end for start, end in occupied):
            continue
        year = int(match.group(1))
        start_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        end_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        rows.append(TemporalRange(start_dt.timestamp(), end_dt.timestamp() - 1e-6, "year", match.group(0)))
        if len(rows) >= _MAX_RANGES:
            break
    rows.sort(key=lambda item: (item.start_utc, item.end_utc, item.surface))
    return tuple(rows)


def _bounded_entities(values: Iterable[CanonicalEntity]) -> tuple[CanonicalEntity, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("entity resolver returned an invalid result.")
    result: list[CanonicalEntity] = []
    for value in values:
        if len(result) >= _MAX_ENTITIES:
            raise ValueError("entity resolver exceeded the entity limit.")
        if not isinstance(value, CanonicalEntity):
            raise ValueError("entity resolver returned an invalid entity.")
        result.append(value)
    return tuple(result)


def _bounded_ranges(values: Iterable[TemporalRange]) -> tuple[TemporalRange, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError("temporal parser returned an invalid result.")
    result: list[TemporalRange] = []
    for value in values:
        if len(result) >= _MAX_RANGES:
            raise ValueError("temporal parser exceeded the range limit.")
        if not isinstance(value, TemporalRange):
            raise ValueError("temporal parser returned an invalid range.")
        result.append(value)
    return tuple(result)


def normalize_query_context(
    query: str,
    *,
    entity_aliases: Mapping[str, tuple[str, str]] | None = None,
    entity_resolver: EntityResolver | None = None,
    temporal_parser: TemporalParser | None = None,
) -> NormalizedQueryContext:
    """Normalize only what is validated; injected failures fall back deterministically."""

    if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY:
        raise ValueError("query must be a bounded non-empty string.")
    cleaned = query.strip()
    entities: tuple[CanonicalEntity, ...]
    entity_source = "fallback"
    if entity_resolver is not None:
        if not callable(entity_resolver):
            raise ValueError("entity_resolver must be callable.")
        try:
            entities = _bounded_entities(entity_resolver(cleaned))
            entity_source = "resolver"
        except Exception:
            entities = _fallback_entities(cleaned, entity_aliases)
    else:
        entities = _fallback_entities(cleaned, entity_aliases)
    temporal_ranges: tuple[TemporalRange, ...]
    temporal_source = "fallback"
    if temporal_parser is not None:
        if not callable(temporal_parser):
            raise ValueError("temporal_parser must be callable.")
        try:
            temporal_ranges = _bounded_ranges(temporal_parser(cleaned))
            temporal_source = "parser"
        except Exception:
            temporal_ranges = _fallback_temporal(cleaned)
    else:
        temporal_ranges = _fallback_temporal(cleaned)
    return NormalizedQueryContext(cleaned, entities, temporal_ranges, entity_source, temporal_source)


__all__ = [
    "CanonicalEntity",
    "EntityResolver",
    "NormalizedQueryContext",
    "TemporalParser",
    "TemporalRange",
    "normalize_query_context",
]
