"""Deterministic spatiotemporal evidence index for geospatial/scientific retrieval.

The reference implementation uses bounded in-memory interval filtering and explicit CRS
identity. Production backends may compile the same records to PostGIS/R-tree/vector
stores. No reprojection is guessed: query and evidence CRS must match or a reviewed
transformer must be supplied by the caller.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

from tools.hydrology_domain import CRSRef

_MAX_RECORDS = 2_000_000


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _utc(value: dt.datetime) -> dt.datetime:
    if not isinstance(value, dt.datetime):
        raise ValueError("timestamp must be datetime")
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class SpatialEnvelope:
    crs: CRSRef
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        if not isinstance(self.crs, CRSRef):
            raise ValueError("crs must be CRSRef")
        for name in ("min_x", "min_y", "max_x", "max_y"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.max_x < self.min_x or self.max_y < self.min_y:
            raise ValueError("spatial envelope bounds are invalid")

    def intersects(self, other: "SpatialEnvelope") -> bool:
        if self.crs != other.crs:
            raise ValueError("cannot compare spatial envelopes with different CRS")
        return not (
            self.max_x < other.min_x or other.max_x < self.min_x or
            self.max_y < other.min_y or other.max_y < self.min_y
        )

    def contains(self, other: "SpatialEnvelope") -> bool:
        if self.crs != other.crs:
            raise ValueError("cannot compare spatial envelopes with different CRS")
        return self.min_x <= other.min_x and self.min_y <= other.min_y and self.max_x >= other.max_x and self.max_y >= other.max_y


@dataclass(frozen=True)
class TimeEnvelope:
    start: dt.datetime
    end: dt.datetime

    def __post_init__(self) -> None:
        start, end = _utc(self.start), _utc(self.end)
        if end < start:
            raise ValueError("time envelope end precedes start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def intersects(self, other: "TimeEnvelope") -> bool:
        return max(self.start, other.start) <= min(self.end, other.end)


@dataclass(frozen=True)
class SpatiotemporalRecord:
    record_id: str
    source_id: str
    spatial: SpatialEnvelope | None
    temporal: TimeEnvelope | None
    variable: str = ""
    modality: str = "text"
    content_sha256: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "record_id", _text(self.record_id, "record_id", 256))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        if self.spatial is None and self.temporal is None:
            raise ValueError("spatiotemporal record must contain spatial and/or temporal scope")
        if self.spatial is not None and not isinstance(self.spatial, SpatialEnvelope):
            raise ValueError("spatial must be SpatialEnvelope")
        if self.temporal is not None and not isinstance(self.temporal, TimeEnvelope):
            raise ValueError("temporal must be TimeEnvelope")
        object.__setattr__(self, "variable", _text(self.variable, "variable", 256, allow_empty=True))
        object.__setattr__(self, "modality", _text(self.modality, "modality", 64).lower())
        if self.content_sha256:
            digest = self.content_sha256.strip().lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("content_sha256 is invalid")
            object.__setattr__(self, "content_sha256", digest)
        if not isinstance(self.metadata, Mapping) or len(self.metadata) > 64:
            raise ValueError("metadata must be a bounded mapping")
        object.__setattr__(self, "metadata", {_text(str(k), "metadata key", 100): _text(str(v), "metadata value", 1000, allow_empty=True) for k, v in self.metadata.items()})

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()


@dataclass(frozen=True)
class SpatiotemporalQuery:
    spatial: SpatialEnvelope | None = None
    temporal: TimeEnvelope | None = None
    variable: str = ""
    modalities: tuple[str, ...] = ()
    require_containment: bool = False

    def __post_init__(self) -> None:
        if self.spatial is None and self.temporal is None and not self.variable and not self.modalities:
            raise ValueError("spatiotemporal query has no constraints")
        object.__setattr__(self, "variable", _text(self.variable, "variable", 256, allow_empty=True))
        object.__setattr__(self, "modalities", tuple(dict.fromkeys(_text(item, "modality", 64).lower() for item in self.modalities)))
        if not isinstance(self.require_containment, bool):
            raise ValueError("require_containment must be boolean")


class SpatiotemporalIndex:
    def __init__(self) -> None:
        self._records: dict[str, SpatiotemporalRecord] = {}

    def upsert(self, record: SpatiotemporalRecord) -> None:
        if not isinstance(record, SpatiotemporalRecord):
            raise TypeError("record must be SpatiotemporalRecord")
        existing = self._records.get(record.record_id)
        if existing is not None and existing.fingerprint != record.fingerprint:
            raise ValueError("record ID collision; immutable record IDs may not change content")
        if existing is None and len(self._records) >= _MAX_RECORDS:
            raise RuntimeError("spatiotemporal index capacity reached")
        self._records[record.record_id] = record

    def delete(self, record_id: str) -> None:
        del self._records[_text(record_id, "record_id", 256)]

    def search(
        self,
        query: SpatiotemporalQuery,
        *,
        limit: int = 200,
        transformer: Callable[[SpatialEnvelope, CRSRef], SpatialEnvelope] | None = None,
    ) -> tuple[SpatiotemporalRecord, ...]:
        if not isinstance(query, SpatiotemporalQuery):
            raise TypeError("query must be SpatiotemporalQuery")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit is invalid")
        matches: list[SpatiotemporalRecord] = []
        for record in self._records.values():
            if query.variable and record.variable.casefold() != query.variable.casefold():
                continue
            if query.modalities and record.modality not in query.modalities:
                continue
            if query.temporal is not None:
                if record.temporal is None or not record.temporal.intersects(query.temporal):
                    continue
            if query.spatial is not None:
                if record.spatial is None:
                    continue
                spatial_query = query.spatial
                if record.spatial.crs != spatial_query.crs:
                    if transformer is None:
                        continue
                    spatial_query = transformer(spatial_query, record.spatial.crs)
                    if not isinstance(spatial_query, SpatialEnvelope) or spatial_query.crs != record.spatial.crs:
                        raise RuntimeError("spatial transformer returned an invalid envelope")
                if query.require_containment:
                    if not record.spatial.contains(spatial_query):
                        continue
                elif not record.spatial.intersects(spatial_query):
                    continue
            matches.append(record)
        matches.sort(key=lambda item: (item.source_id, item.record_id))
        return tuple(matches[:limit])

    @property
    def fingerprint(self) -> str:
        payload = [(record_id, record.fingerprint) for record_id, record in sorted(self._records.items())]
        return hashlib.sha256(_canonical(payload)).hexdigest()


__all__ = ["SpatialEnvelope", "SpatiotemporalIndex", "SpatiotemporalQuery", "SpatiotemporalRecord", "TimeEnvelope"]
