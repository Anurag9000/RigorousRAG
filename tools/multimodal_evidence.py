"""Immutable page-coordinate evidence regions for structured multimodal RAG.

Regions contain server-owned identities, normalized page coordinates and content
hashes, never raw extracted text. OCR/layout engines are injected through a small
protocol so extraction can be model-specific without weakening citation lineage.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import operator
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from tools.security import normalize_owner_id

_REGION_KINDS = frozenset({"text", "table", "figure", "caption", "chart", "equation"})
_MAX_REGIONS = 100_000
_MAX_PAGE = 1_000_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    result = value.strip()
    if (
        not result
        or len(result) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise ValueError(f"{label} is invalid.")
    return result


def _digest(value: Any, label: str) -> str:
    result = _identifier(value, label, 64).lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return result


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _unit(value: Any, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1.")
    return result


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        result = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return result


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def content_digest(value: bytes | bytearray | memoryview | str) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    else:
        raise ValueError("content must be bytes or text.")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class NormalizedBBox:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        for name in ("x0", "y0", "x1", "y1"):
            object.__setattr__(self, name, _unit(getattr(self, name), name))
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive normalized area.")

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def intersection_over_union(self, other: "NormalizedBBox") -> float:
        if not isinstance(other, NormalizedBBox):
            raise ValueError("other must be NormalizedBBox.")
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        intersection = (x1 - x0) * (y1 - y0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class EvidenceRegion:
    region_id: str
    owner_id: str
    doc_id: str
    source_sha256: str
    page_number: int
    kind: str
    bbox: NormalizedBBox
    content_sha256: str
    extractor_id: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        owner = normalize_owner_id(self.owner_id)
        document = _identifier(self.doc_id, "doc_id", 200)
        source = _digest(self.source_sha256, "source_sha256")
        content = _digest(self.content_sha256, "content_sha256")
        page = _integer(self.page_number, "page_number", 1, _MAX_PAGE)
        kind = _identifier(self.kind, "kind", 50)
        if kind not in _REGION_KINDS:
            raise ValueError("region kind is unsupported.")
        if not isinstance(self.bbox, NormalizedBBox):
            raise ValueError("bbox must be NormalizedBBox.")
        extractor = _identifier(self.extractor_id, "extractor_id", 200)
        confidence = _unit(self.confidence, "confidence")
        expected = deterministic_region_id(
            owner_id=owner,
            doc_id=document,
            source_sha256=source,
            page_number=page,
            kind=kind,
            bbox=self.bbox,
            content_sha256=content,
            extractor_id=extractor,
        )
        if _digest(self.region_id, "region_id") != expected:
            raise ValueError("region_id does not match deterministic identity.")
        object.__setattr__(self, "region_id", expected)
        object.__setattr__(self, "owner_id", owner)
        object.__setattr__(self, "doc_id", document)
        object.__setattr__(self, "source_sha256", source)
        object.__setattr__(self, "content_sha256", content)
        object.__setattr__(self, "page_number", page)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "extractor_id", extractor)
        object.__setattr__(self, "confidence", confidence)

    @property
    def provenance_digest(self) -> str:
        return _sha256(asdict(self))


def deterministic_region_id(
    *,
    owner_id: str,
    doc_id: str,
    source_sha256: str,
    page_number: int,
    kind: str,
    bbox: NormalizedBBox,
    content_sha256: str,
    extractor_id: str,
) -> str:
    owner = normalize_owner_id(owner_id)
    document = _identifier(doc_id, "doc_id", 200)
    source = _digest(source_sha256, "source_sha256")
    page = _integer(page_number, "page_number", 1, _MAX_PAGE)
    selected_kind = _identifier(kind, "kind", 50)
    if selected_kind not in _REGION_KINDS:
        raise ValueError("region kind is unsupported.")
    if not isinstance(bbox, NormalizedBBox):
        raise ValueError("bbox must be NormalizedBBox.")
    content = _digest(content_sha256, "content_sha256")
    extractor = _identifier(extractor_id, "extractor_id", 200)
    return _sha256(
        {
            "contract": "rigorousrag-evidence-region-v1",
            "owner_id": owner,
            "doc_id": document,
            "source_sha256": source,
            "page_number": page,
            "kind": selected_kind,
            "bbox": asdict(bbox),
            "content_sha256": content,
            "extractor_id": extractor,
        }
    )


def build_evidence_region(
    *,
    owner_id: str,
    doc_id: str,
    source_sha256: str,
    page_number: int,
    kind: str,
    bbox: NormalizedBBox,
    content_sha256: str,
    extractor_id: str,
    confidence: float = 1.0,
) -> EvidenceRegion:
    region_id = deterministic_region_id(
        owner_id=owner_id,
        doc_id=doc_id,
        source_sha256=source_sha256,
        page_number=page_number,
        kind=kind,
        bbox=bbox,
        content_sha256=content_sha256,
        extractor_id=extractor_id,
    )
    return EvidenceRegion(
        region_id=region_id,
        owner_id=owner_id,
        doc_id=doc_id,
        source_sha256=source_sha256,
        page_number=page_number,
        kind=kind,
        bbox=bbox,
        content_sha256=content_sha256,
        extractor_id=extractor_id,
        confidence=confidence,
    )


@dataclass(frozen=True)
class PageCoordinateCitation:
    citation_id: str
    doc_id: str
    page_number: int
    region_id: str
    kind: str
    bbox: NormalizedBBox
    source_sha256: str

    def __post_init__(self) -> None:
        document = _identifier(self.doc_id, "doc_id", 200)
        region = _digest(self.region_id, "region_id")
        source = _digest(self.source_sha256, "source_sha256")
        page = _integer(self.page_number, "page_number", 1, _MAX_PAGE)
        kind = _identifier(self.kind, "kind", 50)
        if kind not in _REGION_KINDS:
            raise ValueError("citation region kind is unsupported.")
        if not isinstance(self.bbox, NormalizedBBox):
            raise ValueError("bbox must be NormalizedBBox.")
        expected = _sha256(
            {
                "contract": "rigorousrag-page-coordinate-citation-v1",
                "doc_id": document,
                "page_number": page,
                "region_id": region,
                "kind": kind,
                "bbox": asdict(self.bbox),
                "source_sha256": source,
            }
        )
        if _digest(self.citation_id, "citation_id") != expected:
            raise ValueError("citation_id does not match deterministic identity.")
        object.__setattr__(self, "citation_id", expected)
        object.__setattr__(self, "doc_id", document)
        object.__setattr__(self, "region_id", region)
        object.__setattr__(self, "source_sha256", source)
        object.__setattr__(self, "page_number", page)
        object.__setattr__(self, "kind", kind)


def region_citation(region: EvidenceRegion) -> PageCoordinateCitation:
    if not isinstance(region, EvidenceRegion):
        raise ValueError("region must be EvidenceRegion.")
    payload = {
        "contract": "rigorousrag-page-coordinate-citation-v1",
        "doc_id": region.doc_id,
        "page_number": region.page_number,
        "region_id": region.region_id,
        "kind": region.kind,
        "bbox": asdict(region.bbox),
        "source_sha256": region.source_sha256,
    }
    return PageCoordinateCitation(
        citation_id=_sha256(payload),
        doc_id=region.doc_id,
        page_number=region.page_number,
        region_id=region.region_id,
        kind=region.kind,
        bbox=region.bbox,
        source_sha256=region.source_sha256,
    )


class LayoutRegionExtractor(Protocol):
    extractor_id: str

    def extract_regions(self, document_bytes: bytes) -> Iterable[Mapping[str, Any]]: ...


def normalize_extracted_regions(
    *,
    owner_id: str,
    doc_id: str,
    source_bytes: bytes,
    extractor: LayoutRegionExtractor,
) -> tuple[EvidenceRegion, ...]:
    """Normalize model/OCR output into immutable regions without retaining raw text."""

    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("source_bytes must be non-empty immutable bytes.")
    extractor_id = _identifier(getattr(extractor, "extractor_id", None), "extractor_id", 200)
    method = getattr(extractor, "extract_regions", None)
    if not callable(method):
        raise ValueError("extractor must expose extract_regions().")
    try:
        raw_regions = list(
            itertools.islice(iter(method(source_bytes)), _MAX_REGIONS + 1)
        )
    except Exception as exc:
        raise RuntimeError("layout extraction failed.") from exc
    if len(raw_regions) > _MAX_REGIONS:
        raise ValueError("extracted region count exceeds the limit.")
    source_sha = content_digest(source_bytes)
    regions: list[EvidenceRegion] = []
    seen: set[str] = set()
    for raw in raw_regions:
        if not isinstance(raw, Mapping):
            raise ValueError("every extracted region must be a mapping.")
        raw_content = raw.get("content")
        if not isinstance(raw_content, (str, bytes, bytearray, memoryview)):
            raise ValueError("extracted region content must be bytes or text.")
        bbox = NormalizedBBox(
            raw.get("x0"),
            raw.get("y0"),
            raw.get("x1"),
            raw.get("y1"),
        )
        region = build_evidence_region(
            owner_id=owner_id,
            doc_id=doc_id,
            source_sha256=source_sha,
            page_number=raw.get("page_number"),
            kind=raw.get("kind"),
            bbox=bbox,
            content_sha256=content_digest(raw_content),
            extractor_id=extractor_id,
            confidence=raw.get("confidence", 1.0),
        )
        if region.region_id not in seen:
            seen.add(region.region_id)
            regions.append(region)
    return tuple(
        sorted(
            regions,
            key=lambda value: (
                value.page_number,
                value.bbox.y0,
                value.bbox.x0,
                value.kind,
                value.region_id,
            ),
        )
    )


def deduplicate_overlapping_regions(
    regions: Sequence[EvidenceRegion],
    *,
    iou_threshold: float = 0.90,
) -> tuple[EvidenceRegion, ...]:
    """Remove duplicate same-content regions from the same page using deterministic IoU."""

    if isinstance(regions, (str, bytes, bytearray)) or len(regions) > _MAX_REGIONS:
        raise ValueError("regions must be a bounded sequence.")
    values = tuple(regions)
    if any(not isinstance(region, EvidenceRegion) for region in values):
        raise ValueError("every region must be EvidenceRegion.")
    threshold = _unit(iou_threshold, "iou_threshold")
    result: list[EvidenceRegion] = []
    for region in sorted(
        values,
        key=lambda value: (
            value.page_number,
            value.kind,
            -value.confidence,
            value.region_id,
        ),
    ):
        duplicate = any(
            existing.doc_id == region.doc_id
            and existing.source_sha256 == region.source_sha256
            and existing.page_number == region.page_number
            and existing.kind == region.kind
            and existing.content_sha256 == region.content_sha256
            and existing.bbox.intersection_over_union(region.bbox) >= threshold
            for existing in result
        )
        if not duplicate:
            result.append(region)
    return tuple(result)


__all__ = [
    "EvidenceRegion",
    "LayoutRegionExtractor",
    "NormalizedBBox",
    "PageCoordinateCitation",
    "build_evidence_region",
    "content_digest",
    "deduplicate_overlapping_regions",
    "deterministic_region_id",
    "normalize_extracted_regions",
    "region_citation",
]