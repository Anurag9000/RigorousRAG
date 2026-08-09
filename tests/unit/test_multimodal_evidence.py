from __future__ import annotations

import pytest

from tools.multimodal_evidence import (
    NormalizedBBox,
    build_evidence_region,
    content_digest,
    deduplicate_overlapping_regions,
    normalize_extracted_regions,
    region_citation,
)


class Extractor:
    extractor_id = "layout-v1"

    def extract_regions(self, document_bytes):
        assert document_bytes == b"pdf-bytes"
        return [
            {
                "page_number": 2,
                "kind": "figure",
                "x0": 0.1,
                "y0": 0.2,
                "x1": 0.8,
                "y1": 0.7,
                "content": b"figure pixels",
                "confidence": 0.91,
            },
            {
                "page_number": 1,
                "kind": "table",
                "x0": 0.05,
                "y0": 0.1,
                "x1": 0.95,
                "y1": 0.5,
                "content": "table cells",
                "confidence": 0.98,
            },
        ]


def test_layout_output_normalizes_to_hash_only_regions_and_coordinate_citations():
    regions = normalize_extracted_regions(
        owner_id="alice",
        doc_id="doc-1",
        source_bytes=b"pdf-bytes",
        extractor=Extractor(),
    )
    assert [region.page_number for region in regions] == [1, 2]
    assert [region.kind for region in regions] == ["table", "figure"]
    assert all(region.source_sha256 == content_digest(b"pdf-bytes") for region in regions)
    assert "table cells" not in repr(regions)
    assert "figure pixels" not in repr(regions)

    citation = region_citation(regions[0])
    assert citation.doc_id == "doc-1"
    assert citation.page_number == 1
    assert citation.region_id == regions[0].region_id
    assert citation.bbox == regions[0].bbox
    assert len(citation.citation_id) == 64


def test_region_identity_changes_with_coordinates_content_or_extractor():
    source = content_digest(b"source")
    content = content_digest(b"content")
    first = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=source,
        page_number=1,
        kind="chart",
        bbox=NormalizedBBox(0.1, 0.1, 0.9, 0.9),
        content_sha256=content,
        extractor_id="extractor-a",
    )
    moved = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=source,
        page_number=1,
        kind="chart",
        bbox=NormalizedBBox(0.1, 0.2, 0.9, 0.9),
        content_sha256=content,
        extractor_id="extractor-a",
    )
    other_extractor = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=source,
        page_number=1,
        kind="chart",
        bbox=NormalizedBBox(0.1, 0.1, 0.9, 0.9),
        content_sha256=content,
        extractor_id="extractor-b",
    )
    assert len({first.region_id, moved.region_id, other_extractor.region_id}) == 3
    assert len(first.provenance_digest) == 64


def test_overlapping_same_content_regions_deduplicate_by_confidence():
    source = content_digest(b"source")
    content = content_digest(b"same")
    high = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=source,
        page_number=1,
        kind="caption",
        bbox=NormalizedBBox(0.1, 0.1, 0.8, 0.3),
        content_sha256=content,
        extractor_id="layout-v1",
        confidence=0.95,
    )
    overlap = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=source,
        page_number=1,
        kind="caption",
        bbox=NormalizedBBox(0.11, 0.1, 0.81, 0.3),
        content_sha256=content,
        extractor_id="layout-v1",
        confidence=0.80,
    )
    distinct = build_evidence_region(
        owner_id="alice",
        doc_id="doc-1",
        source_sha256=source,
        page_number=1,
        kind="caption",
        bbox=NormalizedBBox(0.1, 0.6, 0.8, 0.8),
        content_sha256=content,
        extractor_id="layout-v1",
        confidence=0.70,
    )
    deduplicated = deduplicate_overlapping_regions(
        [overlap, distinct, high],
        iou_threshold=0.90,
    )
    assert high in deduplicated
    assert overlap not in deduplicated
    assert distinct in deduplicated


def test_bbox_and_layout_boundaries_fail_closed_without_raw_content_leakage():
    with pytest.raises(ValueError, match="positive"):
        NormalizedBBox(0.5, 0.1, 0.5, 0.9)

    class BadExtractor:
        extractor_id = "layout-v1"

        def extract_regions(self, document_bytes):
            return [
                {
                    "page_number": 1,
                    "kind": "unknown",
                    "x0": 0.1,
                    "y0": 0.1,
                    "x1": 0.9,
                    "y1": 0.9,
                    "content": "private OCR text",
                }
            ]

    with pytest.raises(ValueError, match="unsupported"):
        normalize_extracted_regions(
            owner_id="alice",
            doc_id="doc-1",
            source_bytes=b"pdf-bytes",
            extractor=BadExtractor(),
        )


def test_layout_extractor_iteration_is_bounded_before_materialization():
    class UnboundedExtractor:
        extractor_id = "layout-v1"

        def extract_regions(self, document_bytes):
            for index in range(100_001):
                yield {
                    "page_number": 1,
                    "kind": "text",
                    "x0": 0.0,
                    "y0": 0.0,
                    "x1": 1.0,
                    "y1": 1.0,
                    "content": str(index),
                }

    with pytest.raises(ValueError, match="count exceeds"):
        normalize_extracted_regions(
            owner_id="alice",
            doc_id="doc-1",
            source_bytes=b"pdf-bytes",
            extractor=UnboundedExtractor(),
        )


def test_deduplication_validates_region_types_before_sorting():
    with pytest.raises(ValueError, match="EvidenceRegion"):
        deduplicate_overlapping_regions([object()])
