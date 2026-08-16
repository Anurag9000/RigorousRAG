"""Hierarchical multimodal retrieval over a fenced DocumentEvidenceBundle generation.

Stages: query/page late interaction -> winning regions -> region/block alignment -> resolved
cross-page structure expansion -> page-coordinate citation candidates. Runtime vectors stay
outside the bundle but every artifact identity is checked against its manifest reference.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.document_evidence_bundle import DocumentEvidenceBundle
from tools.document_ir import DocumentBlock, ScientificDocumentIR
from tools.document_structure_reconciliation import DocumentStructureResolution
from tools.multimodal_evidence import EvidenceRegion, PageCoordinateCitation, region_citation
from tools.page_late_interaction import (
    PageEmbeddingArtifact,
    PageEmbeddingBackend,
    PageSearchHit,
    query_embeddings,
    rank_pages,
    select_regions,
)

_MAX_PAGES = 1000
_MAX_REGIONS = 1000
_MAX_BLOCKS = 5000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _text(value: Any, label: str, maximum: int = 20_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


@dataclass(frozen=True)
class RegionBlockAlignment:
    region_id: str
    block_id: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExpandedBlock:
    block_id: str
    page_number: int
    role: str
    reason: str
    source_region_id: str = ""


@dataclass(frozen=True)
class DocumentRetrievalPlan:
    bundle_fingerprint: str
    query_sha256: str
    model_id: str
    page_hits: tuple[PageSearchHit, ...]
    selected_region_ids: tuple[str, ...]
    alignments: tuple[RegionBlockAlignment, ...]
    expanded_blocks: tuple[ExpandedBlock, ...]
    citations: tuple[PageCoordinateCitation, ...]
    diagnostics: tuple[str, ...]
    fingerprint: str


def _validate_bundle_runtime(
    bundle: DocumentEvidenceBundle,
    document: ScientificDocumentIR,
    resolution: DocumentStructureResolution,
    regions: Sequence[EvidenceRegion],
    pages: Sequence[PageEmbeddingArtifact],
) -> None:
    if bundle.owner_id != document.owner_id or bundle.doc_id != document.doc_id or bundle.source_sha256 != document.source_sha256:
        raise ValueError("document identity does not match the evidence bundle")
    if bundle.document_ir_fingerprint != document.fingerprint:
        raise ValueError("document IR generation does not match the evidence bundle")
    if bundle.structure_resolution_fingerprint != resolution.fingerprint:
        raise ValueError("document structure resolution does not match the evidence bundle")
    expected_regions = {item.region_id: item for item in bundle.regions}
    supplied_regions = {item.region_id: item for item in regions}
    if set(expected_regions) != set(supplied_regions):
        raise ValueError("runtime evidence-region set does not match the evidence bundle")
    for ref in bundle.regions:
        region = supplied_regions[ref.region_id]
        if region.content_sha256 != ref.content_sha256 or region.page_number != ref.page_number or region.kind != ref.kind:
            raise ValueError("runtime evidence region differs from its bundle reference")
    expected_pages = {item.page_artifact_sha256: item for item in bundle.page_embeddings}
    supplied_pages = {item.artifact_sha256: item for item in pages}
    if set(expected_pages) != set(supplied_pages):
        raise ValueError("runtime page-embedding set does not match the evidence bundle")
    for ref in bundle.page_embeddings:
        artifact = supplied_pages[ref.page_artifact_sha256]
        if artifact.page_number != ref.page_number or artifact.rendered_page_sha256 != ref.rendered_page_sha256 or artifact.model_id != ref.model_id:
            raise ValueError("runtime page embedding differs from its bundle reference")


def _alignment_score(region: EvidenceRegion, block: DocumentBlock) -> tuple[float, tuple[str, ...]]:
    if region.page_number != block.page_number:
        return 0.0, ()
    score = region.bbox.intersection_over_union(block.bbox)
    reasons: list[str] = []
    if score > 0:
        reasons.append("bbox_overlap")
    if region.content_sha256 == block.content_sha256:
        score += 1.0
        reasons.append("content_sha256_match")
    role_match = (
        region.kind == block.role
        or (region.kind == "equation" and block.role == "formula")
        or (region.kind in {"figure", "chart"} and block.role in {"figure", "chart"})
        or (region.kind == "text" and block.role in {"title", "heading", "paragraph", "list_item", "caption", "footnote", "reference"})
    )
    if role_match:
        score += 0.25
        reasons.append("role_compatible")
    return score, tuple(reasons)


def align_regions_to_blocks(
    document: ScientificDocumentIR,
    regions: Sequence[EvidenceRegion],
    *,
    minimum_score: float = 0.20,
) -> tuple[RegionBlockAlignment, ...]:
    if not 0.0 <= float(minimum_score) <= 2.25:
        raise ValueError("minimum_score is invalid")
    by_page: dict[int, list[DocumentBlock]] = {}
    for block in document.blocks:
        by_page.setdefault(block.page_number, []).append(block)
    output: list[RegionBlockAlignment] = []
    for region in regions:
        candidates: list[tuple[float, DocumentBlock, tuple[str, ...]]] = []
        for block in by_page.get(region.page_number, ()):
            score, reasons = _alignment_score(region, block)
            if score >= minimum_score:
                candidates.append((score, block, reasons))
        candidates.sort(key=lambda item: (-item[0], item[1].block_id))
        if not candidates:
            continue
        best = candidates[0]
        # Equal best candidates are ambiguous: do not guess a block identity.
        if len(candidates) > 1 and abs(best[0] - candidates[1][0]) < 1e-12:
            continue
        output.append(RegionBlockAlignment(region.region_id, best[1].block_id, best[0], best[2]))
    return tuple(output)


def _expand_structures(
    document: ScientificDocumentIR,
    resolution: DocumentStructureResolution,
    selected_regions: Sequence[EvidenceRegion],
    alignments: Sequence[RegionBlockAlignment],
    *,
    max_blocks: int,
) -> tuple[ExpandedBlock, ...]:
    by_id = {block.block_id: block for block in document.blocks}
    region_by_id = {region.region_id: region for region in selected_regions}
    block_to_regions: dict[str, list[str]] = {}
    for alignment in alignments:
        block_to_regions.setdefault(alignment.block_id, []).append(alignment.region_id)
    selected_blocks = set(block_to_regions)
    output: list[ExpandedBlock] = []
    seen: set[str] = set()

    def add(block_id: str, reason: str, source_region_id: str = "") -> None:
        if block_id in seen or block_id not in by_id or len(output) >= max_blocks:
            return
        seen.add(block_id)
        block = by_id[block_id]
        output.append(ExpandedBlock(block_id, block.page_number, block.role, reason, source_region_id))

    for alignment in alignments:
        add(alignment.block_id, "selected_region_alignment", alignment.region_id)

    for table in resolution.cross_page_tables:
        if not selected_blocks.intersection(table.block_ids):
            continue
        source_region = ""
        for block_id in table.block_ids:
            if block_id in block_to_regions:
                source_region = block_to_regions[block_id][0]
                break
        for block_id in table.block_ids:
            add(block_id, "cross_page_table_continuation", source_region)

    for binding in resolution.formula_symbols:
        if binding.formula_block_id in selected_blocks:
            add(binding.evidence_block_id, "formula_symbol_definition")
        elif binding.evidence_block_id in selected_blocks:
            add(binding.formula_block_id, "formula_defined_by_selected_prose")

    for binding in resolution.figure_panels:
        if binding.figure_block_id in selected_blocks:
            add(binding.evidence_block_id, "figure_panel_caption")
        elif binding.evidence_block_id in selected_blocks:
            add(binding.figure_block_id, "caption_describes_figure_panel")

    # One-hop structural links make caption/reference context available without unbounded graph expansion.
    for link in document.links:
        if link.source_block_id in selected_blocks and link.kind in {"caption_of", "refers_to", "equation_ref", "defines", "continues"}:
            add(link.target_block_id, f"document_link:{link.kind}")
        if link.target_block_id in selected_blocks and link.kind in {"caption_of", "defines", "continues"}:
            add(link.source_block_id, f"reverse_document_link:{link.kind}")
    return tuple(output)


def plan_document_hierarchical_retrieval(
    bundle: DocumentEvidenceBundle,
    document: ScientificDocumentIR,
    resolution: DocumentStructureResolution,
    regions: Sequence[EvidenceRegion],
    page_embeddings: Sequence[PageEmbeddingArtifact],
    *,
    query: str,
    backend: PageEmbeddingBackend,
    top_k_pages: int = 20,
    max_pages_per_document: int = 4,
    max_regions: int = 50,
    max_blocks: int = 200,
) -> DocumentRetrievalPlan:
    if not isinstance(bundle, DocumentEvidenceBundle):
        raise TypeError("bundle must be DocumentEvidenceBundle")
    if isinstance(top_k_pages, bool) or not isinstance(top_k_pages, int) or not 1 <= top_k_pages <= _MAX_PAGES:
        raise ValueError("top_k_pages is invalid")
    if isinstance(max_regions, bool) or not isinstance(max_regions, int) or not 1 <= max_regions <= _MAX_REGIONS:
        raise ValueError("max_regions is invalid")
    if isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or not 1 <= max_blocks <= _MAX_BLOCKS:
        raise ValueError("max_blocks is invalid")
    normalized_query = _text(query, "query")
    _validate_bundle_runtime(bundle, document, resolution, regions, page_embeddings)
    query_vectors = query_embeddings(normalized_query, backend)
    page_models = {item.model_id for item in page_embeddings}
    if page_models and query_vectors.model_id not in page_models:
        raise ValueError("query backend model_id does not match the bundled page embeddings")
    page_hits = rank_pages(
        query_vectors,
        page_embeddings,
        top_k=top_k_pages,
        max_pages_per_document=max_pages_per_document,
    )
    selected_regions = select_regions(page_hits, regions, max_regions=max_regions)
    alignments = align_regions_to_blocks(document, selected_regions)
    expanded = _expand_structures(
        document,
        resolution,
        selected_regions,
        alignments,
        max_blocks=max_blocks,
    )
    citations = tuple(region_citation(region) for region in selected_regions)
    diagnostics = list(bundle.diagnostics)
    aligned_ids = {item.region_id for item in alignments}
    for region in selected_regions:
        if region.region_id not in aligned_ids:
            diagnostics.append(f"selected_region_not_aligned_to_ir:{region.region_id}")
    payload = {
        "contract": "rigorousrag-document-hierarchical-retrieval-plan-v1",
        "bundle_fingerprint": bundle.fingerprint,
        "query_sha256": query_vectors.query_sha256,
        "model_id": query_vectors.model_id,
        "page_hits": [asdict(item) for item in page_hits],
        "selected_region_ids": [item.region_id for item in selected_regions],
        "alignments": [asdict(item) for item in alignments],
        "expanded_blocks": [asdict(item) for item in expanded],
        "citations": [asdict(item) for item in citations],
        "diagnostics": diagnostics,
    }
    fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
    return DocumentRetrievalPlan(
        bundle.fingerprint,
        query_vectors.query_sha256,
        query_vectors.model_id,
        page_hits,
        tuple(item.region_id for item in selected_regions),
        alignments,
        expanded,
        citations,
        tuple(dict.fromkeys(diagnostics)),
        fingerprint,
    )


__all__ = [
    "DocumentRetrievalPlan",
    "ExpandedBlock",
    "RegionBlockAlignment",
    "align_regions_to_blocks",
    "plan_document_hierarchical_retrieval",
]
