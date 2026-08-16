"""Content-addressed manifest for a complete multimodal scientific-document generation.

The bundle binds structured IR, reconciled cross-page structure, page-coordinate evidence,
page-native late-interaction artifacts and the cross-modal evidence graph. It stores only
identities/metadata—never embedding vectors or duplicated raw source text.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.document_graph_builder import build_document_evidence_graph
from tools.document_ir import ScientificDocumentIR
from tools.document_structure_reconciliation import (
    DocumentStructureResolution,
    enrich_with_resolved_structure_links,
)
from tools.graph_reasoning import EvidenceGraph
from tools.multimodal_evidence import EvidenceRegion
from tools.page_late_interaction import PageEmbeddingArtifact

_MAX_REGIONS = 500_000
_MAX_PAGE_EMBEDDINGS = 100_000


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _text(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if not cleaned or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _sha(value: Any, label: str) -> str:
    cleaned = _text(value, label, 64).lower()
    if len(cleaned) != 64 or any(ch not in "0123456789abcdef" for ch in cleaned):
        raise ValueError(f"{label} must be SHA-256")
    return cleaned


@dataclass(frozen=True)
class RegionReference:
    region_id: str
    page_number: int
    kind: str
    content_sha256: str
    extractor_id: str
    confidence: float
    bbox: Mapping[str, float]


@dataclass(frozen=True)
class PageEmbeddingReference:
    page_artifact_sha256: str
    page_number: int
    rendered_page_sha256: str
    model_id: str
    dimension: int
    patch_count: int
    region_bound_patch_count: int


@dataclass(frozen=True)
class DocumentEvidenceBundle:
    owner_id: str
    doc_id: str
    source_sha256: str
    generation_id: str
    document_ir_fingerprint: str
    resolved_document_ir_fingerprint: str
    structure_resolution_fingerprint: str
    graph_fingerprint: str
    graph_node_count: int
    graph_edge_count: int
    regions: tuple[RegionReference, ...]
    page_embeddings: tuple[PageEmbeddingReference, ...]
    extractor_id: str
    schema_version: str
    diagnostics: tuple[str, ...]
    fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "source_sha256",
            "document_ir_fingerprint",
            "resolved_document_ir_fingerprint",
            "structure_resolution_fingerprint",
            "graph_fingerprint",
            "fingerprint",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), name))
        object.__setattr__(self, "generation_id", _text(self.generation_id, "generation_id", 256))
        object.__setattr__(self, "doc_id", _text(self.doc_id, "doc_id", 200))
        if len(self.regions) > _MAX_REGIONS or len(self.page_embeddings) > _MAX_PAGE_EMBEDDINGS:
            raise ValueError("document evidence bundle exceeds item limits")
        if len({item.region_id for item in self.regions}) != len(self.regions):
            raise ValueError("document evidence bundle contains duplicate region identities")
        if len({item.page_artifact_sha256 for item in self.page_embeddings}) != len(self.page_embeddings):
            raise ValueError("document evidence bundle contains duplicate page embedding artifacts")


def _region_ref(region: EvidenceRegion) -> RegionReference:
    return RegionReference(
        region.region_id,
        region.page_number,
        region.kind,
        region.content_sha256,
        region.extractor_id,
        region.confidence,
        asdict(region.bbox),
    )


def _page_ref(page: PageEmbeddingArtifact) -> PageEmbeddingReference:
    return PageEmbeddingReference(
        page.artifact_sha256,
        page.page_number,
        page.rendered_page_sha256,
        page.model_id,
        page.dimension,
        len(page.patch_vectors),
        sum(1 for value in page.patch_region_ids if value),
    )


def _validate_inputs(
    document: ScientificDocumentIR,
    resolution: DocumentStructureResolution,
    regions: Sequence[EvidenceRegion],
    page_embeddings: Sequence[PageEmbeddingArtifact],
) -> tuple[str, ...]:
    if resolution.document_fingerprint != document.fingerprint:
        raise ValueError("structure resolution does not belong to the supplied document IR")
    if len(regions) > _MAX_REGIONS or len(page_embeddings) > _MAX_PAGE_EMBEDDINGS:
        raise ValueError("multimodal evidence exceeds bundle limits")
    page_by_number = {page.page_number: page for page in document.pages}
    diagnostics: list[str] = list(resolution.unresolved)
    for region in regions:
        if not isinstance(region, EvidenceRegion):
            raise TypeError("regions must contain EvidenceRegion objects")
        if region.owner_id != document.owner_id or region.doc_id != document.doc_id or region.source_sha256 != document.source_sha256:
            raise ValueError("evidence region belongs to a different document/source/owner")
        if region.page_number not in page_by_number:
            raise ValueError("evidence region references an unknown page")
    model_ids: set[str] = set()
    for artifact in page_embeddings:
        if not isinstance(artifact, PageEmbeddingArtifact):
            raise TypeError("page_embeddings must contain PageEmbeddingArtifact objects")
        if artifact.owner_id != document.owner_id or artifact.doc_id != document.doc_id or artifact.source_sha256 != document.source_sha256:
            raise ValueError("page embedding belongs to a different document/source/owner")
        page = page_by_number.get(artifact.page_number)
        if page is None:
            raise ValueError("page embedding references an unknown page")
        if page.rendered_sha256 and page.rendered_sha256 != artifact.rendered_page_sha256:
            raise ValueError("page embedding rendered hash does not match the document page generation")
        if artifact.patch_region_ids:
            known_regions = {region.region_id for region in regions if region.page_number == artifact.page_number}
            missing = sorted(set(artifact.patch_region_ids) - known_regions)
            if missing:
                raise ValueError("page embedding references unknown evidence regions")
        model_ids.add(artifact.model_id)
    if len(model_ids) > 1:
        diagnostics.append("multiple_page_embedding_models")
    pages_with_regions = {region.page_number for region in regions}
    pages_with_embeddings = {artifact.page_number for artifact in page_embeddings}
    for page_number in sorted(set(page_by_number) - pages_with_regions):
        diagnostics.append(f"page_without_regions:{page_number}")
    for page_number in sorted(set(page_by_number) - pages_with_embeddings):
        diagnostics.append(f"page_without_page_embedding:{page_number}")
    return tuple(dict.fromkeys(diagnostics))


def build_document_evidence_bundle(
    document: ScientificDocumentIR,
    resolution: DocumentStructureResolution,
    *,
    regions: Sequence[EvidenceRegion] = (),
    page_embeddings: Sequence[PageEmbeddingArtifact] = (),
    generation_id: str,
    graph: EvidenceGraph | None = None,
) -> DocumentEvidenceBundle:
    if not isinstance(document, ScientificDocumentIR):
        raise TypeError("document must be ScientificDocumentIR")
    generation = _text(generation_id, "generation_id", 256)
    diagnostics = _validate_inputs(document, resolution, regions, page_embeddings)
    resolved_document = enrich_with_resolved_structure_links(document, resolution)
    selected_graph = graph or build_document_evidence_graph(resolved_document, generation_id=generation)
    if not isinstance(selected_graph, EvidenceGraph):
        raise TypeError("graph must be EvidenceGraph or null")
    region_refs = tuple(sorted((_region_ref(item) for item in regions), key=lambda item: (item.page_number, item.kind, item.region_id)))
    page_refs = tuple(sorted((_page_ref(item) for item in page_embeddings), key=lambda item: (item.page_number, item.page_artifact_sha256)))
    payload = {
        "contract": "rigorousrag-document-evidence-bundle-v1",
        "owner_id": document.owner_id,
        "doc_id": document.doc_id,
        "source_sha256": document.source_sha256,
        "generation_id": generation,
        "document_ir_fingerprint": document.fingerprint,
        "resolved_document_ir_fingerprint": resolved_document.fingerprint,
        "structure_resolution_fingerprint": resolution.fingerprint,
        "graph_fingerprint": selected_graph.fingerprint,
        "graph_node_count": len(selected_graph.nodes),
        "graph_edge_count": len(selected_graph.edges),
        "regions": [asdict(item) for item in region_refs],
        "page_embeddings": [asdict(item) for item in page_refs],
        "extractor_id": document.extractor_id,
        "schema_version": document.schema_version,
        "diagnostics": diagnostics,
    }
    fingerprint = hashlib.sha256(_canonical(payload)).hexdigest()
    return DocumentEvidenceBundle(
        owner_id=document.owner_id,
        doc_id=document.doc_id,
        source_sha256=document.source_sha256,
        generation_id=generation,
        document_ir_fingerprint=document.fingerprint,
        resolved_document_ir_fingerprint=resolved_document.fingerprint,
        structure_resolution_fingerprint=resolution.fingerprint,
        graph_fingerprint=selected_graph.fingerprint,
        graph_node_count=len(selected_graph.nodes),
        graph_edge_count=len(selected_graph.edges),
        regions=region_refs,
        page_embeddings=page_refs,
        extractor_id=document.extractor_id,
        schema_version=document.schema_version,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
    )


def bundle_manifest(bundle: DocumentEvidenceBundle) -> Mapping[str, Any]:
    return {
        "contract": "rigorousrag-document-evidence-bundle-v1",
        **asdict(bundle),
        "raw_text_included": False,
        "embedding_vectors_included": False,
    }


__all__ = [
    "DocumentEvidenceBundle",
    "PageEmbeddingReference",
    "RegionReference",
    "build_document_evidence_bundle",
    "bundle_manifest",
]
