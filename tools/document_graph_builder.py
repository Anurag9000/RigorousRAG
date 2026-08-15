"""Deterministic authoritative-document -> evidence-graph construction.

The builder maps immutable document/page/block structures into graph nodes and typed
cross-modal edges.  It never synthesizes semantic support/contradiction edges by itself;
those require separately governed claim/extractor evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from tools.document_ir import DocumentBlock, ScientificDocumentIR
from tools.graph_reasoning import EvidenceGraph, GraphEdge, GraphNode


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


def _node_id(*parts: str) -> str:
    return hashlib.sha256(_canonical(parts)).hexdigest()


def _edge_id(source: str, target: str, kind: str, evidence: Sequence[str]) -> str:
    return hashlib.sha256(_canonical((source, target, kind, tuple(evidence)))).hexdigest()


def _block_kind(block: DocumentBlock) -> str:
    if block.role in {"table", "figure", "formula"}:
        return block.role
    if block.role == "chart":
        return "figure"
    if block.role in {"heading", "paragraph", "list_item", "caption", "footnote", "reference", "title"}:
        return "section"
    return "other"


def build_document_evidence_graph(
    document: ScientificDocumentIR,
    *,
    generation_id: str,
    extra_nodes: Sequence[GraphNode] = (),
    extra_edges: Sequence[GraphEdge] = (),
) -> EvidenceGraph:
    if not isinstance(document, ScientificDocumentIR):
        raise TypeError("document must be ScientificDocumentIR")
    generation = str(generation_id).strip()
    if not generation or len(generation) > 256:
        raise ValueError("generation_id is invalid")

    document_node_id = _node_id("document", document.owner_id, document.doc_id, document.source_sha256, generation)
    nodes: list[GraphNode] = [
        GraphNode(
            node_id=document_node_id,
            kind="document",
            source_id=document.doc_id,
            content_sha256=document.source_sha256,
            label=document.doc_id,
            attributes={
                "owner_id": document.owner_id,
                "generation": generation,
                "document_ir_fingerprint": document.fingerprint,
                "extractor_id": document.extractor_id,
                "schema_version": document.schema_version,
            },
        )
    ]
    edges: list[GraphEdge] = []
    block_node_ids: dict[str, str] = {}

    for block in document.blocks:
        node_id = _node_id("block", document_node_id, block.block_id, block.content_sha256)
        block_node_ids[block.block_id] = node_id
        label = block.text[:1000] if block.text else block.role
        attributes: dict[str, str] = {
            "block_id": block.block_id,
            "page_number": str(block.page_number),
            "role": block.role,
            "bbox": json.dumps(asdict(block.bbox), sort_keys=True, separators=(",", ":")),
            "generation": generation,
        }
        if block.table is not None:
            attributes["table_cells"] = str(len(block.table.cells))
            if block.table.caption:
                attributes["caption_sha256"] = hashlib.sha256(block.table.caption.encode("utf-8")).hexdigest()
        if block.figure is not None:
            attributes["figure_panels"] = str(len(block.figure.panels))
            if block.figure.caption:
                attributes["caption_sha256"] = hashlib.sha256(block.figure.caption.encode("utf-8")).hexdigest()
        if block.formula is not None:
            attributes["formula_sha256"] = hashlib.sha256(block.formula.normalized_text.encode("utf-8")).hexdigest()
        nodes.append(
            GraphNode(
                node_id=node_id,
                kind=_block_kind(block),
                source_id=document.doc_id,
                content_sha256=block.content_sha256,
                label=label,
                attributes=attributes,
            )
        )
        edge_id = _edge_id(document_node_id, node_id, "contains", (document.doc_id,))
        edges.append(GraphEdge(edge_id, document_node_id, node_id, "contains", 1.0, (document.doc_id,)))

    link_kind_map = {
        "contains": "contains",
        "caption_of": "references",
        "refers_to": "references",
        "defines": "references",
        "continues": "references",
        "table_cell_of": "contains",
        "panel_of": "contains",
        "equation_ref": "references",
        "citation_ref": "cites",
        "same_entity": "same_entity",
        "reading_next": "references",
    }
    for link in document.links:
        source = block_node_ids[link.source_block_id]
        target = block_node_ids[link.target_block_id]
        kind = link_kind_map[link.kind]
        evidence = (document.doc_id, link.source_block_id, link.target_block_id)
        edges.append(GraphEdge(_edge_id(source, target, kind, evidence), source, target, kind, link.confidence, evidence))

    extra_node_ids = {node.node_id for node in extra_nodes}
    if document_node_id in extra_node_ids or any(node.node_id in block_node_ids.values() for node in extra_nodes):
        raise ValueError("extra_nodes collide with document-derived graph identities")
    nodes.extend(extra_nodes)
    known_nodes = {node.node_id for node in nodes}
    for edge in extra_edges:
        if edge.source_node_id not in known_nodes or edge.target_node_id not in known_nodes:
            raise ValueError("extra edge references a node outside the combined graph")
        edges.append(edge)
    return EvidenceGraph(tuple(nodes), tuple(edges))


def graph_generation_manifest(
    document: ScientificDocumentIR,
    graph: EvidenceGraph,
    *,
    generation_id: str,
) -> Mapping[str, Any]:
    return {
        "contract": "rigorousrag-document-graph-generation-v1",
        "owner_id": document.owner_id,
        "doc_id": document.doc_id,
        "source_sha256": document.source_sha256,
        "generation_id": str(generation_id),
        "document_ir_fingerprint": document.fingerprint,
        "graph_fingerprint": graph.fingerprint,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "semantic_support_edges_generated": False,
    }


__all__ = ["build_document_evidence_graph", "graph_generation_manifest"]
