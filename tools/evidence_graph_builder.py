"""Deterministic evidence-graph construction from finalized documents and explicit annotations."""

from __future__ import annotations

import hashlib
import itertools
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tools.evidence_graph_types import (
    EDGE_TYPES,
    NODE_TYPES,
    EvidenceEdge,
    EvidenceGraphBatch,
    EvidenceNode,
    deterministic_edge_id,
    deterministic_node_id,
)
from tools.security import normalize_owner_id

_MAX_ANNOTATIONS = 50_000
_MAX_RELATIONS = 250_000
_MAX_SECTIONS = 10_000
_EXPLICIT_RELATION_TYPES = EDGE_TYPES - {"contains"}


def _identifier(value: Any, label: str, maximum: int = 2_000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping.")
    return dict(value)


def _page(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1_000_000:
        raise ValueError("page_number must be a positive integer or null.")
    return value


def _section_data(section: Any, index: int) -> tuple[str, str, int | None, dict[str, Any]]:
    if hasattr(section, "model_dump") and callable(section.model_dump):
        raw = section.model_dump()
    elif isinstance(section, Mapping):
        raw = dict(section)
    else:
        raw = {
            "title": getattr(section, "title", f"Section {index + 1}"),
            "content": getattr(section, "content", None),
            "page_number": getattr(section, "page_number", None),
            "metadata": getattr(section, "metadata", {}),
        }
    if not isinstance(raw, Mapping):
        raise ValueError("section must be object-like.")
    title = _identifier(raw.get("title") or f"Section {index + 1}", "section title", 2_000)
    content = raw.get("content")
    if not isinstance(content, str) or not content.strip() or len(content) > 5_000_000:
        raise ValueError("section content is invalid.")
    return title, content.strip(), _page(raw.get("page_number")), _metadata(raw.get("metadata"))


@dataclass(frozen=True)
class GraphAnnotation:
    annotation_key: str
    node_type: str
    label: str
    text: str = ""
    section_index: int | None = None
    page_number: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotation_key", _identifier(self.annotation_key, "annotation_key"))
        kind = _identifier(self.node_type, "node_type", 50)
        if kind not in NODE_TYPES - {"document", "section"}:
            raise ValueError("annotation node_type is unsupported.")
        object.__setattr__(self, "node_type", kind)
        object.__setattr__(self, "label", _identifier(self.label, "label", 2_000))
        if not isinstance(self.text, str) or len(self.text) > 5_000_000 or "\x00" in self.text:
            raise ValueError("annotation text is invalid.")
        object.__setattr__(self, "text", self.text.strip())
        if self.section_index is not None:
            if isinstance(self.section_index, bool) or not isinstance(self.section_index, int) or self.section_index < 0:
                raise ValueError("section_index must be a non-negative integer or null.")
        object.__setattr__(self, "page_number", _page(self.page_number))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True)
class ExplicitGraphRelation:
    relation_key: str
    source_key: str
    target_key: str
    edge_type: str
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_key", _identifier(self.relation_key, "relation_key"))
        object.__setattr__(self, "source_key", _identifier(self.source_key, "source_key"))
        object.__setattr__(self, "target_key", _identifier(self.target_key, "target_key"))
        kind = _identifier(self.edge_type, "edge_type", 50)
        if kind not in _EXPLICIT_RELATION_TYPES:
            raise ValueError("explicit edge_type is unsupported.")
        object.__setattr__(self, "edge_type", kind)
        if isinstance(self.weight, bool):
            raise ValueError("weight must be numeric.")
        try:
            numeric = float(self.weight)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("weight must be numeric.") from exc
        object.__setattr__(self, "weight", numeric)
        object.__setattr__(self, "metadata", _metadata(self.metadata))


def _bounded_tuple(values: Iterable[Any] | None, maximum: int, label: str) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable.")
    try:
        result = tuple(itertools.islice(iter(values), maximum + 1))
    except Exception as exc:
        raise ValueError(f"{label} is not safely iterable.") from exc
    if len(result) > maximum:
        raise ValueError(f"{label} exceeds the item limit.")
    return result


def _node(
    *,
    owner_id: str,
    doc_id: str,
    generation: int,
    node_type: str,
    natural_key: str,
    label: str,
    text: str,
    page_number: int | None,
    section: str | None,
    metadata: Mapping[str, Any],
) -> EvidenceNode:
    return EvidenceNode(
        node_id=deterministic_node_id(
            owner_id=owner_id,
            doc_id=doc_id,
            generation=generation,
            node_type=node_type,
            natural_key=natural_key,
        ),
        owner_id=owner_id,
        doc_id=doc_id,
        generation=generation,
        node_type=node_type,
        natural_key=natural_key,
        label=label,
        text=text,
        page_number=page_number,
        section=section,
        metadata=metadata,
    )


def _edge(
    *,
    owner_id: str,
    doc_id: str,
    generation: int,
    source_node_id: str,
    target_node_id: str,
    edge_type: str,
    relation_key: str,
    weight: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceEdge:
    return EvidenceEdge(
        edge_id=deterministic_edge_id(
            owner_id=owner_id,
            doc_id=doc_id,
            generation=generation,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            relation_key=relation_key,
        ),
        owner_id=owner_id,
        doc_id=doc_id,
        generation=generation,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        relation_key=relation_key,
        weight=weight,
        metadata={} if metadata is None else metadata,
    )


def build_evidence_graph(
    document: Any,
    *,
    owner_id: str,
    generation: int,
    profile_fingerprint: str,
    annotations: Iterable[GraphAnnotation] | None = None,
    relations: Iterable[ExplicitGraphRelation] | None = None,
    now: float | None = None,
) -> EvidenceGraphBatch:
    """Build one deterministic generation-scoped graph from explicit inputs only."""

    owner = normalize_owner_id(owner_id)
    doc_id = _identifier(getattr(document, "id", None), "document.id", 200)
    text = getattr(document, "text", None)
    if not isinstance(text, str) or not text.strip() or len(text) > 50_000_000:
        raise ValueError("document.text is invalid.")
    finalized_text = text.strip()
    content_sha256 = hashlib.sha256(finalized_text.encode("utf-8")).hexdigest()
    metadata = getattr(document, "metadata", {})
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("document.metadata must be a mapping.")
    declared = (metadata or {}).get("content_sha256")
    if declared is not None and declared != content_sha256:
        raise ValueError("document content hash does not match finalized text.")
    title = getattr(document, "title", None) or getattr(document, "filename", None) or doc_id
    title = _identifier(title, "document title", 2_000)

    raw_sections = _bounded_tuple(getattr(document, "sections", None), _MAX_SECTIONS, "sections")
    if not raw_sections:
        raise ValueError("document must contain at least one finalized section.")
    annotation_values = _bounded_tuple(annotations, _MAX_ANNOTATIONS, "annotations")
    relation_values = _bounded_tuple(relations, _MAX_RELATIONS, "relations")
    if any(not isinstance(item, GraphAnnotation) for item in annotation_values):
        raise ValueError("every annotation must be GraphAnnotation.")
    if any(not isinstance(item, ExplicitGraphRelation) for item in relation_values):
        raise ValueError("every relation must be ExplicitGraphRelation.")

    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    lookup: dict[str, EvidenceNode] = {}

    document_node = _node(
        owner_id=owner,
        doc_id=doc_id,
        generation=generation,
        node_type="document",
        natural_key="document",
        label=title,
        text="",
        page_number=None,
        section=None,
        metadata={"content_sha256": content_sha256},
    )
    nodes.append(document_node)
    lookup["document"] = document_node

    section_nodes: list[EvidenceNode] = []
    for index, raw_section in enumerate(raw_sections):
        section_title, section_text, page_number, section_metadata = _section_data(raw_section, index)
        natural_key = f"section:{index}"
        section_node = _node(
            owner_id=owner,
            doc_id=doc_id,
            generation=generation,
            node_type="section",
            natural_key=natural_key,
            label=section_title,
            text=section_text,
            page_number=page_number,
            section=section_title,
            metadata={"section_index": index, **section_metadata},
        )
        nodes.append(section_node)
        section_nodes.append(section_node)
        lookup[natural_key] = section_node
        edges.append(
            _edge(
                owner_id=owner,
                doc_id=doc_id,
                generation=generation,
                source_node_id=document_node.node_id,
                target_node_id=section_node.node_id,
                edge_type="contains",
                relation_key=f"document-contains-section:{index}",
                metadata={"section_index": index},
            )
        )

    annotation_keys: set[str] = set()
    for annotation in annotation_values:
        if annotation.annotation_key in annotation_keys or annotation.annotation_key in lookup:
            raise ValueError("annotation keys must be unique and may not shadow reserved keys.")
        annotation_keys.add(annotation.annotation_key)
        parent = document_node
        section_name: str | None = None
        page_number = annotation.page_number
        if annotation.section_index is not None:
            if annotation.section_index >= len(section_nodes):
                raise ValueError("annotation section_index is out of range.")
            parent = section_nodes[annotation.section_index]
            section_name = parent.label
            if page_number is None:
                page_number = parent.page_number
        annotation_node = _node(
            owner_id=owner,
            doc_id=doc_id,
            generation=generation,
            node_type=annotation.node_type,
            natural_key=f"annotation:{annotation.annotation_key}",
            label=annotation.label,
            text=annotation.text,
            page_number=page_number,
            section=section_name,
            metadata={"annotation_key": annotation.annotation_key, **dict(annotation.metadata)},
        )
        nodes.append(annotation_node)
        lookup[annotation.annotation_key] = annotation_node
        edges.append(
            _edge(
                owner_id=owner,
                doc_id=doc_id,
                generation=generation,
                source_node_id=parent.node_id,
                target_node_id=annotation_node.node_id,
                edge_type="contains",
                relation_key=f"container:{parent.natural_key}:{annotation.annotation_key}",
                metadata={"explicit_annotation": True},
            )
        )

    relation_keys: set[str] = set()
    for relation in relation_values:
        if relation.relation_key in relation_keys:
            raise ValueError("relation keys must be unique.")
        relation_keys.add(relation.relation_key)
        try:
            source = lookup[relation.source_key]
            target = lookup[relation.target_key]
        except KeyError as exc:
            raise ValueError("explicit relation references an unknown node key.") from exc
        edges.append(
            _edge(
                owner_id=owner,
                doc_id=doc_id,
                generation=generation,
                source_node_id=source.node_id,
                target_node_id=target.node_id,
                edge_type=relation.edge_type,
                relation_key=relation.relation_key,
                weight=relation.weight,
                metadata={"explicit_relation": True, **dict(relation.metadata)},
            )
        )

    return EvidenceGraphBatch(
        owner_id=owner,
        doc_id=doc_id,
        generation=generation,
        content_sha256=content_sha256,
        profile_fingerprint=profile_fingerprint,
        nodes=tuple(nodes),
        edges=tuple(edges),
        created_at=time.time() if now is None else now,
    )


__all__ = [
    "ExplicitGraphRelation",
    "GraphAnnotation",
    "build_evidence_graph",
]
