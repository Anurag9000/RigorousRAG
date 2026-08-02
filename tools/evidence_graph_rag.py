"""Bounded read-only evidence selection over authoritative graph sets."""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import Any

from tools.evidence_graph_retrieval import outgoing_neighbors, search_nodes
from tools.evidence_graph_types import EDGE_TYPES, NODE_TYPES
from tools.evidence_graph_set_store import EvidenceGraphSetAuthorityReport
from tools.evidence_graph_sets import EvidenceGraphSet, _CROSS_EDGE_TYPES
from tools.index_coordinator import _document_lock
from tools.security import normalize_owner_id

_MAX_MEMBERS = 1_000
_MAX_RESULTS = 2_000
_MAX_PER_DOCUMENT = 100
_MAX_EXPANSIONS = 5_000
_MAX_DEPTH = 6
_MAX_QUERY_CHARS = 20_000


def _identifier(value: Any, label: str, maximum: int = 500) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(
        ord(character) < 32 or ord(character) == 127 for character in cleaned
    ):
        raise ValueError(f"{label} is invalid.")
    return cleaned


def _digest(value: Any, label: str) -> str:
    cleaned = _identifier(value, label, 64).lower()
    if len(cleaned) != 64 or any(character not in "0123456789abcdef" for character in cleaned):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return cleaned


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return value


def _finite(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and at least {minimum}.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite and at least {minimum}.") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}.")
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


def _node_provenance(node: Any) -> str:
    value = getattr(node, "provenance_digest", None)
    if isinstance(value, str):
        return _digest(value, "node provenance_digest")
    return _sha256(asdict(node))


def _edge_provenance(edge: Any) -> str:
    value = getattr(edge, "provenance_digest", None)
    if isinstance(value, str):
        return _digest(value, "edge provenance_digest")
    return _sha256(asdict(edge))


def _query_digest(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("query must be text.")
    cleaned = query.strip()
    if not cleaned or len(cleaned) > _MAX_QUERY_CHARS or "\x00" in cleaned:
        raise ValueError("query is empty, invalid or too long.")
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def _type_values(
    values: Iterable[str] | None,
    label: str,
    *,
    allowed: frozenset[str],
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be an iterable.")
    result = tuple(sorted(set(_identifier(value, label, 50) for value in values)))
    if not result:
        raise ValueError(f"{label} may not be empty.")
    if any(value not in allowed for value in result):
        raise ValueError(f"{label} contains unsupported values.")
    return result


@dataclass(frozen=True)
class GraphTraversalStep:
    traversal_kind: str
    source_doc_id: str
    source_generation: int
    source_node_id: str
    edge_id: str
    edge_type: str
    edge_provenance_digest: str
    target_doc_id: str
    target_generation: int
    target_node_id: str
    depth: int
    weight: float

    def __post_init__(self) -> None:
        if self.traversal_kind not in {"within_document", "cross_document"}:
            raise ValueError("traversal_kind is unsupported.")
        for name in ("source_doc_id", "target_doc_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name, 200))
        for name in (
            "source_node_id",
            "edge_id",
            "edge_provenance_digest",
            "target_node_id",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "edge_type", _identifier(self.edge_type, "edge_type", 50))
        object.__setattr__(
            self,
            "source_generation",
            _integer(self.source_generation, "source_generation", 1, 2**63 - 1),
        )
        object.__setattr__(
            self,
            "target_generation",
            _integer(self.target_generation, "target_generation", 1, 2**63 - 1),
        )
        object.__setattr__(self, "depth", _integer(self.depth, "depth", 1, _MAX_DEPTH))
        object.__setattr__(self, "weight", _finite(self.weight, "weight"))

    @property
    def step_digest(self) -> str:
        return _sha256(asdict(self))


@dataclass(frozen=True)
class GraphEvidenceItem:
    owner_id: str
    doc_id: str
    generation: int
    graph_digest: str
    node_id: str
    node_type: str
    label: str
    text: str
    page_number: int | None
    section: str | None
    score: float
    matched_terms: tuple[str, ...]
    provenance_digest: str
    origin: str
    lineage_step_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(self, "doc_id", _identifier(self.doc_id, "doc_id", 200))
        object.__setattr__(
            self, "generation", _integer(self.generation, "generation", 1, 2**63 - 1)
        )
        object.__setattr__(self, "graph_digest", _digest(self.graph_digest, "graph_digest"))
        object.__setattr__(self, "node_id", _digest(self.node_id, "node_id"))
        object.__setattr__(self, "node_type", _identifier(self.node_type, "node_type", 50))
        object.__setattr__(self, "label", _identifier(self.label, "label", 2_000))
        if not isinstance(self.text, str) or "\x00" in self.text:
            raise ValueError("text must be privacy-finalized text.")
        if self.page_number is not None:
            object.__setattr__(
                self,
                "page_number",
                _integer(self.page_number, "page_number", 1, 1_000_000),
            )
        if self.section is not None:
            object.__setattr__(self, "section", _identifier(self.section, "section", 2_000))
        object.__setattr__(self, "score", _finite(self.score, "score"))
        terms = tuple(sorted(set(self.matched_terms)))
        if any(not isinstance(term, str) or not term for term in terms):
            raise ValueError("matched_terms are invalid.")
        object.__setattr__(self, "matched_terms", terms)
        object.__setattr__(
            self, "provenance_digest", _digest(self.provenance_digest, "provenance_digest")
        )
        if self.origin not in {"lexical", "within_document", "cross_document"}:
            raise ValueError("origin is unsupported.")
        lineage = tuple(
            _digest(value, "lineage_step_digest")
            for value in self.lineage_step_digests
        )
        if self.origin == "lexical" and lineage:
            raise ValueError("lexical evidence may not contain traversal lineage.")
        if self.origin != "lexical" and not lineage:
            raise ValueError("expanded evidence requires traversal lineage.")
        object.__setattr__(self, "lineage_step_digests", lineage)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def evidence_digest(self) -> str:
        return _sha256(
            {
                "owner_id": self.owner_id,
                "doc_id": self.doc_id,
                "generation": self.generation,
                "graph_digest": self.graph_digest,
                "node_id": self.node_id,
                "node_type": self.node_type,
                "label": self.label,
                "text_sha256": self.text_sha256,
                "page_number": self.page_number,
                "section": self.section,
                "score": self.score,
                "matched_terms": self.matched_terms,
                "provenance_digest": self.provenance_digest,
                "origin": self.origin,
                "lineage_step_digests": self.lineage_step_digests,
            }
        )


@dataclass(frozen=True)
class GraphEvidenceSelection:
    owner_id: str
    graph_set_key: str
    graph_set_id: str
    graph_set_digest: str
    authority_digest: str
    query_digest: str
    items: tuple[GraphEvidenceItem, ...]
    traversals: tuple[GraphTraversalStep, ...]
    lexical_seed_count: int
    expanded_count: int
    estimated_work_units: int
    abstained: bool
    citation_conversion_performed: bool = False
    answer_generated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_id", normalize_owner_id(self.owner_id))
        object.__setattr__(
            self, "graph_set_key", _identifier(self.graph_set_key, "graph_set_key", 500)
        )
        for name in (
            "graph_set_id",
            "graph_set_digest",
            "authority_digest",
            "query_digest",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if not isinstance(self.items, tuple) or len(self.items) > _MAX_RESULTS:
            raise ValueError("items must be a bounded tuple.")
        if not isinstance(self.traversals, tuple) or len(self.traversals) > _MAX_EXPANSIONS:
            raise ValueError("traversals must be a bounded tuple.")
        if len({(item.doc_id, item.generation, item.node_id) for item in self.items}) != len(self.items):
            raise ValueError("evidence items must be unique by generation-scoped node.")
        step_digests = {step.step_digest for step in self.traversals}
        if len(step_digests) != len(self.traversals):
            raise ValueError("traversal steps must be unique.")
        if any(
            digest not in step_digests
            for item in self.items
            for digest in item.lineage_step_digests
        ):
            raise ValueError("evidence lineage references an unavailable traversal step.")
        object.__setattr__(
            self,
            "lexical_seed_count",
            _integer(self.lexical_seed_count, "lexical_seed_count", 0, _MAX_RESULTS),
        )
        object.__setattr__(
            self,
            "expanded_count",
            _integer(self.expanded_count, "expanded_count", 0, _MAX_RESULTS),
        )
        object.__setattr__(
            self,
            "estimated_work_units",
            _integer(self.estimated_work_units, "estimated_work_units", 0, 10_000_000),
        )
        if self.lexical_seed_count != sum(item.origin == "lexical" for item in self.items):
            raise ValueError("lexical_seed_count differs from retained lexical evidence.")
        if self.expanded_count != sum(item.origin != "lexical" for item in self.items):
            raise ValueError("expanded_count differs from retained expanded evidence.")
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean.")
        if self.abstained != (len(self.items) == 0):
            raise ValueError("abstained must exactly reflect empty evidence.")
        if self.citation_conversion_performed or self.answer_generated:
            raise ValueError("this selection layer may not convert citations or generate answers.")

    @property
    def selection_digest(self) -> str:
        return _sha256(
            {
                "owner_id": self.owner_id,
                "graph_set_key": self.graph_set_key,
                "graph_set_id": self.graph_set_id,
                "graph_set_digest": self.graph_set_digest,
                "authority_digest": self.authority_digest,
                "query_digest": self.query_digest,
                "items": [item.evidence_digest for item in self.items],
                "traversals": [step.step_digest for step in self.traversals],
                "lexical_seed_count": self.lexical_seed_count,
                "expanded_count": self.expanded_count,
                "estimated_work_units": self.estimated_work_units,
                "abstained": self.abstained,
            }
        )


def _materialize_node(
    member: Any,
    batch: Any,
    node: Any,
    *,
    score: float,
    matched_terms: tuple[str, ...],
    origin: str,
    lineage: tuple[str, ...],
) -> GraphEvidenceItem:
    if (
        batch.owner_id != member.owner_id
        or batch.doc_id != member.doc_id
        or batch.generation != member.generation
        or batch.content_sha256 != member.content_sha256
        or batch.profile_fingerprint != member.profile_fingerprint
        or batch.graph_digest != member.graph_digest
    ):
        raise RuntimeError("member graph differs from graph-set identity.")
    return GraphEvidenceItem(
        owner_id=member.owner_id,
        doc_id=member.doc_id,
        generation=member.generation,
        graph_digest=member.graph_digest,
        node_id=node.node_id,
        node_type=node.node_type,
        label=node.label,
        text=node.text,
        page_number=node.page_number,
        section=node.section,
        score=score,
        matched_terms=matched_terms,
        provenance_digest=_node_provenance(node),
        origin=origin,
        lineage_step_digests=lineage,
    )


def select_graph_set_evidence(
    graph_set: EvidenceGraphSet,
    authority: EvidenceGraphSetAuthorityReport,
    *,
    query: str,
    graphs: Any,
    node_types: Iterable[str] | None = None,
    within_edge_types: Iterable[str] | None = None,
    cross_edge_types: Iterable[str] | None = None,
    per_document_hits: int = 10,
    max_lexical_seeds: int = 100,
    max_within_per_seed: int = 3,
    max_cross_depth: int = 2,
    max_cross_per_seed: int = 20,
    max_total_items: int = 200,
) -> GraphEvidenceSelection:
    """Select provenance-rich evidence without generating an answer or citations."""

    if not isinstance(graph_set, EvidenceGraphSet):
        raise ValueError("graph_set must be EvidenceGraphSet.")
    if not isinstance(authority, EvidenceGraphSetAuthorityReport):
        raise ValueError("authority must be EvidenceGraphSetAuthorityReport.")
    if (
        not authority.authoritative_current
        or authority.graph_set_id != graph_set.graph_set_id
        or authority.graph_set_digest != graph_set.graph_set_digest
    ):
        raise RuntimeError("graph set is not authoritative current.")
    query_hash = _query_digest(query)
    member_count = _integer(len(graph_set.members), "member_count", 2, _MAX_MEMBERS)
    per_doc = _integer(per_document_hits, "per_document_hits", 1, _MAX_PER_DOCUMENT)
    seed_limit = _integer(max_lexical_seeds, "max_lexical_seeds", 1, _MAX_RESULTS)
    within_limit = _integer(
        max_within_per_seed, "max_within_per_seed", 0, _MAX_PER_DOCUMENT
    )
    cross_depth = _integer(max_cross_depth, "max_cross_depth", 0, _MAX_DEPTH)
    cross_limit = _integer(
        max_cross_per_seed, "max_cross_per_seed", 0, _MAX_EXPANSIONS
    )
    total_limit = _integer(max_total_items, "max_total_items", 1, _MAX_RESULTS)
    selected_node_types = _type_values(node_types, "node_types", allowed=NODE_TYPES)
    selected_within_edges = _type_values(
        within_edge_types, "within_edge_types", allowed=EDGE_TYPES
    )
    selected_cross_edges = _type_values(
        cross_edge_types, "cross_edge_types", allowed=_CROSS_EDGE_TYPES
    )

    batches: dict[str, Any] = {}
    members = {member.doc_id: member for member in graph_set.members}
    node_maps: dict[str, dict[str, Any]] = {}
    lexical: list[GraphEvidenceItem] = []
    for member in graph_set.members:
        batch = graphs.get(
            owner_id=member.owner_id,
            doc_id=member.doc_id,
            generation=member.generation,
        )
        if (
            batch.graph_digest != member.graph_digest
            or batch.content_sha256 != member.content_sha256
            or batch.profile_fingerprint != member.profile_fingerprint
        ):
            raise RuntimeError("stored member graph differs from graph-set reference.")
        batches[member.doc_id] = batch
        node_maps[member.doc_id] = {node.node_id: node for node in batch.nodes}
        hits = search_nodes(
            batch,
            query,
            node_types=selected_node_types,
            limit=per_doc,
        )
        for hit in hits:
            lexical.append(
                _materialize_node(
                    member,
                    batch,
                    hit.node,
                    score=hit.score,
                    matched_terms=tuple(hit.matched_terms),
                    origin="lexical",
                    lineage=(),
                )
            )
    lexical.sort(key=lambda item: (-item.score, item.doc_id, item.node_type, item.node_id))
    seeds = lexical[:seed_limit]

    best: dict[tuple[str, int, str], GraphEvidenceItem] = {
        (item.doc_id, item.generation, item.node_id): item for item in seeds
    }
    steps: dict[str, GraphTraversalStep] = {}
    expansions = 0

    def add_expanded(item: GraphEvidenceItem) -> None:
        nonlocal expansions
        key = (item.doc_id, item.generation, item.node_id)
        previous = best.get(key)
        priority = {"lexical": 0, "cross_document": 1, "within_document": 2}
        if previous is None or (-item.score, priority[item.origin], item.evidence_digest) < (
            -previous.score,
            priority[previous.origin],
            previous.evidence_digest,
        ):
            best[key] = item
        expansions += 1

    for seed in seeds:
        if expansions >= _MAX_EXPANSIONS:
            break
        batch = batches[seed.doc_id]
        member = members[seed.doc_id]
        if within_limit:
            for edge, target in outgoing_neighbors(
                batch,
                seed.node_id,
                edge_types=selected_within_edges,
                limit=within_limit,
            ):
                step = GraphTraversalStep(
                    traversal_kind="within_document",
                    source_doc_id=seed.doc_id,
                    source_generation=seed.generation,
                    source_node_id=seed.node_id,
                    edge_id=edge.edge_id,
                    edge_type=edge.edge_type,
                    edge_provenance_digest=_edge_provenance(edge),
                    target_doc_id=seed.doc_id,
                    target_generation=seed.generation,
                    target_node_id=target.node_id,
                    depth=1,
                    weight=edge.weight,
                )
                steps[step.step_digest] = step
                add_expanded(
                    _materialize_node(
                        member,
                        batch,
                        target,
                        score=seed.score * edge.weight * 0.7,
                        matched_terms=seed.matched_terms,
                        origin="within_document",
                        lineage=(step.step_digest,),
                    )
                )
                if expansions >= _MAX_EXPANSIONS:
                    break

        if not cross_depth or not cross_limit or expansions >= _MAX_EXPANSIONS:
            continue
        start = (seed.doc_id, seed.node_id)
        queue = deque([(start, seed.score, (), 0)])
        visited = {start}
        traversed = 0
        while queue and traversed < cross_limit and expansions < _MAX_EXPANSIONS:
            (doc_id, node_id), score, lineage, depth = queue.popleft()
            if depth >= cross_depth:
                continue
            outgoing = [
                edge
                for edge in graph_set.edges
                if edge.source.doc_id == doc_id
                and edge.source.node_id == node_id
                and (
                    selected_cross_edges is None
                    or edge.edge_type in selected_cross_edges
                )
            ]
            outgoing.sort(key=lambda edge: edge.edge_id)
            for edge in outgoing:
                target_key = (edge.target.doc_id, edge.target.node_id)
                if target_key in visited:
                    continue
                target_member = members.get(edge.target.doc_id)
                target_batch = batches.get(edge.target.doc_id)
                target_node = node_maps.get(edge.target.doc_id, {}).get(edge.target.node_id)
                if target_member is None or target_batch is None or target_node is None:
                    raise RuntimeError("cross-document edge target is unavailable.")
                if (
                    edge.target.generation != target_member.generation
                    or edge.target.graph_digest != target_member.graph_digest
                    or edge.target.provenance_digest != _node_provenance(target_node)
                ):
                    raise RuntimeError("cross-document edge target provenance changed.")
                step = GraphTraversalStep(
                    traversal_kind="cross_document",
                    source_doc_id=edge.source.doc_id,
                    source_generation=edge.source.generation,
                    source_node_id=edge.source.node_id,
                    edge_id=edge.edge_id,
                    edge_type=edge.edge_type,
                    edge_provenance_digest=edge.provenance_digest,
                    target_doc_id=edge.target.doc_id,
                    target_generation=edge.target.generation,
                    target_node_id=edge.target.node_id,
                    depth=depth + 1,
                    weight=edge.weight,
                )
                steps[step.step_digest] = step
                next_lineage = lineage + (step.step_digest,)
                next_score = score * edge.weight * 0.8
                add_expanded(
                    _materialize_node(
                        target_member,
                        target_batch,
                        target_node,
                        score=next_score,
                        matched_terms=seed.matched_terms,
                        origin="cross_document",
                        lineage=next_lineage,
                    )
                )
                visited.add(target_key)
                queue.append((target_key, next_score, next_lineage, depth + 1))
                traversed += 1
                if traversed >= cross_limit or expansions >= _MAX_EXPANSIONS:
                    break

    values = sorted(
        best.values(),
        key=lambda item: (
            -item.score,
            {"lexical": 0, "cross_document": 1, "within_document": 2}[item.origin],
            item.doc_id,
            item.node_type,
            item.node_id,
        ),
    )[:total_limit]
    selected_lineage = {
        digest for item in values for digest in item.lineage_step_digests
    }
    traversals = tuple(
        sorted(
            (step for digest, step in steps.items() if digest in selected_lineage),
            key=lambda step: (step.depth, step.traversal_kind, step.step_digest),
        )
    )
    lexical_count = sum(item.origin == "lexical" for item in values)
    expanded_count = len(values) - lexical_count
    return GraphEvidenceSelection(
        owner_id=graph_set.owner_id,
        graph_set_key=graph_set.graph_set_key,
        graph_set_id=graph_set.graph_set_id,
        graph_set_digest=graph_set.graph_set_digest,
        authority_digest=authority.authority_digest,
        query_digest=query_hash,
        items=tuple(values),
        traversals=traversals,
        lexical_seed_count=lexical_count,
        expanded_count=expanded_count,
        estimated_work_units=member_count * per_doc + expansions + len(graph_set.edges),
        abstained=not values,
    )


def select_current_graph_set_evidence(
    *,
    owner_id: str,
    graph_set_key: str,
    query: str,
    set_store: Any,
    generations: Any,
    graphs: Any,
    **kwargs: Any,
) -> GraphEvidenceSelection:
    initial_set, _initial_authority = set_store.resolve_current(
        owner_id=owner_id,
        graph_set_key=graph_set_key,
        generations=generations,
        graphs=graphs,
    )
    with ExitStack() as stack:
        for member in sorted(initial_set.members, key=lambda item: item.doc_id):
            stack.enter_context(_document_lock(initial_set.owner_id, member.doc_id))
        graph_set, authority = set_store.resolve_current(
            owner_id=owner_id,
            graph_set_key=graph_set_key,
            generations=generations,
            graphs=graphs,
        )
        if graph_set.graph_set_id != initial_set.graph_set_id:
            raise RuntimeError("graph set changed before evidence selection.")
        selection = select_graph_set_evidence(
            graph_set,
            authority,
            query=query,
            graphs=graphs,
            **kwargs,
        )
        final_set, final_authority = set_store.resolve_current(
            owner_id=owner_id,
            graph_set_key=graph_set_key,
            generations=generations,
            graphs=graphs,
        )
        if (
            final_set.graph_set_id != graph_set.graph_set_id
            or final_set.graph_set_digest != graph_set.graph_set_digest
            or final_authority.authority_digest != authority.authority_digest
        ):
            raise RuntimeError("graph set changed during evidence selection.")
        return selection


__all__ = [
    "GraphEvidenceItem",
    "GraphEvidenceSelection",
    "GraphTraversalStep",
    "select_current_graph_set_evidence",
    "select_graph_set_evidence",
]
