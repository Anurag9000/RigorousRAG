"""External evidence-graph backend contracts and an injected Neo4j implementation.

The implementation deliberately accepts an already-created driver. Importing this module
never imports a Neo4j SDK, persists credentials, or implies that a live cluster was tested.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


def _clean_identifier(value: Any, label: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split())
    if not cleaned:
        raise ValueError(f"{label} must be non-empty")
    if len(cleaned) > maximum:
        raise ValueError(f"{label} exceeds {maximum} characters")
    return cleaned


def _json_metadata(value: Any) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class ExternalGraphNode:
    node_id: str
    kind: str
    content: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_id", _clean_identifier(self.node_id, "node_id"))
        object.__setattr__(self, "kind", _clean_identifier(self.kind, "kind", maximum=100))
        if not isinstance(self.content, str):
            raise ValueError("content must be a string")
        _json_metadata(self.metadata)


@dataclass(frozen=True)
class ExternalGraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "edge_id", _clean_identifier(self.edge_id, "edge_id"))
        object.__setattr__(self, "source_id", _clean_identifier(self.source_id, "source_id"))
        object.__setattr__(self, "target_id", _clean_identifier(self.target_id, "target_id"))
        object.__setattr__(self, "relation", _clean_identifier(self.relation, "relation", maximum=100))
        _json_metadata(self.metadata)


@dataclass(frozen=True)
class ExternalGraphCommit:
    owner_id: str
    graph_id: str
    generation: str
    node_count: int
    edge_count: int
    digest: str
    activated: bool


class EvidenceGraphExternalBackend(Protocol):
    def upsert_generation(
        self,
        *,
        owner_id: str,
        graph_id: str,
        generation: str,
        nodes: Sequence[ExternalGraphNode],
        edges: Sequence[ExternalGraphEdge],
        activate: bool = False,
    ) -> ExternalGraphCommit: ...


class Neo4jEvidenceGraphBackend:
    """Parameter-only Neo4j writer with owner and generation namespaces."""

    def __init__(self, driver: Any, *, database: Optional[str] = None) -> None:
        if driver is None or not hasattr(driver, "session"):
            raise TypeError("driver must provide session()")
        self._driver = driver
        self._database = _clean_identifier(database, "database") if database is not None else None

    @staticmethod
    def _digest(
        owner_id: str,
        graph_id: str,
        generation: str,
        nodes: Sequence[ExternalGraphNode],
        edges: Sequence[ExternalGraphEdge],
    ) -> str:
        payload = {
            "owner_id": owner_id,
            "graph_id": graph_id,
            "generation": generation,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "kind": n.kind,
                    "content": n.content,
                    "metadata": dict(n.metadata),
                }
                for n in sorted(nodes, key=lambda item: item.node_id)
            ],
            "edges": [
                {
                    "edge_id": e.edge_id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relation": e.relation,
                    "metadata": dict(e.metadata),
                }
                for e in sorted(edges, key=lambda item: item.edge_id)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def upsert_generation(
        self,
        *,
        owner_id: str,
        graph_id: str,
        generation: str,
        nodes: Sequence[ExternalGraphNode],
        edges: Sequence[ExternalGraphEdge],
        activate: bool = False,
    ) -> ExternalGraphCommit:
        owner = _clean_identifier(owner_id, "owner_id")
        graph = _clean_identifier(graph_id, "graph_id")
        gen = _clean_identifier(generation, "generation")
        if not isinstance(activate, bool):
            raise ValueError("activate must be a boolean")
        node_ids = {node.node_id for node in nodes}
        if len(node_ids) != len(nodes):
            raise ValueError("duplicate node_id")
        edge_ids = {edge.edge_id for edge in edges}
        if len(edge_ids) != len(edges):
            raise ValueError("duplicate edge_id")
        for edge in edges:
            if edge.source_id not in node_ids or edge.target_id not in node_ids:
                raise ValueError(f"edge {edge.edge_id} references a missing node")

        digest = self._digest(owner, graph, gen, nodes, edges)
        params = {
            "owner_id": owner,
            "graph_id": graph,
            "generation": gen,
            "digest": digest,
            "activate": activate,
            "nodes": [
                {
                    "node_id": item.node_id,
                    "kind": item.kind,
                    "content": item.content,
                    "metadata_json": _json_metadata(item.metadata),
                }
                for item in nodes
            ],
            "edges": [
                {
                    "edge_id": item.edge_id,
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "relation": item.relation,
                    "metadata_json": _json_metadata(item.metadata),
                }
                for item in edges
            ],
        }

        def write(tx: Any) -> None:
            tx.run(
                """
                MERGE (g:EvidenceGraph {owner_id: $owner_id, graph_id: $graph_id})
                MERGE (v:EvidenceGeneration {owner_id: $owner_id, graph_id: $graph_id, generation: $generation})
                SET v.digest = $digest, v.active = CASE WHEN $activate THEN true ELSE coalesce(v.active, false) END
                MERGE (g)-[:HAS_GENERATION]->(v)
                """,
                **params,
            )
            tx.run(
                """
                UNWIND $nodes AS row
                MATCH (v:EvidenceGeneration {owner_id: $owner_id, graph_id: $graph_id, generation: $generation})
                MERGE (n:EvidenceNode {owner_id: $owner_id, graph_id: $graph_id, generation: $generation, node_id: row.node_id})
                SET n.kind = row.kind, n.content = row.content, n.metadata_json = row.metadata_json
                MERGE (v)-[:CONTAINS]->(n)
                """,
                **params,
            )
            tx.run(
                """
                UNWIND $edges AS row
                MATCH (s:EvidenceNode {owner_id: $owner_id, graph_id: $graph_id, generation: $generation, node_id: row.source_id})
                MATCH (t:EvidenceNode {owner_id: $owner_id, graph_id: $graph_id, generation: $generation, node_id: row.target_id})
                MERGE (s)-[r:EVIDENCE_RELATION {edge_id: row.edge_id}]->(t)
                SET r.relation = row.relation, r.metadata_json = row.metadata_json
                """,
                **params,
            )
            if activate:
                tx.run(
                    """
                    MATCH (v:EvidenceGeneration {owner_id: $owner_id, graph_id: $graph_id})
                    SET v.active = (v.generation = $generation)
                    """,
                    **params,
                )

        session_kwargs = {"database": self._database} if self._database is not None else {}
        with self._driver.session(**session_kwargs) as session:
            if hasattr(session, "execute_write"):
                session.execute_write(write)
            elif hasattr(session, "write_transaction"):
                session.write_transaction(write)
            else:
                write(session)

        return ExternalGraphCommit(owner, graph, gen, len(nodes), len(edges), digest, activate)
