"""Retraction/supersession propagation across evidence, graph and report layers.

Source-status events are explicit provenance metadata.  They invalidate or warn on
*derived* evidence that depends on affected sources without deleting historical records
or asserting that unaffected sources are scientifically correct.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

from tools.graph_reasoning import EvidenceGraph, GraphEdge, GraphNode
from tools.research_report import EvidenceMatrixRow, ResearchReport

_ALLOWED_STATUS = frozenset({"active", "retracted", "superseded", "withdrawn", "corrected"})


def _text(value: Any, label: str, maximum: int = 1000, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    cleaned = " ".join(value.replace("\x00", " ").split())
    if (not cleaned and not allow_empty) or len(cleaned) > maximum:
        raise ValueError(f"{label} is invalid")
    return cleaned


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")


@dataclass(frozen=True)
class SourceStatusEvent:
    source_id: str
    status: str
    effective_at: float
    event_source_id: str
    replacement_source_id: str = ""
    reason: str = ""
    event_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", 1000))
        status = _text(self.status, "status", 32).lower()
        if status not in _ALLOWED_STATUS:
            raise ValueError("unsupported source status")
        object.__setattr__(self, "status", status)
        timestamp = float(self.effective_at)
        if timestamp < 0:
            raise ValueError("effective_at is invalid")
        object.__setattr__(self, "effective_at", timestamp)
        object.__setattr__(self, "event_source_id", _text(self.event_source_id, "event_source_id", 1000))
        object.__setattr__(self, "replacement_source_id", _text(self.replacement_source_id, "replacement_source_id", 1000, allow_empty=True))
        object.__setattr__(self, "reason", _text(self.reason, "reason", 5000, allow_empty=True))
        payload = {
            "source_id": self.source_id,
            "status": self.status,
            "effective_at": self.effective_at,
            "event_source_id": self.event_source_id,
            "replacement_source_id": self.replacement_source_id,
            "reason": self.reason,
        }
        computed = hashlib.sha256(_canonical(payload)).hexdigest()
        if self.event_sha256 and self.event_sha256 != computed:
            raise ValueError("event_sha256 does not match event content")
        object.__setattr__(self, "event_sha256", computed)


@dataclass(frozen=True)
class EvidenceStatus:
    source_id: str
    status: str
    event_sha256: str
    replacement_source_id: str = ""
    usable_for_new_claims: bool = True
    warning: str = ""


def latest_source_status(events: Sequence[SourceStatusEvent], *, as_of: float | None = None) -> Mapping[str, EvidenceStatus]:
    cutoff = time.time() if as_of is None else float(as_of)
    by_source: dict[str, SourceStatusEvent] = {}
    for event in events:
        if event.effective_at > cutoff:
            continue
        current = by_source.get(event.source_id)
        if current is None or (event.effective_at, event.event_sha256) > (current.effective_at, current.event_sha256):
            by_source[event.source_id] = event
    output: dict[str, EvidenceStatus] = {}
    for source_id, event in by_source.items():
        blocked = event.status in {"retracted", "withdrawn", "superseded"}
        warning = ""
        if event.status == "retracted":
            warning = "Source has been retracted; derived claims require re-evaluation."
        elif event.status == "withdrawn":
            warning = "Source has been withdrawn; derived claims require re-evaluation."
        elif event.status == "superseded":
            warning = "Source has been superseded; prefer the replacement where applicable."
        elif event.status == "corrected":
            warning = "Source has a correction; inspect the corrected version before relying on affected claims."
        output[source_id] = EvidenceStatus(source_id, event.status, event.event_sha256, event.replacement_source_id, not blocked, warning)
    return output


def annotate_graph_status(graph: EvidenceGraph, statuses: Mapping[str, EvidenceStatus]) -> EvidenceGraph:
    nodes: list[GraphNode] = []
    for node in graph.nodes:
        status = statuses.get(node.source_id)
        if status is None:
            nodes.append(node)
            continue
        attributes = dict(node.attributes)
        attributes.update(
            {
                "source_status": status.status,
                "source_status_event_sha256": status.event_sha256,
                "usable_for_new_claims": "true" if status.usable_for_new_claims else "false",
            }
        )
        if status.replacement_source_id:
            attributes["replacement_source_id"] = status.replacement_source_id
        nodes.append(GraphNode(node.node_id, node.kind, node.source_id, node.content_sha256, node.label, attributes))
    return EvidenceGraph(tuple(nodes), graph.edges)


def filter_graph_for_new_claims(graph: EvidenceGraph, statuses: Mapping[str, EvidenceStatus]) -> EvidenceGraph:
    kept_nodes = tuple(node for node in graph.nodes if statuses.get(node.source_id, EvidenceStatus(node.source_id, "active", "")).usable_for_new_claims)
    kept_ids = {node.node_id for node in kept_nodes}
    kept_edges = tuple(edge for edge in graph.edges if edge.source_node_id in kept_ids and edge.target_node_id in kept_ids)
    return EvidenceGraph(kept_nodes, kept_edges)


def annotate_report_status(report: ResearchReport, statuses: Mapping[str, EvidenceStatus]) -> ResearchReport:
    warnings = list(report.warnings)
    affected_citations: set[str] = set()
    for citation in report.citations:
        source_id = citation.source_id or citation.url
        status = statuses.get(source_id)
        if status is None:
            continue
        citation_key = hashlib.sha256(
            _canonical(
                {
                    "source": source_id,
                    "doc": citation.doc_id or "",
                    "chunk": citation.chunk_id or "",
                    "page": citation.page_number,
                    "quote": citation.quote or citation.snippet or "",
                }
            )
        ).hexdigest()
        if not status.usable_for_new_claims:
            affected_citations.add(citation_key)
        if status.warning:
            warnings.append(f"{source_id}: {status.warning}")
    matrix: list[EvidenceMatrixRow] = []
    for row in report.evidence_matrix:
        if any(citation_id in affected_citations for citation_id in row.citation_ids):
            matrix.append(replace(row, support_status="unreviewed", limitation=(row.limitation + "; " if row.limitation else "") + "Supporting source status changed; re-review required."))
        else:
            matrix.append(row)
    return replace(report, evidence_matrix=tuple(matrix), warnings=tuple(dict.fromkeys(warnings)))


__all__ = [
    "EvidenceStatus",
    "SourceStatusEvent",
    "annotate_graph_status",
    "annotate_report_status",
    "filter_graph_for_new_claims",
    "latest_source_status",
]
