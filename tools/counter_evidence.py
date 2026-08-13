"""Counter-evidence discovery over evidence graphs and scored evidence candidates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from tools.evidence_graph import EvidenceGraph
from tools.evidence_quality import EvidenceItem


@dataclass(frozen=True)
class CounterEvidenceResult:
    claim_id: str
    graph_node_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    independent_roots: tuple[str, ...]


def find_counter_evidence(
    graph: EvidenceGraph,
    claim_id: str,
    evidence: Sequence[EvidenceItem] = (),
    *,
    max_hops: int = 2,
    limit: int = 10,
) -> CounterEvidenceResult:
    """Collect explicit contradiction links plus claim-tagged counter evidence."""

    if max_hops <= 0 or limit <= 0:
        raise ValueError("max_hops and limit must be positive.")
    graph_nodes = sorted(graph.contradictions(claim_id, max_hops=max_hops))[:limit]
    candidates = [item for item in evidence if claim_id in item.contradicts_claims]
    candidates.sort(
        key=lambda item: (-item.score, item.root_source_id, item.evidence_id)
    )
    selected = []
    roots = set()
    for item in candidates:
        if len(selected) >= limit:
            break
        selected.append(item.evidence_id)
        roots.add(item.root_source_id)
    return CounterEvidenceResult(
        claim_id=claim_id,
        graph_node_ids=tuple(graph_nodes),
        evidence_ids=tuple(selected),
        independent_roots=tuple(sorted(roots)),
    )


def counter_evidence_coverage(
    claim_ids: Iterable[str],
    results: Sequence[CounterEvidenceResult],
) -> float:
    claims = set(claim_ids)
    if not claims:
        return 1.0
    covered = {
        result.claim_id
        for result in results
        if result.claim_id in claims and (result.graph_node_ids or result.evidence_ids)
    }
    return len(covered) / len(claims)
