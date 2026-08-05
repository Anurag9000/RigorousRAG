"""Canonical topological submission boundary for scientific claim proposals."""

from __future__ import annotations

from collections.abc import Iterable

from tools.evidence_graph_claim_contracts import ScientificClaimProposal
from tools.evidence_graph_claim_store import ScientificClaimReviewStore

_MAX_BATCH = 10_000


def order_scientific_claim_proposals(
    proposals: Iterable[ScientificClaimProposal],
) -> tuple[ScientificClaimProposal, ...]:
    """Order same-batch correction chains predecessor-first and reject cycles."""

    if isinstance(proposals, (str, bytes, bytearray)):
        raise ValueError("proposals must be an iterable.")
    values = tuple(proposals)
    if not 1 <= len(values) <= _MAX_BATCH:
        raise ValueError("proposals must contain a bounded non-empty batch.")
    if any(not isinstance(value, ScientificClaimProposal) for value in values):
        raise ValueError("every proposal must be ScientificClaimProposal.")
    by_id = {value.proposal_id: value for value in values}
    if len(by_id) != len(values):
        raise ValueError("claim proposal batch contains duplicate IDs.")

    state: dict[str, int] = {}
    ordered: list[ScientificClaimProposal] = []

    def visit(proposal: ScientificClaimProposal) -> None:
        marker = state.get(proposal.proposal_id, 0)
        if marker == 2:
            return
        if marker == 1:
            raise ValueError("claim correction batch contains a lineage cycle.")
        state[proposal.proposal_id] = 1
        predecessor_id = proposal.supersedes_proposal_id
        if predecessor_id is not None and predecessor_id in by_id:
            predecessor = by_id[predecessor_id]
            if (
                predecessor.owner_id != proposal.owner_id
                or predecessor.doc_id != proposal.doc_id
                or predecessor.generation != proposal.generation
                or predecessor.content_sha256 != proposal.content_sha256
                or predecessor.profile_fingerprint != proposal.profile_fingerprint
            ):
                raise PermissionError(
                    "claim correction must remain in the same document generation scope."
                )
            visit(predecessor)
        state[proposal.proposal_id] = 2
        ordered.append(proposal)

    for proposal in sorted(values, key=lambda value: value.proposal_id):
        visit(proposal)
    return tuple(ordered)


def submit_scientific_claim_proposals(
    store: ScientificClaimReviewStore,
    proposals: Iterable[ScientificClaimProposal],
) -> tuple[ScientificClaimProposal, ...]:
    """Submit one correction-aware atomic batch through the low-level store."""

    if not isinstance(store, ScientificClaimReviewStore):
        raise ValueError("store must be ScientificClaimReviewStore.")
    original = tuple(proposals)
    ordered = order_scientific_claim_proposals(original)
    stored_by_id = {
        value.proposal_id: value for value in store.submit_many(ordered)
    }
    return tuple(stored_by_id[value.proposal_id] for value in original)


__all__ = [
    "order_scientific_claim_proposals",
    "submit_scientific_claim_proposals",
]
