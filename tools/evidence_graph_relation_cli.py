"""Operator CLI for governed text-free relation proposals and review."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import require_relation_review_actor
from tools.evidence_graph_relation_authorization_runtime import (
    get_relation_review_authorization_store,
)
from tools.evidence_graph_relation_policy import GovernedRelationReviewService
from tools.evidence_graph_relation_policy_runtime import get_relation_review_policy
from tools.evidence_graph_relation_review import (
    CrossDocumentRelationProposal,
    RelationEndpoint,
    RelationReviewDecision,
)
from tools.evidence_graph_relation_runtime import get_relation_review_ledger


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _authorization_summary(record: Any | None) -> dict[str, Any] | None:
    if record is None:
        return None
    authorization = record.authorization
    return {
        "state": record.state,
        "decision_id": authorization.decision_id,
        "reviewer_id": authorization.reviewer_id,
        "policy_digest": authorization.policy_digest,
        "grant_digest": authorization.grant_digest,
        "authorization_digest": authorization.authorization_digest,
        "authorized_at": authorization.authorized_at,
        "prepared_at": record.prepared_at,
        "committed_at": record.committed_at,
        "separation_of_duties_enforced": (
            authorization.separation_of_duties_enforced
        ),
        "replacement_scope_validated": (
            authorization.replacement_scope_validated
        ),
    }


def _proposal_summary(
    value: Any,
    decision: Any | None,
    authorization_record: Any | None = None,
) -> dict[str, Any]:
    authorization = _authorization_summary(authorization_record)
    return {
        "proposal_id": value.proposal_id,
        "proposal_digest": value.proposal_digest,
        "owner_id": value.owner_id,
        "graph_set_key": value.graph_set_key,
        "relation_key": value.relation_key,
        "source": {
            "doc_id": value.source.doc_id,
            "generation": value.source.generation,
            "graph_digest": value.source.graph_digest,
            "node_id": value.source.node_id,
            "provenance_digest": value.source.provenance_digest,
        },
        "target": {
            "doc_id": value.target.doc_id,
            "generation": value.target.generation,
            "graph_digest": value.target.graph_digest,
            "node_id": value.target.node_id,
            "provenance_digest": value.target.provenance_digest,
        },
        "edge_type": value.edge_type,
        "proposer_kind": value.proposer_kind,
        "proposer_id": value.proposer_id,
        "evidence_digest": value.evidence_digest,
        "extractor_name": value.extractor_name,
        "extractor_version": value.extractor_version,
        "weight": value.weight,
        "created_at": value.created_at,
        "review": (
            None
            if decision is None
            else {
                "decision_id": decision.decision_id,
                "decision": decision.decision,
                "reviewer_id": decision.reviewer_id,
                "reason_code": decision.reason_code,
                "replacement_proposal_id": decision.replacement_proposal_id,
                "decided_at": decision.decided_at,
            }
        ),
        "review_authorization": authorization,
        "governed_review": decision is not None and authorization is not None,
        "contains_source_text": False,
        "automatic_approval_performed": False,
    }


def _endpoint(prefix: str, args: argparse.Namespace) -> RelationEndpoint:
    return RelationEndpoint(
        doc_id=getattr(args, f"{prefix}_doc_id"),
        generation=getattr(args, f"{prefix}_generation"),
        graph_digest=getattr(args, f"{prefix}_graph_digest"),
        node_id=getattr(args, f"{prefix}_node_id"),
        provenance_digest=getattr(args, f"{prefix}_provenance_digest"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_relation_cli",
        description=(
            "Submit text-free cross-document relation proposals and record explicit "
            "terminal reviewer decisions under configured authorization policy. "
            "No proposal is approved automatically."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    propose = commands.add_parser("propose")
    propose.add_argument("--owner-id", required=True)
    propose.add_argument("--graph-set-key", required=True)
    propose.add_argument("--relation-key", required=True)
    for prefix in ("source", "target"):
        propose.add_argument(f"--{prefix}-doc-id", required=True)
        propose.add_argument(f"--{prefix}-generation", required=True, type=int)
        propose.add_argument(f"--{prefix}-graph-digest", required=True)
        propose.add_argument(f"--{prefix}-node-id", required=True)
        propose.add_argument(f"--{prefix}-provenance-digest", required=True)
    propose.add_argument("--edge-type", required=True)
    propose.add_argument(
        "--proposer-kind",
        choices=("human", "model", "rule"),
        required=True,
    )
    propose.add_argument("--proposer-id", required=True)
    propose.add_argument("--evidence-digest", required=True)
    propose.add_argument("--extractor-name")
    propose.add_argument("--extractor-version")
    propose.add_argument("--weight", type=float, default=1.0)

    decide = commands.add_parser("decide")
    decide.add_argument("proposal_id")
    decide.add_argument("--owner-id", required=True)
    decide.add_argument(
        "--decision",
        choices=("approved", "rejected", "superseded"),
        required=True,
    )
    decide.add_argument(
        "--reviewer-id",
        required=True,
        help=(
            "Must exactly match the process-owned actor configured through "
            "EVIDENCE_GRAPH_REVIEW_ACTOR_ID or EVIDENCE_GRAPH_REVIEW_ACTOR_ID_PATH."
        ),
    )
    decide.add_argument("--reason-code", required=True)
    decide.add_argument("--replacement-proposal-id")

    status = commands.add_parser("status")
    status.add_argument("proposal_id")
    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--graph-set-key", required=True)
    listing.add_argument(
        "--decision",
        choices=("pending", "approved", "rejected", "superseded"),
    )
    listing.add_argument("--limit", type=int, default=100)
    return parser


def _propose(args: argparse.Namespace) -> int:
    value = CrossDocumentRelationProposal.create(
        owner_id=args.owner_id,
        graph_set_key=args.graph_set_key,
        relation_key=args.relation_key,
        source=_endpoint("source", args),
        target=_endpoint("target", args),
        edge_type=args.edge_type,
        proposer_kind=args.proposer_kind,
        proposer_id=args.proposer_id,
        evidence_digest=args.evidence_digest,
        extractor_name=args.extractor_name,
        extractor_version=args.extractor_version,
        weight=args.weight,
    )
    stored = get_relation_review_ledger().submit(value)
    _print(_proposal_summary(stored, None))
    return 0


def _decide(args: argparse.Namespace) -> int:
    actor = require_relation_review_actor(args.reviewer_id)
    value = RelationReviewDecision.create(
        proposal_id=args.proposal_id,
        owner_id=args.owner_id,
        decision=args.decision,
        reviewer_id=actor.actor_id,
        reason_code=args.reason_code,
        replacement_proposal_id=args.replacement_proposal_id,
    )
    ledger = get_relation_review_ledger()
    service = GovernedRelationReviewService(
        ledger=ledger,
        policy=get_relation_review_policy(),
        authorization_store=get_relation_review_authorization_store(),
    )
    stored, receipt = service.decide(value)
    proposal = ledger.get_proposal(stored.proposal_id)
    payload = _proposal_summary(proposal, stored, receipt)
    payload["review_actor_binding"] = {
        "actor_id": actor.actor_id,
        "binding_method": actor.binding_method,
        "binding_digest": actor.binding_digest,
        "loaded_at": actor.loaded_at,
        "durable_receipt_field": False,
    }
    _print(payload)
    return 0


def _status(args: argparse.Namespace) -> int:
    ledger = get_relation_review_ledger()
    proposal = ledger.get_proposal(args.proposal_id)
    decision = ledger.get_decision(proposal.proposal_id)
    receipt = (
        None
        if decision is None
        else get_relation_review_authorization_store().get(decision.decision_id)
    )
    _print(_proposal_summary(proposal, decision, receipt))
    return 0


def _list(args: argparse.Namespace) -> int:
    values = get_relation_review_ledger().list(
        owner_id=args.owner_id,
        graph_set_key=args.graph_set_key,
        decision=args.decision,
        limit=args.limit,
    )
    authorization_store = get_relation_review_authorization_store()
    rendered = []
    for proposal, review in values:
        receipt = (
            None
            if review is None
            else authorization_store.get(review.decision_id)
        )
        rendered.append(_proposal_summary(proposal, review, receipt))
    _print(
        {
            "owner_id": args.owner_id,
            "graph_set_key": args.graph_set_key,
            "count": len(rendered),
            "proposals": rendered,
            "contains_source_text": False,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "propose":
            return _propose(args)
        if args.command == "decide":
            return _decide(args)
        if args.command == "status":
            return _status(args)
        if args.command == "list":
            return _list(args)
        raise ValueError("unsupported relation review command.")
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, PermissionError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
