"""Privacy-conscious operator CLI for reviewed scientific claim proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_claim_contracts import ClaimReviewDecision
from tools.evidence_graph_claim_review import (
    GovernedScientificClaimReviewService,
    approved_claim_annotations,
)
from tools.evidence_graph_claim_runtime import (
    get_scientific_claim_review_policy,
    get_scientific_claim_review_store,
)
from tools.evidence_graph_relation_actor import load_relation_review_actor


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _proposal_summary(proposal: Any, decision: Any, authorization: Any) -> dict[str, Any]:
    text_bytes = proposal.claim_text.encode("utf-8")
    return {
        "proposal_id": proposal.proposal_id,
        "proposal_digest": proposal.proposal_digest,
        "owner_id": proposal.owner_id,
        "doc_id": proposal.doc_id,
        "generation": proposal.generation,
        "content_sha256": proposal.content_sha256,
        "profile_fingerprint": proposal.profile_fingerprint,
        "claim_key": proposal.claim_key,
        "claim_text_sha256": hashlib.sha256(text_bytes).hexdigest(),
        "claim_text_length": len(proposal.claim_text),
        "claim_type": proposal.claim_type,
        "modality": proposal.modality,
        "section_index": proposal.locator.section_index,
        "page_number": proposal.locator.page_number,
        "char_start": proposal.locator.char_start,
        "char_end": proposal.locator.char_end,
        "evidence_sha256": proposal.locator.evidence_sha256,
        "locator_digest": proposal.locator.locator_digest,
        "proposer_kind": proposal.proposer_kind,
        "proposer_id": proposal.proposer_id,
        "extractor_name": proposal.extractor_name,
        "extractor_version": proposal.extractor_version,
        "confidence": proposal.confidence,
        "supersedes_proposal_id": proposal.supersedes_proposal_id,
        "decision": None if decision is None else decision.decision,
        "decision_id": None if decision is None else decision.decision_id,
        "reviewer_id": None if decision is None else decision.reviewer_id,
        "reason_code": None if decision is None else decision.reason_code,
        "replacement_proposal_id": (
            None if decision is None else decision.replacement_proposal_id
        ),
        "authorization_digest": (
            None if authorization is None else authorization.authorization_digest
        ),
        "policy_digest": None if authorization is None else authorization.policy_digest,
        "grant_digest": None if authorization is None else authorization.grant_digest,
        "source_text_returned": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_claim_cli",
        description=(
            "Inspect and govern scientific claim proposals. Claim/source text is never "
            "printed; extraction submission remains a programmatic validated adapter."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status")
    status.add_argument("proposal_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--doc-id", required=True)
    listing.add_argument("--generation", type=int)
    listing.add_argument(
        "--decision",
        choices=["pending", "approved", "rejected", "superseded"],
    )
    listing.add_argument("--limit", type=int, default=100)

    decide = commands.add_parser("decide")
    decide.add_argument("proposal_id")
    decide.add_argument("--owner-id", required=True)
    decide.add_argument(
        "--decision", required=True, choices=["approved", "rejected", "superseded"]
    )
    decide.add_argument("--reviewer-id", required=True)
    decide.add_argument("--reason-code", required=True)
    decide.add_argument("--replacement-proposal-id")

    annotations = commands.add_parser("annotations")
    annotations.add_argument("--owner-id", required=True)
    annotations.add_argument("--doc-id", required=True)
    annotations.add_argument("--generation", required=True, type=int)
    annotations.add_argument("--content-sha256", required=True)
    annotations.add_argument("--profile-fingerprint", required=True)
    annotations.add_argument("--proposal-id", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        store = get_scientific_claim_review_store()
        if args.command == "status":
            proposal = store.get_proposal(args.proposal_id)
            decision = store.get_decision(proposal.proposal_id)
            authorization = store.get_authorization(proposal.proposal_id)
            _print(
                {
                    **_proposal_summary(proposal, decision, authorization),
                    "mutation_performed": False,
                }
            )
            return 0
        if args.command == "list":
            values = store.list(
                owner_id=args.owner_id,
                doc_id=args.doc_id,
                generation=args.generation,
                decision=args.decision,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "doc_id": args.doc_id,
                    "item_count": len(values),
                    "items": [
                        _proposal_summary(proposal, decision, authorization)
                        for proposal, decision, authorization in values
                    ],
                    "mutation_performed": False,
                    "source_text_returned": False,
                }
            )
            return 0
        if args.command == "decide":
            proposal = store.get_proposal(args.proposal_id)
            if proposal.owner_id != args.owner_id:
                raise PermissionError("claim proposal owner mismatch.")
            decision = ClaimReviewDecision.create(
                proposal_id=proposal.proposal_id,
                owner_id=args.owner_id,
                decision=args.decision,
                reviewer_id=args.reviewer_id,
                reason_code=args.reason_code,
                replacement_proposal_id=args.replacement_proposal_id,
            )
            service = GovernedScientificClaimReviewService(
                store=store,
                policy=get_scientific_claim_review_policy(),
            )
            stored, authorization = service.decide(
                decision,
                actor_binding=load_relation_review_actor(),
            )
            _print(
                {
                    "proposal_id": proposal.proposal_id,
                    "decision_id": stored.decision_id,
                    "decision": stored.decision,
                    "reviewer_id": stored.reviewer_id,
                    "authorization_digest": authorization.authorization_digest,
                    "policy_digest": authorization.policy_digest,
                    "grant_digest": authorization.grant_digest,
                    "atomic_decision_authorization_commit": True,
                    "mutation_performed": True,
                    "source_text_returned": False,
                }
            )
            return 0
        if args.command == "annotations":
            values = approved_claim_annotations(
                owner_id=args.owner_id,
                doc_id=args.doc_id,
                generation=args.generation,
                content_sha256=args.content_sha256,
                profile_fingerprint=args.profile_fingerprint,
                proposal_ids=args.proposal_id,
                store=store,
            )
            _print(
                {
                    "annotation_count": len(values),
                    "annotations": [
                        {
                            "annotation_key": value.annotation_key,
                            "node_type": value.node_type,
                            "label_sha256": hashlib.sha256(
                                value.label.encode("utf-8")
                            ).hexdigest(),
                            "text_sha256": hashlib.sha256(
                                value.text.encode("utf-8")
                            ).hexdigest(),
                            "section_index": value.section_index,
                            "page_number": value.page_number,
                            "metadata": dict(value.metadata),
                        }
                        for value in values
                    ],
                    "mutation_performed": False,
                    "graph_mutation_performed": False,
                    "semantic_relation_inference_performed": False,
                    "source_text_returned": False,
                }
            )
            return 0
        raise ValueError("unsupported scientific claim command.")
    except (KeyError, OSError, PermissionError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
