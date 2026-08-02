"""Durable signed actor-use graph-set publication operator CLI."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_relation_actor_use_runtime import get_signed_actor_use_store
from tools.evidence_graph_relation_authorization_runtime import (
    get_relation_review_authorization_store,
)
from tools.evidence_graph_relation_runtime import get_relation_review_ledger
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_publish_attempts import EvidenceGraphSetPublicationAttempt
from tools.evidence_graph_set_publish_reconcile import (
    EvidenceGraphSetPublicationRecoveryError,
)
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_set_signed_actor_provenance_boundary import (
    execute_next_signed_actor_publication_attempt,
    execute_signed_actor_publication_attempt,
    signed_actor_publication_ledger,
)
from tools.evidence_graph_set_signed_publication_runtime import (
    get_evidence_graph_set_signed_publication_journal,
)
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _attempt_summary(value: Any) -> dict[str, Any]:
    return {
        "operation_id": value.operation_id,
        "owner_id": value.owner_id,
        "graph_set_key": value.graph_set_key,
        "proposal_ids": list(value.proposal_ids),
        "expected_current_set_id": value.expected_current_set_id,
        "state": value.state,
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "max_attempts": value.max_attempts,
        "lease_owner": value.lease_owner,
        "lease_expires_at": value.lease_expires_at,
        "previous_graph_set_id": value.previous_graph_set_id,
        "candidate_graph_set_id": value.candidate_graph_set_id,
        "candidate_graph_set_digest": value.candidate_graph_set_digest,
        "member_count": value.member_count,
        "edge_count": value.edge_count,
        "verification_digest": value.verification_digest,
        "failure_type": value.failure_type,
        "compensation_errors": list(value.compensation_errors),
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "completed_at": value.completed_at,
        "authoritative_mutation_performed": False,
        "semantic_inference_performed": False,
        "automatic_approval_performed": False,
        "committed_review_authorizations_required": True,
        "signed_actor_use_provenance_required_when_present": True,
        "signed_publication_journal_isolated": True,
        "source_text_returned": False,
    }


def _execution_summary(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "semantic_inference_performed": False,
            "automatic_approval_performed": False,
            "reviewed_proposals_required": True,
            "committed_review_authorizations_required": True,
            "signed_actor_use_provenance_validated": True,
            "signed_publication_journal_isolated": True,
            "source_text_returned": False,
        }
    )
    return payload


def _dependencies() -> dict[str, Any]:
    return {
        "journal": get_evidence_graph_set_signed_publication_journal(),
        "ledger": get_relation_review_ledger(),
        "authorization_store": get_relation_review_authorization_store(),
        "actor_use_store": get_signed_actor_use_store(),
        "set_store": get_evidence_graph_set_store(),
        "generations": get_generation_store(),
        "graphs": get_evidence_graph_store(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_publication_cli",
        description=(
            "Plan, execute and recover reviewed graph-set publication through an "
            "isolated signed-provenance phase journal while validating committed "
            "authorization receipts and signed actor-use provenance."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed")
    seed.add_argument("--owner-id", required=True)
    seed.add_argument("--graph-set-key", required=True)
    seed.add_argument("--proposal-id", action="append", required=True)
    seed.add_argument("--max-attempts", type=int, default=3)
    expectation = seed.add_mutually_exclusive_group(required=True)
    expectation.add_argument("--expect-no-current", action="store_true")
    expectation.add_argument("--expected-current-set-id")

    status = commands.add_parser("status")
    status.add_argument("operation_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--graph-set-key")
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)

    execute = commands.add_parser("execute")
    execute.add_argument("operation_id")
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=60)

    reconcile = commands.add_parser("reconcile-one")
    reconcile.add_argument("--owner-id", required=True)
    reconcile.add_argument("--worker-id", required=True)
    reconcile.add_argument("--lease-seconds", type=int, default=60)

    for name in ("retry", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("operation_id")
        command.add_argument("--owner-id", required=True)
        command.add_argument("--confirm-operation-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        journal = get_evidence_graph_set_signed_publication_journal()
        if args.command == "seed":
            ledger = get_relation_review_ledger()
            authorization_store = get_relation_review_authorization_store()
            actor_use_store = get_signed_actor_use_store()
            signed_actor_publication_ledger(
                owner_id=args.owner_id,
                graph_set_key=args.graph_set_key,
                proposal_ids=args.proposal_id,
                ledger=ledger,
                authorization_store=authorization_store,
                actor_use_store=actor_use_store,
            )
            expected = None if args.expect_no_current else args.expected_current_set_id
            attempt = EvidenceGraphSetPublicationAttempt.create(
                owner_id=args.owner_id,
                graph_set_key=args.graph_set_key,
                proposal_ids=args.proposal_id,
                expected_current_set_id=expected,
                max_attempts=args.max_attempts,
            )
            _print(_attempt_summary(journal.seed(attempt)))
            return 0
        if args.command == "status":
            _print(_attempt_summary(journal.get(args.operation_id)))
            return 0
        if args.command == "list":
            values = journal.list(
                owner_id=args.owner_id,
                graph_set_key=args.graph_set_key,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "attempts": [_attempt_summary(value) for value in values],
                    "mutation_performed": False,
                    "signed_publication_journal_isolated": True,
                    "source_text_returned": False,
                }
            )
            return 0
        if args.command == "execute":
            result = execute_signed_actor_publication_attempt(
                args.operation_id,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                **_dependencies(),
            )
            _print(_execution_summary(result))
            return 0 if result.state == "completed" else 1
        if args.command == "reconcile-one":
            result = execute_next_signed_actor_publication_attempt(
                owner_id=args.owner_id,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                **_dependencies(),
            )
            if result is None:
                _print(
                    {
                        "status": "idle",
                        "mutation_performed": False,
                        "signed_publication_journal_isolated": True,
                        "source_text_returned": False,
                    }
                )
                return 0
            _print(_execution_summary(result))
            return 0 if result.state == "completed" else 1
        if args.command == "retry":
            value = journal.retry(
                args.operation_id,
                owner_id=args.owner_id,
                confirm_operation_id=args.confirm_operation_id,
            )
            _print(_attempt_summary(value))
            return 0
        if args.command == "cancel":
            value = journal.cancel(
                args.operation_id,
                owner_id=args.owner_id,
                confirm_operation_id=args.confirm_operation_id,
            )
            _print(_attempt_summary(value))
            return 0
        raise ValueError("unsupported signed publication-journal command.")
    except EvidenceGraphSetPublicationRecoveryError as exc:
        _print(
            {
                "error": "publication_failed",
                "operation_id": exc.operation_id,
                "state": exc.state,
                "phase": exc.phase,
                "compensation_complete": not bool(exc.compensation_errors),
                "compensation_errors": list(exc.compensation_errors),
                "signed_actor_use_provenance_validated": False,
                "signed_publication_journal_isolated": True,
                "source_text_returned": False,
            },
            stream=sys.stderr,
        )
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
