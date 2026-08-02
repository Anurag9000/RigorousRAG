"""Operator CLI for crash-recoverable signed publication retirement sagas."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.evidence_graph_set_publish_runtime import (
    get_evidence_graph_set_publication_journal,
)
from tools.evidence_graph_set_runtime import get_evidence_graph_set_store
from tools.evidence_graph_set_signed_publication_runtime import (
    get_evidence_graph_set_signed_publication_journal,
)
from tools.evidence_graph_set_signed_retirement_boundary import (
    SignedPublicationRetirementRecoveryError,
    execute_next_signed_publication_retirement,
    execute_signed_publication_retirement,
    seed_signed_publication_retirement,
)
from tools.evidence_graph_set_signed_retirement_runtime import (
    get_signed_publication_retirement_journal,
)
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _attempt_summary(value: Any) -> dict[str, Any]:
    return {
        "retirement_id": value.retirement_id,
        "owner_id": value.owner_id,
        "publication_operation_id": value.publication_operation_id,
        "graph_set_key": value.graph_set_key,
        "signed_candidate_set_id": value.signed_candidate_set_id,
        "signed_candidate_set_digest": value.signed_candidate_set_digest,
        "authorization_candidate_set_id": value.authorization_candidate_set_id,
        "signed_authority_digest": value.signed_authority_digest,
        "state": value.state,
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "max_attempts": value.max_attempts,
        "lease_owner": value.lease_owner,
        "lease_expires_at": value.lease_expires_at,
        "final_pointer_set_id": value.final_pointer_set_id,
        "verification_digest": value.verification_digest,
        "failure_type": value.failure_type,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "completed_at": value.completed_at,
        "publication_mutation_performed": False,
        "retirement_mutation_performed": False,
        "source_text_returned": False,
    }


def _execution_summary(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "weaker_pointer_restoration_performed": False,
            "automatic_migration_performed": False,
            "source_text_returned": False,
        }
    )
    return payload


def _dependencies() -> dict[str, Any]:
    return {
        "retirement_journal": get_signed_publication_retirement_journal(),
        "authorization_journal": get_evidence_graph_set_publication_journal(),
        "signed_journal": get_evidence_graph_set_signed_publication_journal(),
        "set_store": get_evidence_graph_set_store(),
        "generations": get_generation_store(),
        "graphs": get_evidence_graph_store(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_set_signed_retirement_saga_cli",
        description=(
            "Plan, execute and recover exact retirement of expired authorization-only "
            "publication duplicates after a completed signed publication."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed")
    seed.add_argument("publication_operation_id")
    seed.add_argument("--owner-id", required=True)
    seed.add_argument("--confirm-operation-id", required=True)
    seed.add_argument("--max-attempts", type=int, default=3)

    status = commands.add_parser("status")
    status.add_argument("retirement_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--publication-operation-id")
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)

    execute = commands.add_parser("execute")
    execute.add_argument("retirement_id")
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=60)

    reconcile = commands.add_parser("reconcile-one")
    reconcile.add_argument("--owner-id", required=True)
    reconcile.add_argument("--worker-id", required=True)
    reconcile.add_argument("--lease-seconds", type=int, default=60)

    for name in ("retry", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("retirement_id")
        command.add_argument("--owner-id", required=True)
        command.add_argument("--confirm-retirement-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        dependencies = _dependencies()
        retirement_journal = dependencies["retirement_journal"]
        if args.command == "seed":
            if args.publication_operation_id != args.confirm_operation_id:
                raise ValueError("publication operation confirmation differs.")
            attempt, preflight = seed_signed_publication_retirement(
                owner_id=args.owner_id,
                publication_operation_id=args.publication_operation_id,
                max_attempts=args.max_attempts,
                **dependencies,
            )
            payload = _attempt_summary(attempt)
            payload.update(
                {
                    "preflight_report_digest": preflight.report_digest,
                    "preflight_disposition": preflight.disposition,
                    "preflight_eligible": preflight.eligible,
                    "retirement_journal_mutation_performed": True,
                    "publication_mutation_performed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "status":
            _print(_attempt_summary(retirement_journal.get(args.retirement_id)))
            return 0
        if args.command == "list":
            values = retirement_journal.list(
                owner_id=args.owner_id,
                publication_operation_id=args.publication_operation_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "retirements": [_attempt_summary(value) for value in values],
                    "mutation_performed": False,
                    "source_text_returned": False,
                }
            )
            return 0
        if args.command == "execute":
            result = execute_signed_publication_retirement(
                args.retirement_id,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                **dependencies,
            )
            _print(_execution_summary(result))
            return 0 if result.state == "completed" else 1
        if args.command == "reconcile-one":
            result = execute_next_signed_publication_retirement(
                owner_id=args.owner_id,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                **dependencies,
            )
            if result is None:
                _print(
                    {
                        "status": "idle",
                        "mutation_performed": False,
                        "source_text_returned": False,
                    }
                )
                return 0
            _print(_execution_summary(result))
            return 0 if result.state == "completed" else 1
        if args.command == "retry":
            value = retirement_journal.retry(
                args.retirement_id,
                owner_id=args.owner_id,
                confirm_retirement_id=args.confirm_retirement_id,
            )
            _print(_attempt_summary(value))
            return 0
        if args.command == "cancel":
            value = retirement_journal.cancel(
                args.retirement_id,
                owner_id=args.owner_id,
                confirm_retirement_id=args.confirm_retirement_id,
            )
            _print(_attempt_summary(value))
            return 0
        raise ValueError("unsupported signed retirement command.")
    except SignedPublicationRetirementRecoveryError as exc:
        _print(
            {
                "error": "retirement_failed",
                "retirement_id": exc.retirement_id,
                "state": exc.state,
                "phase": exc.phase,
                "weaker_pointer_restoration_performed": False,
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
