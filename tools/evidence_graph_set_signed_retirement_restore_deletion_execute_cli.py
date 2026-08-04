"""Operator CLI for crash-recoverable restore-intent deletion execution."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_runtime import (
    get_signed_retirement_restore_custody_store,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_execute_runtime import (
    get_signed_retirement_restore_deletion_journal,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_reconcile import (
    SignedRetirementRestoreDeletionRecoveryError,
    execute_next_signed_retirement_restore_deletion,
    execute_signed_retirement_restore_deletion,
    seed_signed_retirement_restore_deletion,
)
from tools.evidence_graph_set_signed_retirement_restore_deletion_runtime import (
    get_signed_retirement_restore_deletion_authorization_store,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_runtime import (
    get_signed_retirement_restore_hold_store,
)
from tools.evidence_graph_set_signed_retirement_restore_runtime import (
    get_signed_retirement_restore_journal,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _attempt(value: Any) -> dict[str, Any]:
    return {
        "deletion_id": value.deletion_id,
        "authorization_id": value.authorization_id,
        "authorization_digest": value.authorization_digest,
        "owner_id": value.owner_id,
        "restore_id": value.restore_id,
        "snapshot_digest": value.snapshot_digest,
        "target_path_digest": value.target_path_digest,
        "restore_state": value.restore_state,
        "restore_phase": value.restore_phase,
        "restore_record_digest": value.restore_record_digest,
        "custody_id": value.custody_id,
        "custody_manifest_digest": value.custody_manifest_digest,
        "state": value.state,
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "max_attempts": value.max_attempts,
        "lease_owner_present": value.lease_owner is not None,
        "lease_expires_at": value.lease_expires_at,
        "marker_digest": value.marker_digest,
        "tombstone_digest": value.tombstone_digest,
        "failure_type": value.failure_type,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "completed_at": value.completed_at,
        "contains_source_text": False,
        "raw_paths_returned": False,
        "custody_deleted": False,
        "holds_deleted": False,
    }


def _dependencies() -> dict[str, Any]:
    return {
        "deletion_journal": (
            get_signed_retirement_restore_deletion_journal()
        ),
        "authorization_store": (
            get_signed_retirement_restore_deletion_authorization_store()
        ),
        "restore_journal": get_signed_retirement_restore_journal(),
        "hold_store": get_signed_retirement_restore_hold_store(),
        "custody_store": get_signed_retirement_restore_custody_store(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_deletion_execute_cli"
        ),
        description=(
            "Seed, execute and recover deletion of one authorized terminal "
            "restore-intent row. Custody, holds, receipts and artifacts are retained."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed")
    seed.add_argument("authorization_id")
    seed.add_argument("--restore-id", required=True)
    seed.add_argument("--confirm-authorization-id", required=True)
    seed.add_argument("--confirm-restore-id", required=True)
    seed.add_argument("--max-attempts", type=int, default=3)

    status = commands.add_parser("status")
    status.add_argument("deletion_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)

    execute = commands.add_parser("execute")
    execute.add_argument("deletion_id")
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=60)

    reconcile = commands.add_parser("reconcile-one")
    reconcile.add_argument("--owner-id", required=True)
    reconcile.add_argument("--worker-id", required=True)
    reconcile.add_argument("--lease-seconds", type=int, default=60)

    retry = commands.add_parser("retry")
    retry.add_argument("deletion_id")
    retry.add_argument("--owner-id", required=True)
    retry.add_argument("--confirm-deletion-id", required=True)

    cancel = commands.add_parser("cancel")
    cancel.add_argument("deletion_id")
    cancel.add_argument("--owner-id", required=True)
    cancel.add_argument("--confirm-deletion-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "seed":
            if args.authorization_id != args.confirm_authorization_id:
                raise ValueError("authorization confirmation differs.")
            if args.restore_id != args.confirm_restore_id:
                raise ValueError("restore confirmation differs.")
        if args.command in {"retry", "cancel"} and (
            args.deletion_id != args.confirm_deletion_id
        ):
            raise ValueError("deletion confirmation differs.")

        if args.command in {"status", "list", "retry", "cancel"}:
            journal = get_signed_retirement_restore_deletion_journal()
            if args.command == "status":
                payload = _attempt(journal.get(args.deletion_id))
                payload["mutation_performed"] = False
                _print(payload)
                return 0
            if args.command == "list":
                values = journal.list(
                    owner_id=args.owner_id,
                    state=args.state,
                    limit=args.limit,
                )
                _print(
                    {
                        "count": len(values),
                        "deletions": [_attempt(value) for value in values],
                        "mutation_performed": False,
                        "contains_source_text": False,
                        "raw_paths_returned": False,
                    }
                )
                return 0
            mutation = (
                journal.retry
                if args.command == "retry"
                else journal.cancel
            )
            value = mutation(
                args.deletion_id,
                owner_id=args.owner_id,
                confirm_deletion_id=args.confirm_deletion_id,
            )
            payload = _attempt(value)
            payload["deletion_journal_mutation_performed"] = True
            payload["restore_row_deleted"] = False
            _print(payload)
            return 0

        dependencies = _dependencies()
        if args.command == "seed":
            authorization = dependencies["authorization_store"].get(
                args.authorization_id
            )
            if authorization.restore_id != args.restore_id:
                raise RuntimeError(
                    "authorization restore differs from confirmation."
                )
            value, report = seed_signed_retirement_restore_deletion(
                authorization_id=args.authorization_id,
                max_attempts=args.max_attempts,
                **dependencies,
            )
            payload = _attempt(value)
            payload.update(
                {
                    "authorization_preflight_disposition": report.disposition,
                    "authorization_preflight_digest": report.report_digest,
                    "deletion_journal_mutation_performed": True,
                    "restore_row_deleted": False,
                    "authorization_consumed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "execute":
            result = execute_signed_retirement_restore_deletion(
                args.deletion_id,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                **dependencies,
            )
            _print(asdict(result))
            return 0
        if args.command == "reconcile-one":
            result = execute_next_signed_retirement_restore_deletion(
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
                        "raw_paths_returned": False,
                    }
                )
            else:
                _print(asdict(result))
            return 0
        raise ValueError("unsupported restore deletion command.")
    except SignedRetirementRestoreDeletionRecoveryError as exc:
        _print(
            {
                "error": "deletion_failed",
                "deletion_id": exc.deletion_id,
                "state": exc.state,
                "phase": exc.phase,
                "custody_deleted": False,
                "holds_deleted": False,
                "source_text_returned": False,
                "raw_paths_returned": False,
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
