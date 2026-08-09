"""Preparation-only operator CLI for future migration cutover operations."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.migration_cutover_runtime import (
    get_migration_cutover_journal,
    prepare_cutover_operation,
)
from tools.migration_types import exact_integer, identifier
from tools.security import normalize_owner_id


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _summary(operation: Any) -> dict[str, Any]:
    preparation = operation.preparation
    return {
        "operation_id": operation.operation_id,
        "task_id": preparation.task_id,
        "owner_id": preparation.owner_id,
        "doc_id": preparation.doc_id,
        "state": operation.state,
        "attempt": operation.attempt,
        "fencing_token": operation.fencing_token,
        "source_sequence": preparation.source_sequence,
        "source_profile_fingerprint": preparation.source_profile_fingerprint,
        "target_profile_fingerprint": preparation.target_profile_fingerprint,
        "source_content_sha256": preparation.source_content_sha256,
        "validation_digest": preparation.validation_digest,
        "promotion_report_digest": preparation.promotion_report_digest,
        "benchmark_fingerprint": preparation.benchmark_fingerprint,
        "preflight_digest": preparation.preflight_digest,
        "rollback_identity_digest": preparation.rollback_identity_digest,
        "rollback_artifact_digest": preparation.rollback_artifact_digest,
        "rollback_key_id": preparation.rollback_key_id,
        "staging_verification_digest": preparation.staging_verification_digest,
        "target_artifact_digest": preparation.target_artifact_digest,
        "vector_snapshot_digest": preparation.vector_snapshot_digest,
        "sparse_snapshot_digest": preparation.sparse_snapshot_digest,
        "failure_type": operation.failure_type,
        "created_at": operation.created_at,
        "updated_at": operation.updated_at,
        "lease_owner": operation.lease_owner,
        "lease_expires_at": operation.lease_expires_at,
        "authoritative_mutation_performed": False,
        "restore_performed": False,
        "cutover_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_cutover_control_cli",
        description=(
            "Create and inspect leased cutover-preparation records. The state "
            "machine intentionally has no execute, committed, restore or rollback action."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare",
        help="Verify all immutable prerequisites and mark one operation ready.",
    )
    prepare.add_argument("task_id")
    prepare.add_argument("--worker-id", required=True)
    prepare.add_argument("--lease-seconds", type=int, default=300)
    prepare.add_argument("--max-attempts", type=int, default=3)
    status = commands.add_parser("status", help="Read one cutover preparation operation.")
    status.add_argument("operation_id")
    listing = commands.add_parser("list", help="List bounded owner-scoped operations.")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)
    cancel = commands.add_parser(
        "cancel", help="Cancel only planned or failed preparation."
    )
    cancel.add_argument("operation_id")
    cancel.add_argument("--confirm-operation-id", required=True)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    operation = prepare_cutover_operation(
        identifier(args.task_id, "task_id", 64),
        worker_id=identifier(args.worker_id, "worker_id", 128),
        lease_seconds=exact_integer(args.lease_seconds, "lease_seconds", 1, 86_400),
        max_attempts=exact_integer(args.max_attempts, "max_attempts", 1, 100),
    )
    _print(_summary(operation))
    return 0 if operation.state == "ready" else 1


def _status(args: argparse.Namespace) -> int:
    operation_id = identifier(args.operation_id, "operation_id", 64)
    operation = get_migration_cutover_journal().get(operation_id)
    if operation is None:
        _print({"error": "not_found", "operation_id": operation_id}, stream=sys.stderr)
        return 1
    _print(_summary(operation))
    return 0


def _list(args: argparse.Namespace) -> int:
    operations = get_migration_cutover_journal().list_operations(
        owner_id=normalize_owner_id(args.owner_id),
        state=args.state,
        limit=exact_integer(args.limit, "limit", 1, 10_000),
    )
    _print(
        {
            "count": len(operations),
            "operations": [_summary(operation) for operation in operations],
        }
    )
    return 0


def _cancel(args: argparse.Namespace) -> int:
    operation_id = identifier(args.operation_id, "operation_id", 64)
    confirmation = identifier(
        args.confirm_operation_id,
        "confirm_operation_id",
        64,
    )
    if confirmation != operation_id:
        raise ValueError("confirmation must exactly match operation_id.")
    operation = get_migration_cutover_journal().cancel(operation_id)
    _print(_summary(operation))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "status":
            return _status(args)
        if args.command == "list":
            return _list(args)
        if args.command == "cancel":
            return _cancel(args)
        raise ValueError("unsupported cutover control command.")
    except FileNotFoundError as exc:
        selected = str(exc.args[0]) if exc.args else "unknown"
        _print({"error": "not_found", "task_id": selected}, stream=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())