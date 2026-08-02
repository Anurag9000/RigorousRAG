"""Operator CLI for durable pre-restore custody artifact publication."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_boundary import (
    RestoreCustodyArtifactRecoveryError,
    execute_restore_custody_artifact_attempt,
    seed_restore_custody_artifact_attempt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_runtime import (
    get_restore_custody_artifact_journal,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _attempt_summary(value: Any) -> dict[str, Any]:
    return {
        "artifact_id": value.artifact_id,
        "owner_id": value.owner_id,
        "snapshot_digest": value.snapshot_digest,
        "target_path_digest": value.target_path_digest,
        "backup_path_digest": value.backup_path_digest,
        "receipt_path_digest": value.receipt_path_digest,
        "state": value.state,
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "max_attempts": value.max_attempts,
        "lease_owner_present": value.lease_owner is not None,
        "lease_expires_at": value.lease_expires_at,
        "backup_sha256": value.backup_sha256,
        "backup_size_bytes": value.backup_size_bytes,
        "receipt_digest": value.receipt_digest,
        "receipt_binding_method": value.receipt_binding_method,
        "receipt_binding_digest": value.receipt_binding_digest,
        "disposition": value.disposition,
        "failure_type": value.failure_type,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "completed_at": value.completed_at,
        "schema_version": value.schema_version,
        "contains_source_text": False,
        "contains_assertion_secrets": False,
        "raw_paths_returned": False,
    }


def _execution_summary(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "contains_assertion_secrets": False,
            "raw_paths_returned": False,
        }
    )
    return payload


def _protected(args: Any) -> tuple[str, ...]:
    return (
        args.target_db_path,
        args.backup_output,
        args.receipt_output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_artifact_cli"
        ),
        description=(
            "Durably publish or recover one pre-restore backup/receipt pair. "
            "No command overwrites or deletes artifacts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed")
    seed.add_argument("--snapshot", required=True)
    seed.add_argument("--target-db-path", required=True)
    seed.add_argument("--backup-output", required=True)
    seed.add_argument("--receipt-output", required=True)
    seed.add_argument("--confirm-snapshot-digest", required=True)
    seed.add_argument("--max-attempts", type=int, default=3)

    execute = commands.add_parser("execute")
    execute.add_argument("artifact_id")
    execute.add_argument("--confirm-artifact-id", required=True)
    execute.add_argument("--snapshot", required=True)
    execute.add_argument("--target-db-path", required=True)
    execute.add_argument("--backup-output", required=True)
    execute.add_argument("--receipt-output", required=True)
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=60)
    execute.add_argument("--actor-id")

    publish = commands.add_parser("publish")
    publish.add_argument("--snapshot", required=True)
    publish.add_argument("--target-db-path", required=True)
    publish.add_argument("--backup-output", required=True)
    publish.add_argument("--receipt-output", required=True)
    publish.add_argument("--confirm-snapshot-digest", required=True)
    publish.add_argument("--worker-id", required=True)
    publish.add_argument("--lease-seconds", type=int, default=60)
    publish.add_argument("--max-attempts", type=int, default=3)
    publish.add_argument("--actor-id")

    status = commands.add_parser("status")
    status.add_argument("artifact_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument(
        "--state",
        choices=("planned", "running", "completed", "orphaned", "failed", "cancelled"),
    )
    listing.add_argument("--limit", type=int, default=100)

    retry = commands.add_parser("retry")
    retry.add_argument("artifact_id")
    retry.add_argument("--owner-id", required=True)
    retry.add_argument("--confirm-artifact-id", required=True)

    cancel = commands.add_parser("cancel")
    cancel.add_argument("artifact_id")
    cancel.add_argument("--owner-id", required=True)
    cancel.add_argument("--confirm-artifact-id", required=True)
    return parser


def _seed(args: Any) -> Any:
    snapshot = verify_signed_retirement_snapshot(args.snapshot)
    if args.confirm_snapshot_digest != snapshot.snapshot_digest:
        raise ValueError("snapshot confirmation differs.")
    journal = get_restore_custody_artifact_journal(
        protected_paths=_protected(args)
    )
    return seed_restore_custody_artifact_attempt(
        snapshot_path=args.snapshot,
        target_db_path=args.target_db_path,
        backup_output_path=args.backup_output,
        receipt_output_path=args.receipt_output,
        journal=journal,
        max_attempts=args.max_attempts,
    )


def _execute(args: Any, *, artifact_id: str) -> Any:
    if args.confirm_artifact_id != artifact_id:
        raise ValueError("artifact confirmation differs.")
    binding = require_relation_review_actor(
        args.actor_id,
        binding=load_relation_review_actor(),
    )
    journal = get_restore_custody_artifact_journal(
        protected_paths=_protected(args)
    )
    return execute_restore_custody_artifact_attempt(
        artifact_id,
        snapshot_path=args.snapshot,
        target_db_path=args.target_db_path,
        backup_output_path=args.backup_output,
        receipt_output_path=args.receipt_output,
        actor=binding,
        worker_id=args.worker_id,
        lease_seconds=args.lease_seconds,
        journal=journal,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "seed":
            value = _seed(args)
            payload = _attempt_summary(value)
            payload.update(
                {
                    "journal_mutation_performed": True,
                    "artifact_mutation_performed": False,
                    "artifact_deletion_performed": False,
                    "artifact_overwrite_performed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "execute":
            value = _execute(args, artifact_id=args.artifact_id)
            _print(_execution_summary(value))
            return 0
        if args.command == "publish":
            seeded = _seed(args)
            args.confirm_artifact_id = seeded.artifact_id
            value = _execute(args, artifact_id=seeded.artifact_id)
            payload = _execution_summary(value)
            payload["seeded_artifact_id"] = seeded.artifact_id
            _print(payload)
            return 0
        if args.command == "status":
            journal = get_restore_custody_artifact_journal()
            payload = _attempt_summary(journal.get(args.artifact_id))
            payload.update(
                {
                    "journal_mutation_performed": False,
                    "artifact_mutation_performed": False,
                    "artifact_deletion_performed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "list":
            journal = get_restore_custody_artifact_journal()
            values = journal.list(
                owner_id=args.owner_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "state": args.state,
                    "count": len(values),
                    "items": [_attempt_summary(value) for value in values],
                    "journal_mutation_performed": False,
                    "artifact_mutation_performed": False,
                    "artifact_deletion_performed": False,
                    "source_text_returned": False,
                    "raw_paths_returned": False,
                }
            )
            return 0
        if args.command in {"retry", "cancel"}:
            if args.confirm_artifact_id != args.artifact_id:
                raise ValueError("artifact confirmation differs.")
            journal = get_restore_custody_artifact_journal()
            method = journal.retry if args.command == "retry" else journal.cancel
            value = method(
                args.artifact_id,
                owner_id=args.owner_id,
                confirm_artifact_id=args.confirm_artifact_id,
            )
            payload = _attempt_summary(value)
            payload.update(
                {
                    "journal_mutation_performed": True,
                    "artifact_mutation_performed": False,
                    "artifact_deletion_performed": False,
                }
            )
            _print(payload)
            return 0
        raise ValueError("unsupported custody artifact command.")
    except RestoreCustodyArtifactRecoveryError as exc:
        _print(
            {
                "error": "artifact_publication_failed",
                "artifact_id": exc.artifact_id,
                "state": exc.state,
                "phase": exc.phase,
                "artifact_deletion_performed": False,
                "artifact_overwrite_performed": False,
                "source_text_returned": False,
                "raw_paths_returned": False,
            },
            stream=sys.stderr,
        )
        return 1
    except PermissionError:
        _print({"error": "not_authorized"}, stream=sys.stderr)
        return 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
