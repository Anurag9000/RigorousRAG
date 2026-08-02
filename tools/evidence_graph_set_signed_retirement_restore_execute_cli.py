"""Operator CLI for custody-governed empty-target retirement restores."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
)
from tools.evidence_graph_set_signed_retirement_restore_contracts import (
    _timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    verify_pre_restore_backup_receipt,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_runtime import (
    get_signed_retirement_restore_custody_store,
)
from tools.evidence_graph_set_signed_retirement_restore_reconcile import (
    SignedRetirementRestoreRecoveryError,
    execute_signed_retirement_restore,
    seed_signed_retirement_restore,
)
from tools.evidence_graph_set_signed_retirement_restore_runtime import (
    get_signed_retirement_restore_journal,
)
from tools.evidence_graph_set_signed_retirement_snapshot_boundary import (
    verify_signed_retirement_snapshot,
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
        "restore_id": value.restore_id,
        "owner_id": value.owner_id,
        "snapshot_digest": value.snapshot_digest,
        "target_path_digest": value.target_path_digest,
        "snapshot_record_count": value.snapshot_record_count,
        "state": value.state,
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "max_attempts": value.max_attempts,
        "lease_owner": value.lease_owner,
        "lease_expires_at": value.lease_expires_at,
        "target_verification_digest": value.target_verification_digest,
        "failure_type": value.failure_type,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "completed_at": value.completed_at,
        "contains_source_text": False,
        "contains_assertion_secrets": False,
        "overwrite_performed": False,
        "merge_performed": False,
    }


def _execution(value: Any, custody: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "custody_id": custody.custody_id,
            "custody_state": custody.state,
            "pre_custody_validated": True,
            "contains_source_text": False,
            "contains_assertion_secrets": False,
        }
    )
    return payload


def _require_custody_args(args: Any) -> None:
    if not getattr(args, "pre_receipt", None) or not getattr(args, "backup", None):
        raise ValueError("pre-restore custody receipt and backup are required.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_execute_cli"
        ),
        description=(
            "Seed, execute and recover custody-governed terminal retirement "
            "snapshot restores into an initialized empty target. "
            "No overwrite or merge exists."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed")
    seed.add_argument("--snapshot", required=True)
    seed.add_argument("--target-db-path", required=True)
    seed.add_argument("--confirm-snapshot-digest", required=True)
    seed.add_argument("--pre-receipt")
    seed.add_argument("--backup")
    seed.add_argument("--actor-id")
    seed.add_argument("--max-attempts", type=int, default=3)

    status = commands.add_parser("status")
    status.add_argument("restore_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)

    execute = commands.add_parser("execute")
    execute.add_argument("restore_id")
    execute.add_argument("--snapshot", required=True)
    execute.add_argument("--target-db-path", required=True)
    execute.add_argument("--pre-receipt")
    execute.add_argument("--backup")
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=60)

    reconcile = commands.add_parser("reconcile-one")
    reconcile.add_argument("--owner-id", required=True)
    reconcile.add_argument("--snapshot", required=True)
    reconcile.add_argument("--target-db-path", required=True)
    reconcile.add_argument("--pre-receipt")
    reconcile.add_argument("--backup")
    reconcile.add_argument("--worker-id", required=True)
    reconcile.add_argument("--lease-seconds", type=int, default=60)

    for name in ("retry", "cancel"):
        command = commands.add_parser(name)
        command.add_argument("restore_id")
        command.add_argument("--owner-id", required=True)
        command.add_argument("--confirm-restore-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if (
            args.command in {"retry", "cancel"}
            and args.restore_id != args.confirm_restore_id
        ):
            raise ValueError("restore confirmation differs.")
        if args.command == "seed":
            confirmed = verify_signed_retirement_snapshot(args.snapshot)
            if args.confirm_snapshot_digest != confirmed.snapshot_digest:
                raise ValueError("snapshot confirmation differs.")
            _require_custody_args(args)
            pre = verify_pre_restore_backup_receipt(
                receipt_path=args.pre_receipt,
                backup_path=args.backup,
            )
            if pre.snapshot_digest != confirmed.snapshot_digest:
                raise RuntimeError("pre custody receipt differs from snapshot.")
            binding = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            journal = get_signed_retirement_restore_journal(
                target_db_path=args.target_db_path
            )
            value, _target_digest = seed_signed_retirement_restore(
                snapshot_path=args.snapshot,
                target_db_path=args.target_db_path,
                journal=journal,
                confirm_snapshot_digest=args.confirm_snapshot_digest,
                max_attempts=args.max_attempts,
            )
            custody = get_signed_retirement_restore_custody_store().bind_pre(
                restore_id=value.restore_id,
                pre_receipt_path=args.pre_receipt,
                backup_path=args.backup,
                restore_journal=journal,
                actor=binding,
            )
            payload = _attempt(value)
            payload.update(
                {
                    "custody_id": custody.custody_id,
                    "custody_state": custody.state,
                    "pre_custody_validated": True,
                    "restore_intent_mutation_performed": True,
                    "custody_mutation_performed": True,
                    "target_mutation_performed": False,
                }
            )
            _print(payload)
            return 0

        if args.command in {"status", "list", "retry", "cancel"}:
            journal = get_signed_retirement_restore_journal()
            if args.command == "status":
                payload = _attempt(journal.get(args.restore_id))
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
                        "restores": [_attempt(value) for value in values],
                        "mutation_performed": False,
                        "contains_source_text": False,
                    }
                )
                return 0
            if args.command == "retry":
                value = journal.retry(
                    args.restore_id,
                    owner_id=args.owner_id,
                    confirm_restore_id=args.confirm_restore_id,
                )
            else:
                value = journal.cancel(
                    args.restore_id,
                    owner_id=args.owner_id,
                    confirm_restore_id=args.confirm_restore_id,
                )
            payload = _attempt(value)
            payload["restore_intent_mutation_performed"] = True
            payload["target_mutation_performed"] = False
            _print(payload)
            return 0

        _require_custody_args(args)
        journal = get_signed_retirement_restore_journal(
            target_db_path=args.target_db_path
        )
        custody_store = get_signed_retirement_restore_custody_store()
        if args.command == "execute":
            custody = custody_store.require_pre_bound(
                restore_id=args.restore_id,
                pre_receipt_path=args.pre_receipt,
                backup_path=args.backup,
                restore_journal=journal,
            )
            result = execute_signed_retirement_restore(
                args.restore_id,
                snapshot_path=args.snapshot,
                target_db_path=args.target_db_path,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                journal=journal,
            )
            _print(_execution(result, custody))
            return 0 if result.state == "completed" else 1
        if args.command == "reconcile-one":
            timestamp = _timestamp(time.time(), "now")
            restore_id = journal.next_claimable_id(
                owner_id=args.owner_id,
                now=timestamp,
            )
            if restore_id is None:
                _print(
                    {
                        "status": "idle",
                        "mutation_performed": False,
                        "contains_source_text": False,
                    }
                )
                return 0
            custody = custody_store.require_pre_bound(
                restore_id=restore_id,
                pre_receipt_path=args.pre_receipt,
                backup_path=args.backup,
                restore_journal=journal,
            )
            result = execute_signed_retirement_restore(
                restore_id,
                snapshot_path=args.snapshot,
                target_db_path=args.target_db_path,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                journal=journal,
                now=timestamp,
            )
            _print(_execution(result, custody))
            return 0 if result.state == "completed" else 1
        raise ValueError("unsupported restore command.")
    except SignedRetirementRestoreRecoveryError as exc:
        _print(
            {
                "error": "restore_failed",
                "restore_id": exc.restore_id,
                "state": exc.state,
                "phase": exc.phase,
                "overwrite_performed": False,
                "merge_performed": False,
                "contains_source_text": False,
            },
            stream=sys.stderr,
        )
        return 1
    except PermissionError:
        _print({"error": "not_authorized"}, stream=sys.stderr)
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
