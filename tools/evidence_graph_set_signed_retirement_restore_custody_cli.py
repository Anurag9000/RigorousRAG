"""Operator CLI for pre/post signed-retirement restore custody receipts."""

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
from tools.evidence_graph_set_signed_retirement_restore_custody_boundary import (
    create_post_restore_comparison_receipt,
    create_pre_restore_backup_receipt,
    verify_post_restore_comparison_receipt,
    verify_pre_restore_backup_receipt,
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


def _summary(value: Any, *, kind: str) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "receipt_kind": kind,
            "contains_source_text": False,
            "contains_assertion_secrets": False,
            "raw_paths_returned": False,
            "restore_mutation_performed": False,
            "target_mutation_performed": False,
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_cli"
        ),
        description=(
            "Create or verify pre-restore backup and post-restore comparison "
            "receipts. Receipt commands never mutate restore or target state."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pre_create = commands.add_parser("pre-create")
    pre_create.add_argument("--snapshot", required=True)
    pre_create.add_argument("--target-db-path", required=True)
    pre_create.add_argument("--backup-output", required=True)
    pre_create.add_argument("--receipt-output", required=True)
    pre_create.add_argument("--confirm-snapshot-digest", required=True)
    pre_create.add_argument("--actor-id")

    pre_verify = commands.add_parser("pre-verify")
    pre_verify.add_argument("--receipt", required=True)
    pre_verify.add_argument("--backup", required=True)

    post_create = commands.add_parser("post-create")
    post_create.add_argument("restore_id")
    post_create.add_argument("--confirm-restore-id", required=True)
    post_create.add_argument("--snapshot", required=True)
    post_create.add_argument("--target-db-path", required=True)
    post_create.add_argument("--pre-receipt", required=True)
    post_create.add_argument("--backup", required=True)
    post_create.add_argument("--receipt-output", required=True)
    post_create.add_argument("--actor-id")

    post_verify = commands.add_parser("post-verify")
    post_verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "pre-create":
            snapshot = verify_signed_retirement_snapshot(args.snapshot)
            if args.confirm_snapshot_digest != snapshot.snapshot_digest:
                raise ValueError("snapshot confirmation differs.")
            binding = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            receipt = create_pre_restore_backup_receipt(
                snapshot_path=args.snapshot,
                target_db_path=args.target_db_path,
                backup_output_path=args.backup_output,
                receipt_output_path=args.receipt_output,
                actor=binding,
            )
            _print(_summary(receipt, kind="pre_restore_backup"))
            return 0
        if args.command == "pre-verify":
            receipt = verify_pre_restore_backup_receipt(
                receipt_path=args.receipt,
                backup_path=args.backup,
            )
            _print(_summary(receipt, kind="pre_restore_backup"))
            return 0
        if args.command == "post-create":
            if args.restore_id != args.confirm_restore_id:
                raise ValueError("restore confirmation differs.")
            binding = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            receipt = create_post_restore_comparison_receipt(
                restore_id=args.restore_id,
                snapshot_path=args.snapshot,
                target_db_path=args.target_db_path,
                pre_restore_receipt_path=args.pre_receipt,
                backup_path=args.backup,
                receipt_output_path=args.receipt_output,
                restore_journal=get_signed_retirement_restore_journal(),
                actor=binding,
            )
            _print(_summary(receipt, kind="post_restore_comparison"))
            return 0
        if args.command == "post-verify":
            receipt = verify_post_restore_comparison_receipt(args.receipt)
            _print(_summary(receipt, kind="post_restore_comparison"))
            return 0
        raise ValueError("unsupported custody command.")
    except PermissionError:
        _print({"error": "not_authorized"}, stream=sys.stderr)
        return 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
