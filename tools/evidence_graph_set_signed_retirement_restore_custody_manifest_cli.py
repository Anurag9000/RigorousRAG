"""Operator CLI for durable restore custody-manifest binding."""

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
from tools.evidence_graph_set_signed_retirement_restore_custody_runtime import (
    get_signed_retirement_restore_custody_store,
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


def _summary(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
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
            "tools.evidence_graph_set_signed_retirement_restore_custody_manifest_cli"
        ),
        description=(
            "Bind verified pre/post custody receipts to one restore intent. "
            "Manifest commands never execute or delete a restore."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pre = commands.add_parser("bind-pre")
    pre.add_argument("restore_id")
    pre.add_argument("--confirm-restore-id", required=True)
    pre.add_argument("--pre-receipt", required=True)
    pre.add_argument("--backup", required=True)
    pre.add_argument("--actor-id")

    post = commands.add_parser("bind-post")
    post.add_argument("restore_id")
    post.add_argument("--confirm-restore-id", required=True)
    post.add_argument("--post-receipt", required=True)
    post.add_argument("--actor-id")

    status = commands.add_parser("status")
    status.add_argument("custody_id")

    restore_status = commands.add_parser("status-for-restore")
    restore_status.add_argument("restore_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if (
            args.command in {"bind-pre", "bind-post"}
            and args.restore_id != args.confirm_restore_id
        ):
            raise ValueError("restore confirmation differs.")
        if args.command in {"status", "status-for-restore", "list"}:
            store = get_signed_retirement_restore_custody_store()
            if args.command == "status":
                payload = _summary(store.get(args.custody_id))
                payload["mutation_performed"] = False
                _print(payload)
                return 0
            if args.command == "status-for-restore":
                payload = _summary(store.get_for_restore(args.restore_id))
                payload["mutation_performed"] = False
                _print(payload)
                return 0
            values = store.list(
                owner_id=args.owner_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "custody_manifests": [_summary(value) for value in values],
                    "mutation_performed": False,
                    "contains_source_text": False,
                    "raw_paths_returned": False,
                }
            )
            return 0

        binding = require_relation_review_actor(
            args.actor_id,
            binding=load_relation_review_actor(),
        )
        store = get_signed_retirement_restore_custody_store()
        restore_journal = get_signed_retirement_restore_journal()
        if args.command == "bind-pre":
            value = store.bind_pre(
                restore_id=args.restore_id,
                pre_receipt_path=args.pre_receipt,
                backup_path=args.backup,
                restore_journal=restore_journal,
                actor=binding,
            )
        elif args.command == "bind-post":
            value = store.bind_post(
                restore_id=args.restore_id,
                post_receipt_path=args.post_receipt,
                restore_journal=restore_journal,
                actor=binding,
            )
        else:
            raise ValueError("unsupported custody manifest command.")
        payload = _summary(value)
        payload["custody_mutation_performed"] = True
        _print(payload)
        return 0
    except PermissionError:
        _print({"error": "not_authorized"}, stream=sys.stderr)
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
