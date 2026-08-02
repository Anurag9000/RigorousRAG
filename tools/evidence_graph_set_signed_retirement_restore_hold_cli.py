"""Operator CLI for durable signed-retirement restore legal holds."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
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


def _summary(value: Any) -> dict[str, Any]:
    return {
        "hold_id": value.hold_id,
        "owner_id": value.owner_id,
        "restore_id": value.restore_id,
        "hold_key": value.hold_key,
        "reason_code": value.reason_code,
        "status": value.status,
        "created_actor_id": value.created_actor_id,
        "created_binding_method": value.created_binding_method,
        "created_binding_digest": value.created_binding_digest,
        "created_at": value.created_at,
        "released_actor_id": value.released_actor_id,
        "released_binding_method": value.released_binding_method,
        "released_binding_digest": value.released_binding_digest,
        "released_at": value.released_at,
        "hold_digest": value.hold_digest,
        "contains_source_text": False,
        "raw_paths_returned": False,
        "deletion_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_hold_cli"
        ),
        description=(
            "Place, release and inspect process-owned legal holds on "
            "signed-retirement restore intents. Holds never delete history."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    place = commands.add_parser("place")
    place.add_argument("restore_id")
    place.add_argument("--owner-id", required=True)
    place.add_argument("--confirm-restore-id", required=True)
    place.add_argument("--hold-key", required=True)
    place.add_argument("--reason-code", required=True)
    place.add_argument("--actor-id")

    release = commands.add_parser("release")
    release.add_argument("hold_id")
    release.add_argument("--owner-id", required=True)
    release.add_argument("--confirm-hold-id", required=True)
    release.add_argument("--actor-id")

    status = commands.add_parser("status")
    status.add_argument("hold_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--restore-id")
    listing.add_argument("--status")
    listing.add_argument("--limit", type=int, default=100)

    active = commands.add_parser("active-restore-ids")
    active.add_argument("--owner-id", required=True)
    active.add_argument("--limit", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if (
            args.command == "place"
            and args.restore_id != args.confirm_restore_id
        ):
            raise ValueError("restore confirmation differs.")
        if (
            args.command == "release"
            and args.hold_id != args.confirm_hold_id
        ):
            raise ValueError("hold confirmation differs.")

        if args.command in {"status", "list", "active-restore-ids"}:
            store = get_signed_retirement_restore_hold_store()
            if args.command == "status":
                payload = _summary(store.get(args.hold_id))
                payload["mutation_performed"] = False
                _print(payload)
                return 0
            if args.command == "list":
                values = store.list(
                    owner_id=args.owner_id,
                    restore_id=args.restore_id,
                    status=args.status,
                    limit=args.limit,
                )
                _print(
                    {
                        "count": len(values),
                        "holds": [_summary(value) for value in values],
                        "mutation_performed": False,
                        "contains_source_text": False,
                        "raw_paths_returned": False,
                    }
                )
                return 0
            values = sorted(
                store.active_restore_ids(
                    owner_id=args.owner_id,
                    limit=args.limit,
                )
            )
            _print(
                {
                    "count": len(values),
                    "restore_ids": values,
                    "mutation_performed": False,
                    "contains_source_text": False,
                    "raw_paths_returned": False,
                }
            )
            return 0

        binding = require_relation_review_actor(
            getattr(args, "actor_id", None),
            binding=load_relation_review_actor(),
        )
        store = get_signed_retirement_restore_hold_store()
        if args.command == "place":
            value = store.place(
                owner_id=args.owner_id,
                restore_id=args.restore_id,
                hold_key=args.hold_key,
                reason_code=args.reason_code,
                actor=binding,
                restore_journal=get_signed_retirement_restore_journal(),
            )
            payload = _summary(value)
            payload["hold_mutation_performed"] = True
            payload["restore_mutation_performed"] = False
            _print(payload)
            return 0
        if args.command == "release":
            value = store.release(
                args.hold_id,
                owner_id=args.owner_id,
                confirm_hold_id=args.confirm_hold_id,
                actor=binding,
            )
            payload = _summary(value)
            payload["hold_mutation_performed"] = True
            payload["restore_mutation_performed"] = False
            _print(payload)
            return 0
        raise ValueError("unsupported restore hold command.")
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
