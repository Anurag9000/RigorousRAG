"""Governed operator CLI for custody timestamp issuance legal holds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_readonly import (
    ReadOnlyCustodyTimestampIssuanceHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_runtime import (
    get_custody_timestamp_issuance_hold_store,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_readonly import (
    ReadOnlyCustodyTimestampIssuanceJournal,
)

_DEFAULT_HOLD_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_issuance_holds.sqlite3"
)
_DEFAULT_ISSUANCE_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_issuances.sqlite3"
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _hold_path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_HOLD_DB_PATH",
        _DEFAULT_HOLD_PATH,
    )


def _issuance_path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH",
        _DEFAULT_ISSUANCE_PATH,
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "hold_id": value.hold_id,
        "owner_id": value.owner_id,
        "issuance_id": value.issuance_id,
        "hold_key": value.hold_key,
        "reason_code": value.reason_code,
        "status": value.status,
        "created_binding_method": value.created_binding_method,
        "created_binding_digest": value.created_binding_digest,
        "created_at": value.created_at,
        "released_binding_method": value.released_binding_method,
        "released_binding_digest": value.released_binding_digest,
        "released_at": value.released_at,
        "hold_digest": value.hold_digest,
        "contains_actor_ids": False,
        "contains_raw_paths": False,
        "deletion_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_holds_cli"
        ),
        description=(
            "Place, release, and inspect durable legal holds over timestamp issuance "
            "records. Holds affect retention planning only."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    place = commands.add_parser("place")
    place.add_argument("--owner-id", required=True)
    place.add_argument("--issuance-id", required=True)
    place.add_argument("--hold-key", required=True)
    place.add_argument("--reason-code", required=True)
    place.add_argument("--actor-id")
    place.add_argument("--hold-db-path")
    place.add_argument("--issuance-db-path")

    release = commands.add_parser("release")
    release.add_argument("hold_id")
    release.add_argument("--owner-id", required=True)
    release.add_argument("--confirm-hold-id", required=True)
    release.add_argument("--actor-id")
    release.add_argument("--hold-db-path")

    status = commands.add_parser("status")
    status.add_argument("hold_id")
    status.add_argument("--hold-db-path")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--issuance-id")
    listing.add_argument("--status", choices=("active", "released"))
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--hold-db-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "release" and args.confirm_hold_id != args.hold_id:
            raise ValueError("timestamp issuance hold confirmation differs.")
        if args.command in {"place", "release"}:
            actor = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            store = get_custody_timestamp_issuance_hold_store(
                _hold_path(args.hold_db_path)
            )
            if args.command == "place":
                value = store.place(
                    owner_id=args.owner_id,
                    issuance_id=args.issuance_id,
                    hold_key=args.hold_key,
                    reason_code=args.reason_code,
                    actor=actor,
                    issuance_journal=ReadOnlyCustodyTimestampIssuanceJournal(
                        _issuance_path(args.issuance_db_path)
                    ),
                )
            else:
                value = store.release(
                    args.hold_id,
                    owner_id=args.owner_id,
                    confirm_hold_id=args.confirm_hold_id,
                    actor=actor,
                )
            _print(
                {
                    **_summary(value),
                    "hold_store_mutation_performed": True,
                    "issuance_journal_mutation_performed": False,
                }
            )
            return 0
        read_only = ReadOnlyCustodyTimestampIssuanceHoldStore(
            _hold_path(args.hold_db_path)
        )
        if args.command == "status":
            value = read_only.get(args.hold_id)
            _print({**_summary(value), "hold_store_mutation_performed": False})
            return 0
        values = read_only.list(
            owner_id=args.owner_id,
            issuance_id=args.issuance_id,
            status=args.status,
            limit=args.limit,
        )
        _print(
            {
                "owner_id": args.owner_id,
                "issuance_id": args.issuance_id,
                "status": args.status,
                "count": len(values),
                "items": [_summary(value) for value in values],
                "hold_store_mutation_performed": False,
                "issuance_journal_mutation_performed": False,
                "contains_actor_ids": False,
                "contains_raw_paths": False,
                "deletion_performed": False,
            }
        )
        return 0
    except PermissionError:
        _print({"error": "not_authorized_or_untrusted"}, stream=sys.stderr)
        return 1
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
