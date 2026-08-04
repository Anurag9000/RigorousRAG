"""Operator CLI for governed recovery of stale hold-placement permits."""

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
from tools.evidence_graph_set_signed_retirement_restore_hold_permit_recovery import (
    get_hold_permit_recovery,
    list_hold_permit_recoveries,
    recover_abandoned_hold_placement_permit,
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
            value, ensure_ascii=False, sort_keys=True, allow_nan=False
        ) + "\n"
    )


def _summary(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    payload.update(
        {
            "permit_mutation_performed": False,
            "quarantine_hold_mutation_performed": False,
            "restore_record_mutation_performed": False,
            "deletion_performed": False,
            "source_text_returned": False,
            "raw_paths_returned": False,
        }
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_"
            "hold_permit_recovery_cli"
        ),
        description=(
            "Recover stale restore hold-placement permits. Missing holds are "
            "replaced by an active quarantine hold before permit release."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    recover = commands.add_parser("recover")
    recover.add_argument("hold_id")
    recover.add_argument("--owner-id", required=True)
    recover.add_argument("--confirm-hold-id", required=True)
    recover.add_argument("--confirm-permit-digest", required=True)
    recover.add_argument("--minimum-age-seconds", type=int, default=3600)
    recover.add_argument("--actor-id")

    status = commands.add_parser("status")
    status.add_argument("recovery_id")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if (
            args.command == "recover"
            and args.hold_id != args.confirm_hold_id
        ):
            raise ValueError("hold confirmation differs.")

        restore_journal = get_signed_retirement_restore_journal()
        if args.command == "status":
            payload = _summary(
                get_hold_permit_recovery(
                    restore_journal,
                    args.recovery_id,
                )
            )
            payload["mutation_performed"] = False
            _print(payload)
            return 0
        if args.command == "list":
            values = list_hold_permit_recoveries(
                restore_journal,
                owner_id=args.owner_id,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "recoveries": [_summary(value) for value in values],
                    "mutation_performed": False,
                    "source_text_returned": False,
                    "raw_paths_returned": False,
                }
            )
            return 0
        if args.command == "recover":
            binding = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            receipt, changed = recover_abandoned_hold_placement_permit(
                restore_journal=restore_journal,
                hold_store=get_signed_retirement_restore_hold_store(),
                owner_id=args.owner_id,
                hold_id=args.hold_id,
                confirm_hold_id=args.confirm_hold_id,
                confirm_permit_digest=args.confirm_permit_digest,
                actor=binding,
                minimum_age_seconds=args.minimum_age_seconds,
            )
            payload = _summary(receipt)
            payload["permit_mutation_performed"] = changed
            payload["quarantine_hold_mutation_performed"] = bool(
                changed and receipt.quarantine_hold_id is not None
            )
            payload["mutation_performed"] = changed
            _print(payload)
            return 0
        raise ValueError("unsupported permit recovery command.")
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
