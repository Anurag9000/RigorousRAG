"""Read-only CLI for signed relation-review actor-use reservations."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_actor_use_runtime import get_signed_actor_use_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "assertion_digest": value.assertion_digest,
        "decision_id": value.decision_id,
        "proposal_id": value.proposal_id,
        "owner_id": value.owner_id,
        "graph_set_key": value.graph_set_key,
        "decision": value.decision,
        "actor_id": value.actor_id,
        "issuer": value.issuer,
        "binding_digest": value.binding_digest,
        "assertion_expires_at": value.assertion_expires_at,
        "use_digest": value.use_digest,
        "state": value.state,
        "reserved_at": value.reserved_at,
        "committed_at": value.committed_at,
        "updated_at": value.updated_at,
        "contains_signature": False,
        "contains_key_material": False,
        "contains_source_text": False,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_relation_actor_use_cli",
        description=(
            "Inspect signed actor assertion reservations. This CLI cannot reserve, "
            "commit, alter, retry or delete actor-use records."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("assertion_digest")
    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--decision-id")
    listing.add_argument("--state", choices=("reserved", "committed"))
    listing.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        store = get_signed_actor_use_store()
        if args.command == "status":
            value = store.get(args.assertion_digest)
            if value is None:
                raise KeyError(args.assertion_digest)
            _print(_summary(value))
            return 0
        if args.command == "list":
            values = store.list(
                owner_id=args.owner_id,
                decision_id=args.decision_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "actor_uses": [_summary(value) for value in values],
                    "contains_signature": False,
                    "contains_key_material": False,
                    "contains_source_text": False,
                    "mutation_performed": False,
                }
            )
            return 0
        raise ValueError("unsupported actor-use command.")
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
