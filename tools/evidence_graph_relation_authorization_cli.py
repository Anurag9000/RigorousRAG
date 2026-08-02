"""Read-only operator CLI for semantic relation authorization receipts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_authorization_runtime import (
    get_relation_review_authorization_store,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(record: Any) -> dict[str, Any]:
    authorization = record.authorization
    return {
        "decision_id": authorization.decision_id,
        "proposal_id": authorization.proposal_id,
        "owner_id": authorization.owner_id,
        "graph_set_key": authorization.graph_set_key,
        "decision": authorization.decision,
        "reviewer_id": authorization.reviewer_id,
        "state": record.state,
        "policy_digest": authorization.policy_digest,
        "grant_digest": authorization.grant_digest,
        "authorization_digest": authorization.authorization_digest,
        "authorized_at": authorization.authorized_at,
        "prepared_at": record.prepared_at,
        "committed_at": record.committed_at,
        "updated_at": record.updated_at,
        "separation_of_duties_enforced": (
            authorization.separation_of_duties_enforced
        ),
        "replacement_scope_validated": (
            authorization.replacement_scope_validated
        ),
        "contains_source_text": False,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_relation_authorization_cli",
        description=(
            "Inspect immutable semantic-relation reviewer authorization receipts. "
            "This CLI cannot create, commit, retry, alter or delete receipts."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("decision_id")
    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--graph-set-key")
    listing.add_argument("--state", choices=("authorized", "committed"))
    listing.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        store = get_relation_review_authorization_store()
        if args.command == "status":
            record = store.get(args.decision_id)
            if record is None:
                raise KeyError(args.decision_id)
            _print(_summary(record))
            return 0
        if args.command == "list":
            records = store.list(
                owner_id=args.owner_id,
                graph_set_key=args.graph_set_key,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(records),
                    "authorizations": [_summary(value) for value in records],
                    "contains_source_text": False,
                    "mutation_performed": False,
                }
            )
            return 0
        raise ValueError("unsupported authorization command.")
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
