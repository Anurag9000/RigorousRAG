"""Safe operator CLI for migration inventory, journal seeding and status."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.document_store import get_document_store
from tools.migration_planner import inventory_profile_migrations
from tools.migration_runtime import get_migration_journal
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_generation_store

_MAX_ROWS = 10_000


def _limit(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_ROWS:
        raise ValueError(f"limit must be an integer between 1 and {_MAX_ROWS}.")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="index-migrations",
        description=(
            "Inventory and journal embedding-profile migrations. This command "
            "does not execute reindexing or cutover."
        ),
    )
    parser.add_argument("command", choices=("inventory", "seed", "status", "cancel"))
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--target-profile")
    parser.add_argument("--state")
    parser.add_argument("--task-id")
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--pretty", action="store_true")
    return parser


def _candidates(owner: str, target: str, limit: int):
    return inventory_profile_migrations(
        owner_id=owner,
        target_profile=target,
        generations=get_generation_store(),
        document_store=get_document_store(),
        limit=limit,
    )


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    owner = normalize_owner_id(args.owner_id)
    limit = _limit(args.limit)
    if args.command in {"inventory", "seed"}:
        if not isinstance(args.target_profile, str) or not args.target_profile.strip():
            raise ValueError("target-profile is required for inventory and seed.")
        candidates = _candidates(owner, args.target_profile.strip(), limit)
        if args.command == "inventory":
            return {
                "command": "inventory",
                "owner_id": owner,
                "target_profile": args.target_profile.strip(),
                "candidates": [asdict(candidate) for candidate in candidates],
                "eligible": sum(1 for candidate in candidates if candidate.eligible),
            }
        journal = get_migration_journal()
        tasks = journal.seed(candidates)
        return {
            "command": "seed",
            "owner_id": owner,
            "target_profile": args.target_profile.strip(),
            "seeded": [asdict(task) for task in tasks],
            "ineligible": [
                asdict(candidate) for candidate in candidates if not candidate.eligible
            ],
        }
    journal = get_migration_journal()
    if args.command == "status":
        tasks = journal.list_tasks(
            owner_id=owner,
            state=args.state,
            limit=limit,
        )
        return {
            "command": "status",
            "owner_id": owner,
            "tasks": [asdict(task) for task in tasks],
        }
    if not isinstance(args.task_id, str) or not args.task_id.strip():
        raise ValueError("task-id is required for cancellation.")
    task = journal.cancel(task_id=args.task_id.strip())
    if task.owner_id != owner:
        raise RuntimeError("migration task does not belong to the selected owner.")
    return {"command": "cancel", "owner_id": owner, "task": asdict(task)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        payload = _payload(args)
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2 if args.pretty else None,
                allow_nan=False,
            )
        )
        return 0
    except SystemExit:
        raise
    except (ValueError, RuntimeError):
        print("Index migration operation could not be completed safely.", file=sys.stderr)
        return 1
    except Exception:
        print("Index migration control plane is unavailable.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
