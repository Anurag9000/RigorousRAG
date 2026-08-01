"""Operator CLI for isolated migration shadow construction and validation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.migration_runtime import get_migration_journal
from tools.migration_shadow_runtime import (
    execute_next_shadow_build,
    get_migration_shadow_store,
)
from tools.migration_types import exact_integer, identifier
from tools.security import normalize_owner_id


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_shadow_cli",
        description=(
            "Build and validate isolated migration shadows. This CLI has no live "
            "cutover command."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    execute = subparsers.add_parser(
        "execute-one",
        help="Claim and build one planned/failed/expired-running task.",
    )
    execute.add_argument("--owner-id", required=True)
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=300)
    execute.add_argument("--max-attempts", type=int, default=3)

    validate = subparsers.add_parser(
        "validate",
        help="Validate one existing shadow against its journal task.",
    )
    validate.add_argument("task_id")

    remove = subparsers.add_parser(
        "remove",
        help="Remove isolated artifacts only for failed or cancelled tasks.",
    )
    remove.add_argument("task_id")
    remove.add_argument("--confirm-task-id", required=True)
    return parser


def _execute(args: argparse.Namespace) -> int:
    owner = normalize_owner_id(args.owner_id)
    worker = identifier(args.worker_id, "worker_id", 128)
    lease = exact_integer(args.lease_seconds, "lease_seconds", 1, 86_400)
    attempts = exact_integer(args.max_attempts, "max_attempts", 1, 100)
    result = execute_next_shadow_build(
        owner_id=owner,
        worker_id=worker,
        lease_seconds=lease,
        max_attempts=attempts,
    )
    if result is None:
        _print({"outcome": "no_buildable_task", "owner_id": owner})
        return 0
    _print(asdict(result))
    return 1 if result.outcome == "failed" else 0


def _validate(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    task = get_migration_journal().get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    manifest = get_migration_shadow_store().validate(task_id)
    matches = bool(
        manifest.owner_id == task.owner_id
        and manifest.doc_id == task.doc_id
        and manifest.source_sequence == task.source_sequence
        and manifest.source_profile_fingerprint
        == task.source_profile_fingerprint
        and manifest.target_profile_name == task.target_profile_name
        and manifest.target_profile_fingerprint
        == task.target_profile_fingerprint
        and (
            task.validation_digest is None
            or task.validation_digest == manifest.validation_digest
        )
    )
    if not matches:
        raise RuntimeError("shadow artifact does not match the migration journal.")
    _print(
        {
            "task_id": task_id,
            "task_state": task.state,
            "validation_digest": manifest.validation_digest,
            "vector_count": manifest.vector_count,
            "sparse_count": manifest.sparse_count,
            "content_sha256": manifest.content_sha256,
            "parser_fingerprint": manifest.parser_fingerprint,
        }
    )
    return 0


def _remove(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    confirmation = identifier(args.confirm_task_id, "confirm_task_id", 64)
    if task_id != confirmation:
        raise ValueError("confirmation must exactly match task_id.")
    task = get_migration_journal().get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    if task.state not in {"failed", "cancelled"}:
        raise ValueError(
            "shadow artifacts may be removed only for failed or cancelled tasks."
        )
    removed = get_migration_shadow_store().remove(task_id)
    _print({"task_id": task_id, "removed": removed})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "execute-one":
            return _execute(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "remove":
            return _remove(args)
        raise ValueError("unsupported migration shadow command.")
    except (ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
