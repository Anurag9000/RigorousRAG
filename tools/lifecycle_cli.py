"""Operator CLI for fourth-store lifecycle reconciliation.

Output contains only public lifecycle summaries. Retained source paths, provider
errors, database paths, and registry payloads are never serialized.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.lifecycle_runtime import (
    get_lifecycle_outbox,
    reconcile_lifecycle_pending,
)
from tools.security import normalize_owner_id

_MAX_LIMIT = 10_000


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def _positive(value: Any, label: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not 0.0 < parsed <= maximum:
        raise ValueError(f"{label} must be greater than zero and at most {maximum}.")
    return parsed


def _identifier(value: Any, label: str, maximum: int = 200) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string.")
    rendered = value.strip()
    if (
        not rendered
        or len(rendered) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError(f"{label} is invalid.")
    return rendered


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _summary(operation: Any) -> dict[str, Any]:
    public = get_lifecycle_outbox().public(operation)
    return asdict(public)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.lifecycle_cli",
        description="Inspect and reconcile durable lifecycle operations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pending = subparsers.add_parser("pending", help="List bounded pending operations.")
    pending.add_argument("--owner-id")
    pending.add_argument("--limit", type=int, default=100)

    status = subparsers.add_parser("status", help="Inspect one lifecycle operation.")
    status.add_argument("operation_id")

    reconcile = subparsers.add_parser(
        "reconcile",
        help="Claim and replay a bounded set of pending operations.",
    )
    reconcile.add_argument("--limit", type=int, default=100)
    reconcile.add_argument("--lease-seconds", type=float, default=60.0)
    reconcile.add_argument("--worker-id")

    retry = subparsers.add_parser(
        "retry-failed",
        help="Reset one failed operation after exact confirmation.",
    )
    retry.add_argument("operation_id")
    retry.add_argument("--confirm-operation-id", required=True)
    return parser


def _pending(args: argparse.Namespace) -> int:
    limit = _integer(args.limit, "limit", 1, _MAX_LIMIT)
    owner = normalize_owner_id(args.owner_id) if args.owner_id else None
    rows = get_lifecycle_outbox().list_pending(owner_id=owner, limit=limit)
    _print(
        {
            "count": len(rows),
            "operations": [asdict(row) for row in rows],
        }
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    operation_id = _identifier(args.operation_id, "operation_id")
    operation = get_lifecycle_outbox().get(operation_id)
    if operation is None:
        _print({"error": "not_found", "operation_id": operation_id}, stream=sys.stderr)
        return 1
    _print(_summary(operation))
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    limit = _integer(args.limit, "limit", 1, _MAX_LIMIT)
    lease = _positive(args.lease_seconds, "lease_seconds", 86_400.0)
    worker = (
        _identifier(args.worker_id, "worker_id")
        if args.worker_id is not None
        else None
    )
    results = reconcile_lifecycle_pending(
        limit=limit,
        lease_seconds=lease,
        worker_id=worker,
    )
    payload = {
        "count": len(results),
        "results": [asdict(result) for result in results],
    }
    _print(payload)
    return 1 if any(result.outcome in {"error", "failed"} for result in results) else 0


def _retry_failed(args: argparse.Namespace) -> int:
    operation_id = _identifier(args.operation_id, "operation_id")
    confirmation = _identifier(
        args.confirm_operation_id,
        "confirm_operation_id",
    )
    if confirmation != operation_id:
        raise ValueError("confirmation must exactly match operation_id.")
    outbox = get_lifecycle_outbox()
    current = outbox.get(operation_id)
    if current is None:
        _print({"error": "not_found", "operation_id": operation_id}, stream=sys.stderr)
        return 1
    if current.state != "failed":
        raise ValueError("only failed lifecycle operations may be retried.")
    retried = outbox.retry_failed(operation_id)
    _print(_summary(retried))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "pending":
            return _pending(args)
        if args.command == "status":
            return _status(args)
        if args.command == "reconcile":
            return _reconcile(args)
        if args.command == "retry-failed":
            return _retry_failed(args)
        raise ValueError("unsupported lifecycle command.")
    except (ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
