#!/usr/bin/env python3
"""Run one explicit bounded research recomputation operator action.

Modes:
- ``local``: claim and execute directly from the authoritative invalidation ledger.
- ``publish``: publish queued task identifiers to the configured durable transport.
- ``worker``: consume a bounded number of durable handoffs and exact-claim each task.

The CLI never daemonizes or polls continuously. Distributed modes fail closed unless a
durable transport is explicitly configured.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process bounded RigorousRAG recompute work.")
    parser.add_argument("--owner-id", required=True, help="Owner whose recompute work is processed.")
    parser.add_argument(
        "--mode",
        choices=("local", "publish", "worker"),
        default="local",
        help="Operator action. Defaults to the historical local execution mode.",
    )
    parser.add_argument("--max-tasks", type=int, default=100, help="Maximum tasks/handoffs (1-10000).")
    parser.add_argument("--retry-task", default="", help="Requeue one failed authoritative task first.")
    parser.add_argument("--retry-only", action="store_true", help="Only requeue --retry-task; do not process work.")
    parser.add_argument("--worker-id", default="", help="Explicit worker identity required by --mode worker.")
    parser.add_argument(
        "--visibility-timeout",
        type=float,
        default=None,
        help="Optional worker transport lease override in seconds.",
    )
    parser.add_argument(
        "--busy-retry-delay",
        type=float,
        default=None,
        help="Optional delay before a busy authoritative claim is transport-visible again.",
    )
    return parser


def _distributed_bridge(args, invalidations, recompute_executor):
    from tools.distributed_recompute import DistributedRecomputeBridge
    from tools.recompute_queue_runtime import build_recompute_queue, load_recompute_transport_config

    config = load_recompute_transport_config()
    queue = build_recompute_queue(config)
    bridge = DistributedRecomputeBridge(
        owner_id=args.owner_id,
        invalidations=invalidations,
        executor=recompute_executor,
        queue=queue,
        max_attempts=config.ledger_max_attempts,
        claim_timeout_seconds=config.claim_timeout_seconds,
    )
    return bridge, config


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.max_tasks <= 10_000:
        raise SystemExit("--max-tasks must be between 1 and 10000")
    if args.retry_only and not args.retry_task:
        raise SystemExit("--retry-only requires --retry-task")
    if args.mode == "worker" and not args.worker_id.strip():
        raise SystemExit("--mode worker requires --worker-id")
    if args.mode == "worker" and args.retry_task and not args.retry_only:
        raise SystemExit("worker mode does not publish retries; use --mode publish with --retry-task first")

    from production_app import invalidations, recompute_executor
    from tools.recompute_operations import (
        publish_recompute_tasks,
        retry_failed_recompute,
        run_distributed_recompute_cycle,
        run_recompute_cycle,
    )

    retried = None
    if args.retry_task:
        retried = retry_failed_recompute(recompute_executor, args.owner_id, args.retry_task)
        if args.retry_only:
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "owner_id": args.owner_id,
                        "task_id": args.retry_task,
                        "requeued": retried,
                    },
                    sort_keys=True,
                )
            )
            return 0 if retried else 2

    if args.mode == "local":
        summary = run_recompute_cycle(
            recompute_executor,
            args.owner_id,
            max_tasks=args.max_tasks,
        )
        payload = {
            "mode": "local",
            "owner_id": summary.owner_id,
            "retried": retried,
            "attempted": summary.attempted,
            "succeeded": summary.succeeded,
            "failed": summary.failed,
            "replacements": summary.replacements,
            "outcomes": list(summary.outcomes),
        }
        print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        return 0 if summary.failed == 0 else 3

    try:
        bridge, config = _distributed_bridge(args, invalidations, recompute_executor)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.mode == "publish":
        summary = publish_recompute_tasks(bridge, limit=args.max_tasks)
        print(
            json.dumps(
                {
                    "mode": "publish",
                    "transport": config.backend,
                    "owner_id": summary.owner_id,
                    "retried": retried,
                    "handoffs": summary.handoffs,
                },
                sort_keys=True,
            )
        )
        return 0

    visibility = (
        config.visibility_timeout_seconds
        if args.visibility_timeout is None
        else args.visibility_timeout
    )
    busy_delay = (
        config.busy_retry_delay_seconds
        if args.busy_retry_delay is None
        else args.busy_retry_delay
    )
    try:
        summary = run_distributed_recompute_cycle(
            bridge,
            worker_id=args.worker_id,
            max_tasks=args.max_tasks,
            visibility_timeout=visibility,
            busy_retry_delay=busy_delay,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    payload = {
        "mode": "worker",
        "transport": config.backend,
        "owner_id": summary.owner_id,
        "worker_id": summary.worker_id,
        "attempted": summary.attempted,
        "completed": summary.completed,
        "failed": summary.failed,
        "busy": summary.busy,
        "duplicates": summary.duplicates,
        "invalid": summary.invalid,
        "outcomes": list(summary.outcomes),
    }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return 0 if summary.failed == 0 and summary.invalid == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
