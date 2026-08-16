#!/usr/bin/env python3
"""Run bounded deterministic hydrology recomputation from the authoritative stale ledger.

The command performs no model/provider calls and never daemonizes. It rebuilds only
recipe-backed plans, projections and deterministic hydrology reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process bounded deterministic hydrology recompute work.")
    parser.add_argument("--owner-id", required=True, help="Owner whose hydrology recompute work is processed.")
    parser.add_argument("--max-tasks", type=int, default=100, help="Maximum tasks to process (1-10000).")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum authoritative attempts per task (1-100).")
    parser.add_argument("--retry-task", default="", help="Requeue one failed authoritative task before processing.")
    parser.add_argument("--retry-only", action="store_true", help="Only requeue --retry-task; do not process work.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.max_tasks <= 10_000:
        raise SystemExit("--max-tasks must be between 1 and 10000")
    if not 1 <= args.max_attempts <= 100:
        raise SystemExit("--max-attempts must be between 1 and 100")
    if args.retry_only and not args.retry_task:
        raise SystemExit("--retry-only requires --retry-task")

    from production_app import hydrology, hydrology_recipes, invalidations, replacements
    from tools.hydrology_recompute_executor import HydrologyRecomputeExecutor
    from tools.recompute_executor import requeue_failed_task

    executor = HydrologyRecomputeExecutor(
        invalidations=invalidations,
        replacements=replacements,
        store=hydrology,
        recipes=hydrology_recipes,
    )
    retried = None
    if args.retry_task:
        retried = requeue_failed_task(invalidations, args.owner_id, args.retry_task)
        if args.retry_only:
            print(
                json.dumps(
                    {
                        "owner_id": args.owner_id,
                        "task_id": args.retry_task,
                        "requeued": retried,
                    },
                    sort_keys=True,
                )
            )
            return 0 if retried else 2

    values = executor.drain(
        args.owner_id,
        limit=args.max_tasks,
        max_attempts=args.max_attempts,
    )
    failed = sum(1 for item in values if item.status != "completed")
    payload = {
        "owner_id": args.owner_id,
        "retried": retried,
        "attempted": len(values),
        "completed": len(values) - failed,
        "failed": failed,
        "outcomes": [
            {
                "task_id": item.task_id,
                "artifact_kind": item.artifact_kind,
                "old_fingerprint": item.old_fingerprint,
                "new_fingerprint": item.new_fingerprint,
                "project_id": item.project_id,
                "logical_id": item.logical_id,
                "status": item.status,
                "error_type": item.error_type,
            }
            for item in values
        ],
    }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
