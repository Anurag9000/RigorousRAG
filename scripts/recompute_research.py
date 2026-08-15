#!/usr/bin/env python3
"""Run one explicit bounded research recomputation cycle.

The CLI imports ``production_app`` only after argument parsing so deployments that need
injected providers can wrap/import this module from their own bootstrap instead. It does
not daemonize or poll continuously.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process bounded RigorousRAG recompute work.")
    parser.add_argument("--owner-id", required=True, help="Owner whose recompute queue is processed.")
    parser.add_argument("--max-tasks", type=int, default=100, help="Maximum tasks to process (1-10000).")
    parser.add_argument("--retry-task", default="", help="Requeue one failed task before processing.")
    parser.add_argument("--retry-only", action="store_true", help="Only requeue --retry-task; do not process work.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.max_tasks <= 10_000:
        raise SystemExit("--max-tasks must be between 1 and 10000")
    if args.retry_only and not args.retry_task:
        raise SystemExit("--retry-only requires --retry-task")

    # Importing the production app constructs the exact configured stores/agent factory.
    # A distributed deployment that requires pre-registered providers should invoke the
    # service helpers from its own trusted bootstrap before importing production_app.
    from production_app import recompute_executor
    from tools.recompute_operations import retry_failed_recompute, run_recompute_cycle

    retried = None
    if args.retry_task:
        retried = retry_failed_recompute(recompute_executor, args.owner_id, args.retry_task)
        if args.retry_only:
            print(json.dumps({"owner_id": args.owner_id, "task_id": args.retry_task, "requeued": retried}, sort_keys=True))
            return 0 if retried else 2

    summary = run_recompute_cycle(
        recompute_executor,
        args.owner_id,
        max_tasks=args.max_tasks,
    )
    payload = {
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


if __name__ == "__main__":
    sys.exit(main())
