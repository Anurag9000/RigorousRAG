#!/usr/bin/env python3
"""Reconcile bounded pending source-trust activations into research invalidation.

This command performs one explicit cycle and exits. It does not daemonize, poll, load
models or execute recomputation. The durable activation outbox remains authoritative if
this command fails partway through.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile pending RigorousRAG source-trust governance activations."
    )
    parser.add_argument("--owner-id", required=True, help="Owner whose pending activations are reconciled.")
    parser.add_argument("--source-id", default=None, help="Optional single source identifier.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum activations to process (1-10000).")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed activation instead of continuing the bounded cycle.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.limit <= 10_000:
        raise SystemExit("--limit must be between 1 and 10000")
    if args.source_id is not None and not args.source_id.strip():
        raise SystemExit("--source-id may not be empty")

    # Import after parsing so trusted deployment bootstraps can register providers first.
    from production_app import invalidations, source_trust
    from tools.source_trust_reconciliation import reconcile_source_trust_activations

    summary = reconcile_source_trust_activations(
        source_trust,
        invalidations,
        args.owner_id,
        source_id=args.source_id,
        limit=args.limit,
        stop_on_error=args.stop_on_error,
    )
    payload = {
        "owner_id": summary.owner_id,
        "attempted": summary.attempted,
        "completed": summary.completed,
        "failed": summary.failed,
        "affected_artifacts": summary.affected_artifacts,
        "recompute_tasks": summary.recompute_tasks,
        "outcomes": [asdict(item) for item in summary.outcomes],
    }
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    return 0 if summary.failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
