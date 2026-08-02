"""Operator CLI for exact-generation derived evidence-graph reconciliation."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_job_runtime import get_evidence_graph_job_journal
from tools.evidence_graph_jobs import EvidenceGraphJob
from tools.evidence_graph_reconcile import (
    EvidenceGraphReconciliationError,
    execute_next_graph_job,
    seed_current_graph_job,
)
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.sparse_runtime import get_generation_store, get_sparse_index


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(job: EvidenceGraphJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "owner_id": job.owner_id,
        "doc_id": job.doc_id,
        "source_sequence": job.source_sequence,
        "source_state": job.source_state,
        "content_sha256": job.content_sha256,
        "profile_fingerprint": job.profile_fingerprint,
        "sparse_generation": job.sparse_generation,
        "state": job.state,
        "attempt_count": job.attempt_count,
        "max_attempts": job.max_attempts,
        "lease_owner": job.lease_owner,
        "lease_expires_at": job.lease_expires_at,
        "graph_digest": job.graph_digest,
        "failure_type": job.failure_type,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "authoritative_mutation_performed": False,
        "semantic_inference_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_jobs_cli",
        description=(
            "Seed and reconcile exact authoritative generations into the derived "
            "structural evidence graph. This CLI does not mutate authoritative indexes."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed", help="Seed the current authoritative generation.")
    seed.add_argument("--owner-id", required=True)
    seed.add_argument("--doc-id", required=True)
    seed.add_argument("--max-attempts", type=int, default=3)

    status = commands.add_parser("status", help="Inspect one graph job.")
    status.add_argument("job_id")

    listing = commands.add_parser("list", help="List bounded graph jobs.")
    listing.add_argument("--owner-id")
    listing.add_argument("--state")
    listing.add_argument("--limit", type=int, default=100)

    reconcile = commands.add_parser(
        "reconcile-one", help="Claim and reconcile one exact-generation job."
    )
    reconcile.add_argument("--owner-id", required=True)
    reconcile.add_argument("--worker-id", required=True)
    reconcile.add_argument("--lease-seconds", type=int, default=60)

    retry = commands.add_parser("retry", help="Reset one failed job after review.")
    retry.add_argument("job_id")
    retry.add_argument("--owner-id", required=True)
    retry.add_argument("--confirm-job-id", required=True)

    cancel = commands.add_parser("cancel", help="Cancel one planned or failed job.")
    cancel.add_argument("job_id")
    cancel.add_argument("--owner-id", required=True)
    cancel.add_argument("--confirm-job-id", required=True)
    return parser


def _seed(args: argparse.Namespace) -> int:
    job = seed_current_graph_job(
        owner_id=args.owner_id,
        doc_id=args.doc_id,
        generations=get_generation_store(),
        journal=get_evidence_graph_job_journal(),
        max_attempts=args.max_attempts,
    )
    _print(_summary(job))
    return 0


def _status(args: argparse.Namespace) -> int:
    job = get_evidence_graph_job_journal().get(args.job_id)
    if job is None:
        _print({"error": "not_found", "job_id": args.job_id}, stream=sys.stderr)
        return 1
    _print(_summary(job))
    return 0


def _list(args: argparse.Namespace) -> int:
    jobs = get_evidence_graph_job_journal().list(
        owner_id=args.owner_id,
        state=args.state,
        limit=args.limit,
    )
    _print({"count": len(jobs), "jobs": [_summary(job) for job in jobs]})
    return 0


def _reconcile(args: argparse.Namespace) -> int:
    result = execute_next_graph_job(
        owner_id=args.owner_id,
        worker_id=args.worker_id,
        journal=get_evidence_graph_job_journal(),
        generations=get_generation_store(),
        sparse=get_sparse_index(),
        graphs=get_evidence_graph_store(),
        lease_seconds=args.lease_seconds,
    )
    if result is None:
        _print({"status": "idle", "owner_id": args.owner_id})
        return 0
    _print(_summary(result))
    return 0


def _confirmed(args: argparse.Namespace) -> None:
    if args.confirm_job_id != args.job_id:
        raise ValueError("confirmation must exactly match job_id.")


def _retry(args: argparse.Namespace) -> int:
    _confirmed(args)
    result = get_evidence_graph_job_journal().retry_failed(
        args.job_id,
        owner_id=args.owner_id,
    )
    _print(_summary(result))
    return 0


def _cancel(args: argparse.Namespace) -> int:
    _confirmed(args)
    result = get_evidence_graph_job_journal().cancel(
        args.job_id,
        owner_id=args.owner_id,
    )
    _print(_summary(result))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "seed":
            return _seed(args)
        if args.command == "status":
            return _status(args)
        if args.command == "list":
            return _list(args)
        if args.command == "reconcile-one":
            return _reconcile(args)
        if args.command == "retry":
            return _retry(args)
        if args.command == "cancel":
            return _cancel(args)
        raise ValueError("unsupported evidence graph job command.")
    except EvidenceGraphReconciliationError:
        _print({"error": "reconciliation_failed"}, stream=sys.stderr)
        return 1
    except KeyError:
        _print({"error": "not_found"}, stream=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
