"""Privacy-safe audit, retention and compaction CLI for evidence-graph jobs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_compaction import compact_evidence_graph_retention_plan
from tools.evidence_graph_compaction_reconciliation import (
    reconcile_evidence_graph_compactions,
)
from tools.evidence_graph_compaction_recovery import (
    EvidenceGraphCompactionRecoveryJournal,
    recover_reconciled_compaction_receipts_durable,
)
from tools.evidence_graph_compaction_runtime import get_evidence_graph_compaction_store
from tools.evidence_graph_job_runtime import get_evidence_graph_job_journal
from tools.evidence_graph_operations import (
    audit_evidence_graph_jobs,
    plan_evidence_graph_job_retention,
)
from tools.evidence_graph_runtime import get_evidence_graph_store
from tools.sparse_runtime import get_generation_store


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _compaction_summary(value: Any) -> dict[str, Any]:
    return {
        "job_id": value.job_id,
        "owner_id": value.owner_id,
        "doc_id": value.doc_id,
        "source_sequence": value.source_sequence,
        "job_state": value.job_state,
        "graph_digest": value.graph_digest,
        "action": value.action,
        "plan_digest": value.plan_digest,
        "phase": value.phase,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "authoritative_mutation_performed": False,
        "contains_graph_text": False,
    }


def _recovery_summary(value: Any) -> dict[str, Any]:
    return {
        "recovery_id": value.recovery_id,
        "owner_id": value.owner_id,
        "report_digest": value.report_digest,
        "confirmed_job_ids": value.confirmed_job_ids,
        "actor_id": value.actor_id,
        "reason_recorded": bool(value.reason),
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "completed_job_ids": value.completed_job_ids,
        "already_completed_job_ids": value.already_completed_job_ids,
        "result_digest": value.result_digest,
        "last_error_type": value.last_error_type,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "receipt_digest": value.receipt_digest,
        "authoritative_mutation_performed": False,
        "graph_payload_mutation_performed": False,
        "contains_graph_text": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evidence_graph_operations_cli",
        description=(
            "Audit derived graph jobs, plan conservative retention and compact only "
            "exactly confirmed non-current graph payloads. Authoritative stores and "
            "minimal graph-job audit rows are never deleted."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--owner-id", required=True)
    audit.add_argument("--limit", type=int, default=10_000)
    audit.add_argument("--as-of", type=float)

    retention = commands.add_parser("retention-plan")
    retention.add_argument("--owner-id", required=True)
    retention.add_argument("--min-age-seconds", type=float, required=True)
    retention.add_argument("--limit", type=int, default=10_000)
    retention.add_argument("--as-of", type=float)

    apply = commands.add_parser("retention-apply")
    apply.add_argument("--owner-id", required=True)
    apply.add_argument("--min-age-seconds", type=float, required=True)
    apply.add_argument("--limit", type=int, default=10_000)
    apply.add_argument("--as-of", type=float, required=True)
    apply.add_argument("--confirm-plan-digest", required=True)
    apply.add_argument("--confirm-job-id", action="append", required=True)

    status = commands.add_parser("compaction-status")
    status.add_argument("job_id")

    listing = commands.add_parser("compaction-list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--phase")
    listing.add_argument("--limit", type=int, default=100)

    reconcile = commands.add_parser("compaction-reconcile")
    reconcile.add_argument("--owner-id", required=True)
    reconcile.add_argument("--limit", type=int, default=100)
    reconcile.add_argument("--as-of", type=float)

    recover = commands.add_parser("compaction-recover")
    recover.add_argument("--owner-id", required=True)
    recover.add_argument("--limit", type=int, default=100)
    recover.add_argument("--as-of", type=float, required=True)
    recover.add_argument("--confirm-report-digest", required=True)
    recover.add_argument("--confirm-job-id", action="append", required=True)
    recover.add_argument("--actor-id", default="operator")
    recover.add_argument("--reason", default="reconcile-interrupted-compaction")

    recovery_status = commands.add_parser("compaction-recovery-status")
    recovery_status.add_argument("recovery_id")

    recovery_list = commands.add_parser("compaction-recovery-list")
    recovery_list.add_argument("--owner-id", required=True)
    recovery_list.add_argument("--phase")
    recovery_list.add_argument("--limit", type=int, default=100)
    return parser


def _common(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "owner_id": args.owner_id,
        "journal": get_evidence_graph_job_journal(),
        "generations": get_generation_store(),
        "graphs": get_evidence_graph_store(),
        "limit": args.limit,
    }


def _compaction_and_recovery_stores():
    compactions = get_evidence_graph_compaction_store()
    recovery = EvidenceGraphCompactionRecoveryJournal(compactions.path)
    return compactions, recovery


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "audit":
            report = audit_evidence_graph_jobs(**_common(args), now=args.as_of)
            payload = asdict(report)
            payload["report_digest"] = report.report_digest
            payload["mutation_performed"] = False
            payload["contains_graph_text"] = False
            _print(payload)
            return 0
        if args.command == "retention-plan":
            plan = plan_evidence_graph_job_retention(
                **_common(args),
                min_age_seconds=args.min_age_seconds,
                now=args.as_of,
            )
            payload = asdict(plan)
            payload["plan_digest"] = plan.plan_digest
            payload["mutation_performed"] = False
            payload["deletion_authorized"] = False
            payload["job_journal_rows_retained"] = True
            _print(payload)
            return 0
        if args.command == "retention-apply":
            common = _common(args)
            plan = plan_evidence_graph_job_retention(
                **common,
                min_age_seconds=args.min_age_seconds,
                now=args.as_of,
            )
            result = compact_evidence_graph_retention_plan(
                plan=plan,
                journal=common["journal"],
                generations=common["generations"],
                graphs=common["graphs"],
                compactions=get_evidence_graph_compaction_store(),
                confirm_plan_digest=args.confirm_plan_digest,
                confirm_job_ids=args.confirm_job_id,
            )
            payload = asdict(result)
            payload["result_digest"] = result.result_digest
            payload["authoritative_mutation_performed"] = False
            payload["semantic_inference_performed"] = False
            payload["job_journal_rows_retained"] = True
            payload["graph_payload_mutation_performed"] = bool(
                result.deleted_graph_generation_job_ids
            )
            _print(payload)
            return 0
        if args.command == "compaction-status":
            value = get_evidence_graph_compaction_store().get(args.job_id)
            if value is None:
                _print({"error": "not_found"}, stream=sys.stderr)
                return 1
            _print(_compaction_summary(value))
            return 0
        if args.command == "compaction-list":
            values = get_evidence_graph_compaction_store().list(
                owner_id=args.owner_id,
                phase=args.phase,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "items": [_compaction_summary(value) for value in values],
                    "authoritative_mutation_performed": False,
                    "contains_graph_text": False,
                }
            )
            return 0
        if args.command == "compaction-recovery-status":
            _, recovery = _compaction_and_recovery_stores()
            value = recovery.get(args.recovery_id)
            if value is None:
                _print({"error": "not_found"}, stream=sys.stderr)
                return 1
            _print(_recovery_summary(value))
            return 0
        if args.command == "compaction-recovery-list":
            _, recovery = _compaction_and_recovery_stores()
            values = recovery.list(
                owner_id=args.owner_id,
                phase=args.phase,
                limit=args.limit,
            )
            _print(
                {
                    "count": len(values),
                    "items": [_recovery_summary(value) for value in values],
                    "authoritative_mutation_performed": False,
                    "graph_payload_mutation_performed": False,
                    "contains_graph_text": False,
                }
            )
            return 0
        if args.command in {"compaction-reconcile", "compaction-recover"}:
            compactions = get_evidence_graph_compaction_store()
            stores = {
                "compactions": compactions,
                "journal": get_evidence_graph_job_journal(),
                "generations": get_generation_store(),
                "graphs": get_evidence_graph_store(),
            }
            report = reconcile_evidence_graph_compactions(
                owner_id=args.owner_id,
                limit=args.limit,
                now=args.as_of,
                **stores,
            )
            if args.command == "compaction-reconcile":
                payload = asdict(report)
                payload["report_digest"] = report.report_digest
                payload["healthy"] = report.healthy
                payload["recoverable_job_ids"] = report.recoverable_job_ids
                payload["conflict_job_ids"] = report.conflict_job_ids
                payload["status_counts"] = dict(report.status_counts)
                payload["mutation_performed"] = False
                payload["contains_graph_text"] = False
                _print(payload)
                return 0
            receipt = recover_reconciled_compaction_receipts_durable(
                report=report,
                recovery_journal=EvidenceGraphCompactionRecoveryJournal(
                    compactions.path
                ),
                confirm_report_digest=args.confirm_report_digest,
                confirm_job_ids=args.confirm_job_id,
                actor_id=args.actor_id,
                reason=args.reason,
                now=args.as_of,
                **stores,
            )
            payload = _recovery_summary(receipt)
            payload["receipt_mutation_performed"] = bool(receipt.completed_job_ids)
            payload["job_journal_rows_retained"] = True
            _print(payload)
            return 0
        raise ValueError("unsupported graph operations command.")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())