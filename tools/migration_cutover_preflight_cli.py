"""Plan/status-only CLI for migration cutover preflights."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.migration_cutover_preflight import build_cutover_preflight
from tools.migration_cutover_preflight_runtime import (
    get_migration_cutover_preflight_store,
)
from tools.migration_promotion_runtime import get_migration_promotion_store
from tools.migration_runtime import get_migration_journal
from tools.migration_shadow_runtime import get_migration_shadow_store
from tools.migration_types import digest, exact_integer, identifier


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _capture_snapshot(task: Any) -> Any:
    from tools.authoritative_document_index import capture_authoritative_document
    from tools.rag import get_rag_layer

    return capture_authoritative_document(
        owner_id=task.owner_id,
        doc_id=task.doc_id,
        rag=get_rag_layer(),
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "task_id": value.task_id,
        "owner_id": value.owner_id,
        "doc_id": value.doc_id,
        "source_sequence": value.source_sequence,
        "source_profile_fingerprint": value.source_profile_fingerprint,
        "target_profile_fingerprint": value.target_profile_fingerprint,
        "validation_digest": value.validation_digest,
        "promotion_report_digest": value.promotion_report_digest,
        "benchmark_fingerprint": value.benchmark_fingerprint,
        "rollback_identity_digest": value.rollback_identity_digest,
        "target_artifact_digest": value.target_artifact_digest,
        "source_vector_rows": value.source_vector_rows,
        "source_sparse_generation": value.source_sparse_generation,
        "source_sparse_fields": value.source_sparse_fields,
        "target_vector_rows": value.target_vector_rows,
        "target_sparse_rows": value.target_sparse_rows,
        "preflight_digest": value.preflight_digest,
        "created_at": value.created_at,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_cutover_preflight_cli",
        description=(
            "Capture non-mutating rollback/target identities for an eligible paired "
            "migration report. This CLI has no approve, execute or rollback action."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Create one immutable cutover preflight.")
    plan.add_argument("task_id")
    status = commands.add_parser("status", help="Read a current or historical preflight.")
    status.add_argument("task_id")
    status.add_argument("--preflight-digest")
    history = commands.add_parser("history", help="List bounded preflight history.")
    history.add_argument("task_id")
    history.add_argument("--limit", type=int, default=100)
    remove = commands.add_parser(
        "remove-task",
        help="Remove preflight records only for failed or cancelled tasks.",
    )
    remove.add_argument("task_id")
    remove.add_argument("--confirm-task-id", required=True)
    return parser


def _plan(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    task = get_migration_journal().get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    shadow = get_migration_shadow_store().validate(task_id)
    promotion = get_migration_promotion_store().read(task_id)
    snapshot = _capture_snapshot(task)
    value = build_cutover_preflight(
        task=task,
        shadow_manifest=shadow,
        promotion_report=promotion,
        authoritative_snapshot=snapshot,
    )
    persisted = get_migration_cutover_preflight_store().write(value)
    _print(_summary(persisted))
    return 0


def _status(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    selected = (
        digest(args.preflight_digest, "preflight_digest")
        if args.preflight_digest is not None
        else None
    )
    try:
        value = get_migration_cutover_preflight_store().read(
            task_id, preflight_digest=selected
        )
    except FileNotFoundError:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    _print(_summary(value))
    return 0


def _history(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    limit = exact_integer(args.limit, "limit", 1, 10_000)
    try:
        values = get_migration_cutover_preflight_store().history(task_id, limit=limit)
    except FileNotFoundError:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    _print(
        {
            "task_id": task_id,
            "count": len(values),
            "preflights": [_summary(value) for value in values],
        }
    )
    return 0


def _remove(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    confirmation = identifier(args.confirm_task_id, "confirm_task_id", 64)
    if confirmation != task_id:
        raise ValueError("confirmation must exactly match task_id.")
    task = get_migration_journal().get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    if task.state not in {"failed", "cancelled"}:
        raise ValueError("preflights may be removed only for failed or cancelled tasks.")
    removed = get_migration_cutover_preflight_store().remove_task(task_id)
    _print({"task_id": task_id, "removed": removed})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "plan":
            return _plan(args)
        if args.command == "status":
            return _status(args)
        if args.command == "history":
            return _history(args)
        if args.command == "remove-task":
            return _remove(args)
        raise ValueError("unsupported cutover preflight command.")
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
