"""Operator CLI for non-mutating migration promotion reports."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.migration_promotion import (
    PromotionPolicy,
    evaluate_promotion,
    evidence_from_mapping,
    policy_from_mapping,
)
from tools.migration_promotion_runtime import get_migration_promotion_store
from tools.migration_runtime import get_migration_journal
from tools.migration_shadow_runtime import get_migration_shadow_store
from tools.migration_types import digest, exact_integer, identifier

_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_MAX_INPUT_BYTES = 2_000_000
_MAX_PATH = 4096


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _redirecting(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0)) & _REPARSE
    )


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError("input file must be a filesystem path.")
    rendered = os.fspath(value)
    if (
        not isinstance(rendered, str)
        or not rendered
        or len(rendered) > _MAX_PATH
        or any(ord(character) < 32 or ord(character) == 127 for character in rendered)
    ):
        raise ValueError("input file path is invalid.")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    absolute = Path(os.path.abspath(candidate))
    for part in (absolute, *absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ValueError("input file path could not be validated.") from exc
        if _redirecting(info):
            raise ValueError("input file path may not contain redirects.")
    return absolute


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_object(value: str | os.PathLike[str], label: str) -> Mapping[str, Any]:
    path = _path(value)
    before = path.lstat()
    if _redirecting(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file.")
    if before.st_size <= 0 or before.st_size > _MAX_INPUT_BYTES:
        raise ValueError(f"{label} exceeds the byte limit.")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError(f"{label} changed before reading.")
        payload = handle.read(_MAX_INPUT_BYTES + 1)
    if len(payload) > _MAX_INPUT_BYTES:
        raise ValueError(f"{label} exceeds the byte limit.")
    after = path.lstat()
    if (
        _redirecting(after)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_size != before.st_size
    ):
        raise ValueError(f"{label} changed during reading.")
    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{label} must contain strict JSON.") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must contain a JSON object.")
    return parsed


def _summary(report: Any) -> dict[str, Any]:
    return {
        "task_id": report.task_id,
        "decision": report.decision,
        "reason_codes": list(report.reason_codes),
        "report_digest": report.report_digest,
        "validation_digest": report.validation_digest,
        "benchmark_fingerprint": report.benchmark_fingerprint,
        "evidence_digest": report.evidence_digest,
        "policy_id": report.policy_id,
        "policy_digest": report.policy_digest,
        "quality_deltas": dict(report.quality_deltas),
        "resource_ratios": dict(report.resource_ratios),
        "evaluated_at": report.evaluated_at,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_promotion_cli",
        description=(
            "Evaluate and persist non-mutating migration promotion reports. "
            "This CLI cannot cut over live generations."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate aggregate benchmark evidence for one validated shadow.",
    )
    evaluate.add_argument("task_id")
    evaluate.add_argument("--evidence-file", required=True)
    evaluate.add_argument("--policy-file")

    status = commands.add_parser("status", help="Read one current or historical report.")
    status.add_argument("task_id")
    status.add_argument("--report-digest")

    history = commands.add_parser("history", help="List bounded report history.")
    history.add_argument("task_id")
    history.add_argument("--limit", type=int, default=100)

    remove = commands.add_parser(
        "remove-task",
        help="Remove reports only for failed or cancelled migration tasks.",
    )
    remove.add_argument("task_id")
    remove.add_argument("--confirm-task-id", required=True)
    return parser


def _evaluate(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    journal = get_migration_journal()
    task = journal.get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    manifest = get_migration_shadow_store().validate(task_id)
    evidence = evidence_from_mapping(
        _read_object(args.evidence_file, "promotion evidence")
    )
    policy = (
        policy_from_mapping(_read_object(args.policy_file, "promotion policy"))
        if args.policy_file
        else PromotionPolicy()
    )
    from tools.sparse_runtime import get_generation_store

    generation = get_generation_store().current(
        owner_id=task.owner_id,
        doc_id=task.doc_id,
    )
    report = evaluate_promotion(
        task=task,
        manifest=manifest,
        generation=generation,
        evidence=evidence,
        policy=policy,
    )
    persisted = get_migration_promotion_store().write(report)
    _print(_summary(persisted))
    return 0 if persisted.decision == "eligible" else 1


def _status(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    report_digest = (
        digest(args.report_digest, "report_digest")
        if args.report_digest is not None
        else None
    )
    try:
        report = get_migration_promotion_store().read(
            task_id,
            report_digest=report_digest,
        )
    except FileNotFoundError:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    _print(_summary(report))
    return 0


def _history(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    limit = exact_integer(args.limit, "limit", 1, 10_000)
    try:
        reports = get_migration_promotion_store().history(task_id, limit=limit)
    except FileNotFoundError:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    _print(
        {
            "task_id": task_id,
            "count": len(reports),
            "reports": [_summary(report) for report in reports],
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
        raise ValueError("reports may be removed only for failed or cancelled tasks.")
    removed = get_migration_promotion_store().remove_task(task_id)
    _print({"task_id": task_id, "removed": removed})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "status":
            return _status(args)
        if args.command == "history":
            return _history(args)
        if args.command == "remove-task":
            return _remove(args)
        raise ValueError("unsupported migration promotion command.")
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
