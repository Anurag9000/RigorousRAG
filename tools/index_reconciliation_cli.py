"""Bounded JSON CLI for authoritative index reconciliation and narrow repairs."""

from __future__ import annotations

import argparse
import json
import operator
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.index_reconciliation import (
    apply_deleted_residue_repairs,
    plan_repairs,
)
from tools.rag import get_rag_layer
from tools.security import normalize_owner_id
from tools.sparse_runtime import get_authoritative_index_coordinator

_MAX_REPAIR_ROWS = 1_000


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        parsed = int(operator.index(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconcile-indexes",
        description=(
            "Inspect vector, sparse, and durable generation alignment without "
            "exposing source paths or document contents."
        ),
    )
    parser.add_argument("command", choices=("scan", "plan", "repair-deleted-residue"))
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--maximum", type=int, default=100)
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--pretty", action="store_true")
    return parser


def _payload(command: str, owner_id: str, maximum: int, confirmation: str) -> dict[str, Any]:
    owner = normalize_owner_id(owner_id)
    limit = _integer(maximum, "maximum", 1, _MAX_REPAIR_ROWS)
    rag = get_rag_layer()
    coordinator = get_authoritative_index_coordinator(rag=rag)
    report = coordinator.reconcile_owner(owner_id=owner)
    if command == "scan":
        return {"command": command, "report": report.as_dict()}
    actions = plan_repairs(report)
    if command == "plan":
        return {
            "command": command,
            "owner_id": owner,
            "actions": [asdict(action) for action in actions[:limit]],
            "truncated": len(actions) > limit,
        }
    repaired = apply_deleted_residue_repairs(
        coordinator,
        report,
        confirmation=confirmation,
        maximum=limit,
    )
    after = coordinator.reconcile_owner(owner_id=owner)
    return {
        "command": command,
        "owner_id": owner,
        "repaired": repaired,
        "remaining": after.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        payload = _payload(
            args.command,
            args.owner_id,
            args.maximum,
            args.confirmation,
        )
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2 if args.pretty else None,
                allow_nan=False,
            )
        )
        return 0
    except SystemExit:
        raise
    except (ValueError, RuntimeError):
        print("Index reconciliation could not be completed safely.", file=sys.stderr)
        return 1
    except Exception:
        print("Index reconciliation is unavailable.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
