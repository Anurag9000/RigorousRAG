#!/usr/bin/env python3
"""Align the legacy production inventory with semantic audit infrastructure scope.

The framework-oriented legacy scanner intentionally uses broad training tokens.  The
standalone census and semantic scanner contain those tokens because they inspect training
code, but they are themselves audit implementation, not executable repository learners.
This layer removes only those two exact paths from the legacy inventory and records the
removal in the same exclusion ledger used by the production-scope layer.
"""
from __future__ import annotations

from typing import Any, Dict

import universal_training_controller_current as current

AUDIT_INFRASTRUCTURE_SCHEMA = 1
_EXACT_AUDIT_INFRASTRUCTURE = frozenset(
    {
        "tools/training_surface_census.py",
        "tools/training_surface_semantic_scan.py",
    }
)
_FILTER_KEYS = (
    "executable_training_candidates",
    "model_surfaces",
    "training_logic_surfaces",
    "quiet_training_logic_surfaces",
)


def _normalized(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _filter_report(report: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(report)
    removed: set[str] = set()

    rows = []
    for row in report.get("training_files", []) or []:
        if not isinstance(row, dict):
            rows.append(row)
            continue
        path = _normalized(row.get("path"))
        if path in _EXACT_AUDIT_INFRASTRUCTURE:
            removed.add(path)
            continue
        rows.append(row)
    result["training_files"] = rows

    for key in _FILTER_KEYS:
        kept = []
        for value in report.get(key, []) or []:
            path = _normalized(value)
            if path in _EXACT_AUDIT_INFRASTRUCTURE:
                removed.add(path)
                continue
            kept.append(path)
        result[key] = sorted(set(kept))

    ledger = [
        dict(row)
        for row in report.get("excluded_nontraining_sources", []) or []
        if isinstance(row, dict)
    ]
    known = {_normalized(row.get("path")) for row in ledger}
    for path in sorted(removed):
        if path not in known:
            ledger.append({"path": path, "reason": "training_audit_infrastructure"})
    result["excluded_nontraining_sources"] = sorted(
        ledger, key=lambda row: _normalized(row.get("path"))
    )
    result["excluded_nontraining_source_count"] = len(result["excluded_nontraining_sources"])
    result["audit_infrastructure_scope_schema"] = AUDIT_INFRASTRUCTURE_SCHEMA
    return result


def install() -> None:
    original_inventory = current._inventory

    def inventory(root):
        return _filter_report(original_inventory(root))

    current._inventory = inventory


__all__ = [
    "AUDIT_INFRASTRUCTURE_SCHEMA",
    "_EXACT_AUDIT_INFRASTRUCTURE",
    "_filter_report",
    "install",
]
