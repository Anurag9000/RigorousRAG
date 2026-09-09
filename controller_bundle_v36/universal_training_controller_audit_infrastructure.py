#!/usr/bin/env python3
"""Align the legacy production inventory with semantic audit infrastructure scope.

The framework-oriented legacy scanner intentionally uses broad training tokens. The
standalone census and semantic scanner contain those tokens because they inspect training
code, but they are themselves audit implementation, not executable repository learners.
This layer removes only those two exact paths from the legacy inventory and records the
removal in the same exclusion ledger used by the production-scope layer.
"""
from __future__ import annotations

from typing import Any, Dict

import universal_training_controller_current as current

AUDIT_INFRASTRUCTURE_SCHEMA = 2
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
_INSTALL_MARKER = "__training_control_audit_infrastructure_scope_v2__"


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
    """Wrap the exact inventory hook used by ``inventory_scope.install()``.

    Installation is intentionally idempotent because the universal controller may be
    imported by diagnostics/tests more than once in a process.  Missing/renamed hooks
    fail with a targeted error instead of an opaque attribute failure before certificate
    creation.
    """
    original_inventory = getattr(current, "_training_inventory", None)
    if not callable(original_inventory):
        raise RuntimeError(
            "universal_training_controller_current._training_inventory is missing/non-callable; "
            "audit infrastructure scope cannot be installed safely"
        )
    if getattr(original_inventory, _INSTALL_MARKER, False):
        return

    def training_inventory(root):
        return _filter_report(original_inventory(root))

    setattr(training_inventory, _INSTALL_MARKER, True)
    current._training_inventory = training_inventory


__all__ = [
    "AUDIT_INFRASTRUCTURE_SCHEMA",
    "_EXACT_AUDIT_INFRASTRUCTURE",
    "_filter_report",
    "install",
]
