#!/usr/bin/env python3
"""Closed-world production scope for training/model surface inventory.

The universal source scanner intentionally uses broad lexical signatures to avoid
missing obscure training implementations.  Test fixtures and the controller's
own implementation contain those same signatures by design, however, and are
not repository training surfaces.  This layer removes only structurally proven
infrastructure paths from the production inventory and records every removal in
an explicit audit ledger.  No repository-specific model/trainer path is hidden.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import universal_training_controller_current as current

INVENTORY_SCOPE_SCHEMA = 1


def _exclusion_reason(rel: str) -> str | None:
    normalized = rel.replace("\\", "/").strip("/")
    path = Path(normalized)
    parts = {part.casefold() for part in path.parts[:-1]}
    if "tests" in parts or "test" in parts:
        return "test_source"
    if normalized == "run_all_training.py":
        return "launcher_infrastructure"
    if normalized == "tools/account_wide_training_control_audit.py":
        return "account_audit_infrastructure"
    if normalized.startswith("tools/universal_training_controller") and normalized.endswith(".py"):
        return "training_controller_infrastructure"
    return None


def _filter_inventory(report: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(report)
    excluded: dict[str, str] = {}

    def keep(path: str) -> bool:
        reason = _exclusion_reason(str(path))
        if reason is not None:
            excluded[str(path).replace("\\", "/")] = reason
            return False
        return True

    result["training_files"] = [
        row for row in report.get("training_files", [])
        if isinstance(row, dict) and keep(str(row.get("path") or ""))
    ]
    for key in (
        "executable_training_candidates",
        "model_surfaces",
        "training_logic_surfaces",
    ):
        result[key] = sorted({
            str(path).replace("\\", "/")
            for path in report.get(key, []) or []
            if keep(str(path))
        })

    result["inventory_scope_schema"] = INVENTORY_SCOPE_SCHEMA
    result["excluded_nontraining_sources"] = [
        {"path": path, "reason": excluded[path]}
        for path in sorted(excluded)
    ]
    result["excluded_nontraining_source_count"] = len(excluded)
    return result


def install() -> None:
    original = current._training_inventory

    def training_inventory(root):
        return _filter_inventory(original(root))

    current._training_inventory = training_inventory


__all__ = ["INVENTORY_SCOPE_SCHEMA", "_exclusion_reason", "_filter_inventory", "install"]
