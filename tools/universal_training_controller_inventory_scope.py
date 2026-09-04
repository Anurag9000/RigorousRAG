#!/usr/bin/env python3
"""Closed-world production scope for training/model surface inventory.

The universal source scanner intentionally uses broad lexical signatures to avoid
missing obscure training implementations.  Test fixtures and the controller's
own implementation contain those same signatures by design, however, and are
not repository training surfaces.  This layer removes only structurally proven
infrastructure paths from the production inventory and records every removal in
an explicit audit ledger.

Repositories may additionally classify *model-only* production files as frozen
inference/materialization surfaces through
``training_control/non_training_surface_accounting.json``.  That contract is
fail-closed: entries use exact paths (no globs), a closed category vocabulary,
non-trivial reasons, must resolve to existing model surfaces, and are rejected if
the scanner sees executable training or training logic in the file.  A malformed
or stale declaration forces the overall coverage audit to fail.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current

INVENTORY_SCOPE_SCHEMA = 2
ACCOUNTING_SCHEMA = "training-control/non-training-surface-accounting/v1"
ACCOUNTING_PATH = Path("training_control/non_training_surface_accounting.json")
ALLOWED_NONTRAINING_CATEGORIES = frozenset(
    {
        "frozen_model_backend",
        "inference_adapter",
        "post_training_scorer",
        "runtime_feature_provider",
        "scientific_inference_adapter",
        "serving_provider",
        "supervision_materializer",
        "training_data_materializer",
    }
)
_GLOB_CHARS = frozenset("*?[]{}")


def _exclusion_reason(rel: str) -> str | None:
    normalized = rel.replace("\\", "/").strip("/")
    path = Path(normalized)
    parts = {part.casefold() for part in path.parts[:-1]}
    if "tests" in parts or "test" in parts:
        return "test_source"
    if normalized == "run_all_training.py":
        return "launcher_infrastructure"
    if normalized in {
        "tools/account_wide_training_control_audit.py",
        "tools/account_wide_training_control_audit_v2.py",
    }:
        return "account_audit_infrastructure"
    if normalized.startswith("tools/universal_training_controller") and normalized.endswith(".py"):
        return "training_controller_infrastructure"
    return None


def _normalize_exact_path(root: Path, value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "path must be a non-empty string"
    raw = value.strip().replace("\\", "/")
    if any(char in raw for char in _GLOB_CHARS):
        return None, f"glob/meta characters are forbidden in path {raw!r}"
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None, f"path must be repository-relative and non-traversing: {raw!r}"
    normalized = candidate.as_posix().lstrip("./")
    if not normalized or normalized == ".":
        return None, f"invalid repository path {raw!r}"
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, f"path escapes repository root: {raw!r}"
    if not resolved.is_file():
        return None, f"accounted source does not exist as a file: {normalized}"
    return normalized, None


def _load_nontraining_accounting(
    root: Path,
    *,
    model_surfaces: Sequence[str],
    training_logic_surfaces: Sequence[str],
    executable_training_candidates: Sequence[str],
) -> tuple[list[dict[str, str]], list[str]]:
    source = root / ACCOUNTING_PATH
    if not source.is_file():
        return [], []
    errors: list[str] = []
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"{ACCOUNTING_PATH.as_posix()} is not valid JSON: {exc}"]
    if not isinstance(value, Mapping):
        return [], [f"{ACCOUNTING_PATH.as_posix()} must contain a JSON object"]
    if value.get("schema") != ACCOUNTING_SCHEMA:
        errors.append(
            f"accounting schema must be {ACCOUNTING_SCHEMA!r}, got {value.get('schema')!r}"
        )
    raw_surfaces = value.get("surfaces")
    if not isinstance(raw_surfaces, list):
        return [], [*errors, "accounting surfaces must be a JSON array"]

    models = {str(path).replace("\\", "/") for path in model_surfaces}
    training_logic = {str(path).replace("\\", "/") for path in training_logic_surfaces}
    executable = {str(path).replace("\\", "/") for path in executable_training_candidates}
    seen: set[str] = set()
    accepted: list[dict[str, str]] = []
    for index, row in enumerate(raw_surfaces):
        prefix = f"surfaces[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        if set(row) != {"path", "category", "reason"}:
            errors.append(f"{prefix} must contain exactly path/category/reason")
            continue
        path, path_error = _normalize_exact_path(root, row.get("path"))
        if path_error:
            errors.append(f"{prefix}: {path_error}")
            continue
        assert path is not None
        category = row.get("category")
        if category not in ALLOWED_NONTRAINING_CATEGORIES:
            errors.append(
                f"{prefix}: category {category!r} is not in the closed non-training vocabulary"
            )
            continue
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 24:
            errors.append(f"{prefix}: reason must be a specific explanation of at least 24 characters")
            continue
        if path in seen:
            errors.append(f"{prefix}: duplicate accounted path {path}")
            continue
        seen.add(path)
        if path in training_logic or path in executable:
            errors.append(
                f"{prefix}: {path} contains executable/training logic and cannot be declared non-training"
            )
            continue
        if path not in models:
            errors.append(
                f"{prefix}: {path} is stale/not currently detected as a production model surface"
            )
            continue
        accepted.append(
            {"path": path, "category": str(category), "reason": reason.strip()}
        )
    return sorted(accepted, key=lambda row: row["path"]), errors


def _filter_inventory(root: Path, report: Dict[str, Any]) -> Dict[str, Any]:
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

    accounted, accounting_errors = _load_nontraining_accounting(
        root,
        model_surfaces=result.get("model_surfaces", []),
        training_logic_surfaces=result.get("training_logic_surfaces", []),
        executable_training_candidates=result.get("executable_training_candidates", []),
    )
    accounted_paths = {row["path"] for row in accounted}
    if not accounting_errors:
        result["model_surfaces"] = [
            path for path in result.get("model_surfaces", []) if path not in accounted_paths
        ]
        result["training_files"] = [
            row for row in result.get("training_files", [])
            if not (
                isinstance(row, dict)
                and str(row.get("path") or "").replace("\\", "/") in accounted_paths
                and not row.get("training_logic")
                and not row.get("executable")
            )
        ]

    result["inventory_scope_schema"] = INVENTORY_SCOPE_SCHEMA
    result["excluded_nontraining_sources"] = [
        {"path": path, "reason": excluded[path]}
        for path in sorted(excluded)
    ]
    result["excluded_nontraining_source_count"] = len(excluded)
    result["non_training_surface_accounting_schema"] = ACCOUNTING_SCHEMA
    result["non_training_surface_accounting"] = accounted
    result["non_training_surface_accounting_count"] = len(accounted)
    result["non_training_surface_accounting_errors"] = accounting_errors
    result["non_training_surface_accounting_pass"] = not accounting_errors
    return result


def install() -> None:
    original_inventory = current._training_inventory
    original_report = current._enhanced_coverage_report

    def training_inventory(root):
        return _filter_inventory(root, original_inventory(root))

    def coverage_report(root, profile, jobs):
        report = original_report(root, profile, jobs)
        inventory = report.get("inventory") or {}
        errors = (
            list(inventory.get("non_training_surface_accounting_errors", []))
            if isinstance(inventory, Mapping)
            else ["training inventory is missing from coverage report"]
        )
        report["non_training_surface_accounting_errors"] = errors
        report["non_training_surface_accounting_pass"] = not errors
        if errors:
            report["coverage_ok"] = False
        return report

    current._training_inventory = training_inventory
    current._enhanced_coverage_report = coverage_report


__all__ = [
    "ACCOUNTING_PATH",
    "ACCOUNTING_SCHEMA",
    "ALLOWED_NONTRAINING_CATEGORIES",
    "INVENTORY_SCOPE_SCHEMA",
    "_exclusion_reason",
    "_filter_inventory",
    "_load_nontraining_accounting",
    "install",
]
