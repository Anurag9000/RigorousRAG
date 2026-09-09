#!/usr/bin/env python3
"""Universal controller v34: fail-closed retained-trainable-source closure.

The earlier controller generations deliberately allowed ``ignore_entrypoints``
and ``dynamic_registry_covers`` to *account for* source that was not reachable
from any central job.  That is useful for tooling and for a production-only
subset, but it is weaker than an exhaustive repository training controller: a
real trainer/model can be labelled manual/reference/research and disappear from
the central workload while the ordinary accounting report still passes.

v34 separates those concepts:

* ignore/dynamic-cover patterns may still suppress duplicate/tooling discovery;
* they may NOT waive a retained source file that contains real training logic or
  a trainable model surface and is otherwise unreachable from the central graph;
* executable trainers must be in the executed-source closure of a central job;
* non-executable model/training helpers must be in the reachable import/execute
  closure of a central job;
* exclusions are narrow, explicit and reasoned.  They exist for genuinely
  user-removed scientific branches or source that is provably non-trainable,
  not as a replacement for cataloguing a retained experiment.

This module is source accounting only.  It does not create resource scheduling,
does not alter job concurrency/admission and does not modify OPF_ADP.  Every
ready job continues to execute through the exact byte-pinned OPF_ADP scheduler.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller_current as current

RETAINED_TRAINING_CLOSURE_SCHEMA = 1

# These are the only semantic exclusion classes accepted by this layer.  Tests,
# generated/build/vendor trees are already outside current._iter_sources().
_ALLOWED_EXCLUSION_CLASSES = {
    "user_removed",          # explicit product/project instruction to remove it
    "non_trainable",         # analytic/runtime/tooling source misdetected by scan
    "external_vendor",       # vendored external implementation, not repo workload
    "generated_source",      # generated source kept in-tree, authoritative input elsewhere
}


def _patterns(profile: Mapping[str, Any], key: str) -> list[str]:
    raw = profile.get(key, []) or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [str(item).replace("\\", "/") for item in raw if str(item).strip()]


def _matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        candidate = str(pattern).replace("\\", "/")
        if normalized == candidate or fnmatch.fnmatch(normalized, candidate):
            return True
    return False


def _load_exclusions(profile: Mapping[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    """Return validated explicit exclusions and malformed-entry diagnostics."""
    raw = profile.get("retained_training_surface_exclusions", []) or []
    if not isinstance(raw, list):
        return [], ["retained_training_surface_exclusions must be a list"]
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            errors.append(f"exclusion[{index}] must be an object")
            continue
        pattern = str(item.get("path") or item.get("pattern") or "").strip().replace("\\", "/")
        reason = str(item.get("reason") or "").strip()
        category = str(item.get("category") or "").strip().lower()
        if not pattern:
            errors.append(f"exclusion[{index}] missing path/pattern")
            continue
        if category not in _ALLOWED_EXCLUSION_CLASSES:
            errors.append(
                f"exclusion[{index}] category={category!r} is invalid; "
                f"allowed={sorted(_ALLOWED_EXCLUSION_CLASSES)}"
            )
            continue
        if len(reason) < 12:
            errors.append(f"exclusion[{index}] requires a substantive reason")
            continue
        rows.append({"pattern": pattern, "category": category, "reason": reason})
    return rows, errors


def _exclusion_for(path: str, rows: Sequence[Mapping[str, str]]) -> dict[str, str] | None:
    for row in rows:
        if _matches(path, [row["pattern"]]):
            return dict(row)
    return None


def _inventory_map(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in inventory.get("training_files", []) or []:
        if isinstance(row, dict) and row.get("path"):
            result[str(row["path"]).replace("\\", "/")] = dict(row)
    return result


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(
        root: Path,
        profile: Dict[str, Any],
        jobs: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
        inventory = report.get("inventory")
        if not isinstance(inventory, dict):
            inventory = current._training_inventory(root)
        reach = current._reachability(root, jobs)
        executed = set(str(x) for x in (reach.get("executed_sources") or []))
        reachable = set(str(x) for x in (reach.get("reachable_sources") or []))
        by_path = _inventory_map(inventory)

        ignore_patterns = _patterns(profile, "ignore_entrypoints")
        dynamic_patterns = _patterns(profile, "dynamic_registry_covers")
        exclusions, malformed_exclusions = _load_exclusions(profile)

        hidden_by_ignore: list[str] = []
        hidden_by_dynamic: list[str] = []
        unreachable_executable: list[str] = []
        unreachable_logic: list[str] = []
        unreachable_models: list[str] = []
        excluded_rows: list[dict[str, str]] = []

        for path, row in sorted(by_path.items()):
            training_logic = bool(row.get("training_logic"))
            model_surface = bool(row.get("model_surface"))
            executable = bool(row.get("executable"))
            if not training_logic and not model_surface:
                continue

            # A central executable trainer must actually be on an execution edge;
            # an imported helper/model only needs ordinary reachability.
            is_reached = path in (executed if executable and training_logic else reachable)
            if is_reached:
                continue

            exclusion = _exclusion_for(path, exclusions)
            if exclusion is not None:
                excluded_rows.append({"path": path, **exclusion})
                continue

            if _matches(path, ignore_patterns):
                hidden_by_ignore.append(path)
            if _matches(path, dynamic_patterns):
                hidden_by_dynamic.append(path)
            if executable and training_logic:
                unreachable_executable.append(path)
            if training_logic:
                unreachable_logic.append(path)
            if model_surface:
                unreachable_models.append(path)

        require = bool(
            profile.get(
                "require_all_retained_trainable_source_reachability",
                profile.get("strict_coverage", True),
            )
        )
        blockers = sorted(
            set(unreachable_executable) | set(unreachable_logic) | set(unreachable_models)
        )
        ok = (not require or (not blockers and not malformed_exclusions))

        controls = dict(report.get("strict_controls") or {})
        controls["require_all_retained_trainable_source_reachability"] = require
        report.update({
            "retained_training_closure_schema": RETAINED_TRAINING_CLOSURE_SCHEMA,
            "retained_training_cartesian_products_invented": False,
            "retained_training_surface_exclusions": excluded_rows,
            "malformed_retained_training_surface_exclusions": malformed_exclusions,
            "trainable_sources_hidden_by_ignore_patterns": sorted(set(hidden_by_ignore)),
            "trainable_sources_hidden_by_dynamic_cover_patterns": sorted(set(hidden_by_dynamic)),
            "unreachable_retained_executable_trainers": sorted(set(unreachable_executable)),
            "unreachable_retained_training_logic_surfaces": sorted(set(unreachable_logic)),
            "unreachable_retained_model_surfaces": sorted(set(unreachable_models)),
            "unreachable_retained_trainable_sources": blockers,
            "strict_retained_training_closure_pass": ok,
            "strict_controls": controls,
        })
        report["coverage_ok"] = bool(report.get("coverage_ok", False)) and ok
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["RETAINED_TRAINING_CLOSURE_SCHEMA", "install"]
