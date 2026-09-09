#!/usr/bin/env python3
"""Fail-closed semantic learner accounting for the universal training controller.

The legacy controller inventory is deliberately framework-oriented.  The repository also
ships a deterministic multi-language semantic census that catches hand-written fit/train/
calibration code and non-Python learner APIs.  This layer makes that census part of the
same ``coverage_ok`` certificate used for resume, early-stopping, DAG, and model-surface
acceptance.

No profile ignore or glob can exempt a semantic learner here.  A production learner must
be reachable from at least one scheduled job.  If the semantic scanner is too broad, its
scanner logic/tests must be corrected instead of hiding the finding in a repository
profile.  Production parse errors also fail closed because an unparseable file cannot be
proved non-training.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current

SEMANTIC_CONTRACT_SCHEMA = 1
_TOOLS_DIR = Path(__file__).resolve().parent
_SCANNER_PATH = _TOOLS_DIR / "training_surface_semantic_scan.py"

# These are audit/control implementation files, not repository training surfaces.  We do
# not exclude the whole tools/ tree because real fitting/calibration implementations live
# there in several repositories.
_INFRASTRUCTURE_BASENAMES = {
    "run_all_training.py",
    "training_surface_census.py",
    "training_surface_semantic_scan.py",
    "account_wide_training_control_audit.py",
}
_INFRASTRUCTURE_PREFIXES = (
    "tools/universal_training_controller",
    ".github/workflows/",
)
_STRUCTURAL_TEST_PARTS = {"test", "tests", "testing", "__pycache__"}


def _load_scanner():
    module_name = "_training_control_semantic_scan_runtime"
    spec = importlib.util.spec_from_file_location(module_name, _SCANNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load semantic training scanner: {_SCANNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    scan = getattr(module, "scan", None)
    if not callable(scan):
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise RuntimeError("semantic training scanner does not expose callable scan(root)")
    return scan


def _normalized(rel: str) -> str:
    return str(rel).replace("\\", "/").lstrip("./")


def _structural_nonproduction(rel: str) -> bool:
    normalized = _normalized(rel)
    path = Path(normalized)
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & _STRUCTURAL_TEST_PARTS:
        return True
    if path.name in _INFRASTRUCTURE_BASENAMES:
        return True
    return any(normalized.startswith(prefix) for prefix in _INFRASTRUCTURE_PREFIXES)


def _parse_error_path(message: str) -> str:
    # Scanner errors are emitted as ``relative/path.py:<line>: ...`` or
    # ``relative/path.ipynb: ...``.  Paths in this controller are repository-relative
    # POSIX paths; a colon is therefore a safe delimiter for supported source trees.
    return _normalized(str(message).split(":", 1)[0])


def _production_semantic_inventory(root: Path, jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scan = _load_scanner()
    raw = scan(root)
    if not isinstance(raw, Mapping):
        raise RuntimeError("semantic training scanner returned a non-mapping report")

    learner_files = sorted(
        {
            _normalized(path)
            for path in (raw.get("learner_files") or [])
            if not _structural_nonproduction(str(path))
        }
    )
    parse_errors = sorted(
        {
            str(error)
            for error in (raw.get("parse_errors") or [])
            if not _structural_nonproduction(_parse_error_path(str(error)))
        }
    )

    reachability = current._reachability(root, jobs)
    reachable = {
        _normalized(path) for path in (reachability.get("reachable_sources") or [])
    }
    accounted = sorted(path for path in learner_files if path in reachable)
    unaccounted = sorted(path for path in learner_files if path not in reachable)

    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), Mapping) else {}
    production_evidence = {
        path: list(evidence.get(path) or [])
        for path in learner_files
    }
    return {
        "schema": raw.get("schema"),
        "inventory_sha256": raw.get("inventory_sha256"),
        "base_inventory_sha256": raw.get("base_inventory_sha256"),
        "scanned_by_language": dict(raw.get("scanned_by_language") or {}),
        "production_learner_files": learner_files,
        "production_learner_evidence": production_evidence,
        "accounted_learner_files": accounted,
        "unaccounted_learner_files": unaccounted,
        "production_parse_errors": parse_errors,
        "structural_exclusion_policy": {
            "test_directory_parts": sorted(_STRUCTURAL_TEST_PARTS),
            "infrastructure_basenames": sorted(_INFRASTRUCTURE_BASENAMES),
            "infrastructure_prefixes": list(_INFRASTRUCTURE_PREFIXES),
        },
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root, profile: Dict[str, Any], jobs):
        report = original_report(root, profile, jobs)
        semantic = _production_semantic_inventory(Path(root), jobs)
        unaccounted = list(semantic["unaccounted_learner_files"])
        parse_errors = list(semantic["production_parse_errors"])
        semantic_ok = not unaccounted and not parse_errors

        controls = dict(report.get("strict_controls") or {})
        controls["require_semantic_training_surface_accounting"] = True
        controls["require_zero_production_semantic_parse_errors"] = True

        report.update(
            {
                "semantic_training_contract_schema": SEMANTIC_CONTRACT_SCHEMA,
                "semantic_training_surface_schema": semantic.get("schema"),
                "semantic_training_surface_inventory_sha256": semantic.get("inventory_sha256"),
                "semantic_training_surface_base_inventory_sha256": semantic.get("base_inventory_sha256"),
                "semantic_training_scanned_by_language": semantic.get("scanned_by_language", {}),
                "semantic_learner_surfaces": semantic["production_learner_files"],
                "semantic_learner_evidence": semantic["production_learner_evidence"],
                "accounted_semantic_learner_surfaces": semantic["accounted_learner_files"],
                "unaccounted_semantic_learner_surfaces": unaccounted,
                "semantic_production_parse_errors": parse_errors,
                "semantic_structural_exclusion_policy": semantic["structural_exclusion_policy"],
                "strict_semantic_training_surface_pass": semantic_ok,
                "strict_controls": controls,
            }
        )
        report["coverage_ok"] = bool(report.get("coverage_ok", False)) and semantic_ok
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = [
    "SEMANTIC_CONTRACT_SCHEMA",
    "_parse_error_path",
    "_production_semantic_inventory",
    "_structural_nonproduction",
    "install",
]
