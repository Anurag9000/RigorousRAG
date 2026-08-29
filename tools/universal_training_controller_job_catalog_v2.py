#!/usr/bin/env python3
"""Correct repository-local job-catalog expansion at the vCurrent hook boundary.

``universal_training_controller_current`` captures ``base._job_records`` when it
is imported.  Later it installs an enhanced wrapper that delegates to that
captured callable.  Therefore a catalog layer that only replaces
``base._job_records`` can be bypassed.  This module replaces the captured
``current._ORIGINAL_JOB_RECORDS`` callable itself, ensuring generated jobs pass
through every existing metadata, reachability, exact-resume, DAG and coverage
audit before reaching the literal pinned OPF_ADP scheduler.

No scheduling/resource-admission logic lives here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import universal_training_controller_current as current


def _load_catalog(root: Path, descriptor: Dict[str, Any]) -> List[Dict[str, Any]]:
    relative = str(descriptor.get("path") or "").strip()
    if not relative:
        raise SystemExit("job_catalog.path is required")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except Exception as exc:
        raise SystemExit(f"job_catalog.path escapes repository root: {relative}") from exc
    if not path.is_file():
        raise SystemExit(f"job catalog does not exist: {path}")

    module_name = "_training_control_catalog_" + path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import job catalog {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    function_name = str(descriptor.get("function") or "iter_jobs")
    function = getattr(module, function_name, None)
    if not callable(function):
        raise SystemExit(f"job catalog {path} has no callable {function_name}()")
    args = descriptor.get("args", []) or []
    kwargs = descriptor.get("kwargs", {}) or {}
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise SystemExit("job_catalog.args must be a list and kwargs must be an object")
    generated = list(function(*args, **kwargs))
    for index, item in enumerate(generated):
        if not isinstance(item, dict):
            raise SystemExit(f"job catalog item {index} is not an object")
    return generated


def _catalog_aware_original(root: Path, profile: Dict[str, Any]):
    descriptor = profile.get("job_catalog")
    if isinstance(descriptor, dict):
        if profile.get("jobs"):
            raise SystemExit("Use either profile.jobs or profile.job_catalog, not both")
        generated = _load_catalog(root, descriptor)
        profile["jobs"] = generated
        profile["job_catalog_materialization"] = {
            "path": str(descriptor.get("path")),
            "function": str(descriptor.get("function") or "iter_jobs"),
            "generated_jobs": len(generated),
        }
    return _CAPTURED_ORIGINAL(root, profile)


def install() -> None:
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_catalog_v2", False):
        return
    _catalog_aware_original._training_control_catalog_v2 = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _catalog_aware_original


_CAPTURED_ORIGINAL = current._ORIGINAL_JOB_RECORDS
