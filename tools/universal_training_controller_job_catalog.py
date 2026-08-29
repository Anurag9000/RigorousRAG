#!/usr/bin/env python3
"""Repository-local explicit job-catalog expansion.

Large projects can declare a small ``job_catalog`` descriptor in their profile:

    {"path": "scripts/training_catalog_opf.py",
     "function": "iter_jobs",
     "args": ["exhaustive"]}

The catalog is expanded before the normal universal-controller normalization and
strict audit.  The resulting jobs are ordinary records and are scheduled by the
literal pinned OPF_ADP runner exactly as if they had been embedded in profile
JSON.  This module contains no scheduling logic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import universal_training_controller as base


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

    name = "_training_control_catalog_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import job catalog {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def _job_records(root: Path, profile: Dict[str, Any]):
    descriptor = profile.get("job_catalog")
    if not isinstance(descriptor, dict):
        return _ORIGINAL_JOB_RECORDS(root, profile)
    if profile.get("jobs"):
        raise SystemExit("Use either profile.jobs or profile.job_catalog, not both")
    generated = _load_catalog(root, descriptor)
    # Mutate the live profile intentionally: downstream metadata, DAG and strict
    # resume/early-stop audits must inspect the full declared records.
    profile["jobs"] = generated
    records = _ORIGINAL_JOB_RECORDS(root, profile)
    profile["job_catalog_materialization"] = {
        "path": str(descriptor.get("path")),
        "function": str(descriptor.get("function") or "iter_jobs"),
        "generated_jobs": len(generated),
    }
    return records


def install() -> None:
    base._job_records = _job_records


_ORIGINAL_JOB_RECORDS = base._job_records
