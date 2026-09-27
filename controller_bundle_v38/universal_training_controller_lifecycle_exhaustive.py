#!/usr/bin/env python3
"""Exhaustive lifecycle expansion while preserving authoritative training catalogs.

This layer does not implement or modify resource scheduling.  It wraps the
existing lifecycle expander so repositories with explicit model/training
catalogs still auto-enrol otherwise-unrepresented dataset/preprocess,
validation/evaluation, testing/inference and metrics/aggregation executables.

For an authoritative catalog, auto-discovered *training* jobs are removed again:
the repository's explicit model/task/dataset training matrix remains the source
of truth.  The surrounding lifecycle is nevertheless exhaustive.  Dependency
edges are pruned/rebuilt after filtering, then every ready job is still executed
by the literal pinned OPF_ADP scheduler.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

import universal_training_controller_current as current
import universal_training_controller_lifecycle as lifecycle

EXHAUSTIVE_LIFECYCLE_SCHEMA = 1
_ORIGINAL = None
_MISSING = object()


def _authoritative(profile: Mapping[str, Any]) -> bool:
    return bool(profile.get("jobs")) or isinstance(profile.get("job_catalog"), Mapping)


def _records(root, profile: Dict[str, Any]):
    authoritative = _authoritative(profile)
    previous = profile.get("auto_lifecycle_discovery", _MISSING)
    profile["auto_lifecycle_discovery"] = True
    try:
        assert _ORIGINAL is not None
        records = [dict(row) for row in _ORIGINAL(root, profile)]
    finally:
        if previous is _MISSING:
            profile.pop("auto_lifecycle_discovery", None)
        else:
            profile["auto_lifecycle_discovery"] = previous

    removed_ids: set[str] = set()
    if authoritative:
        removed_ids = {
            str(row.get("id"))
            for row in records
            if row.get("lifecycle_discovered") is True
            and lifecycle._normalized_phase(row) == "training"
        }
        if removed_ids:
            filtered = []
            for row in records:
                if str(row.get("id")) in removed_ids:
                    continue
                deps = row.get("depends_on", []) or []
                if isinstance(deps, str):
                    deps = [deps]
                row["depends_on"] = [str(dep) for dep in deps if str(dep) not in removed_ids]
                filtered.append(row)
            records = filtered
            lifecycle._add_dependencies(records)

    phase_names = ("setup", "preprocess", "training", "validation", "testing", "metrics")
    phase_counts = {
        phase: sum(1 for row in records if lifecycle._normalized_phase(row) == phase)
        for phase in phase_names
    }
    profile["lifecycle_orchestration"] = {
        "schema": lifecycle.LIFECYCLE_SCHEMA,
        "enabled": True,
        "authoritative_model_catalog": authoritative,
        "auto_discovery": True,
        "authoritative_training_preserved": authoritative,
        "removed_discovered_training_jobs": sorted(removed_ids),
        "job_count": len(records),
        "phase_counts": phase_counts,
    }
    profile["exhaustive_lifecycle_orchestration"] = {
        "schema": EXHAUSTIVE_LIFECYCLE_SCHEMA,
        "enabled": True,
        "training_catalog_policy": "authoritative" if authoritative else "discovered",
        "surrounding_lifecycle_auto_enrolled": True,
        "job_count": len(records),
        "phase_counts": phase_counts,
    }
    return records


def install() -> None:
    global _ORIGINAL
    if getattr(current._ORIGINAL_JOB_RECORDS, "_training_control_exhaustive_lifecycle", False):
        return
    _ORIGINAL = current._ORIGINAL_JOB_RECORDS
    _records._training_control_exhaustive_lifecycle = True  # type: ignore[attr-defined]
    current._ORIGINAL_JOB_RECORDS = _records


__all__ = ["EXHAUSTIVE_LIFECYCLE_SCHEMA", "install"]
