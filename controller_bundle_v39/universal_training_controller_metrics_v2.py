#!/usr/bin/env python3
"""Job-attributed metrics ledger layered on the existing metrics index.

The v1 metrics layer inventories metric-like artifacts globally.  This extension
maps artifacts back to concrete central-runner jobs whenever the path is inside
that job's OPF result directory or matches a declared artifact path.  Unassigned
artifacts remain visible rather than being discarded.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller as base
import universal_training_controller_dag as dag

METRICS_LEDGER_SCHEMA = 1
_ORIGINAL_EXECUTE = None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_atomic(path, (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _declared_artifact_patterns(record: Mapping[str, Any]) -> list[str]:
    patterns: list[str] = []
    for key in ("artifact_outputs", "completion_artifacts", "checkpoint_artifacts", "metrics_artifacts", "metric_artifacts"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            patterns.append(value.replace("\\", "/"))
        elif isinstance(value, (list, tuple, set)):
            patterns.extend(str(item).replace("\\", "/") for item in value)
        elif isinstance(value, Mapping):
            patterns.extend(str(item).replace("\\", "/") for item in value.values() if isinstance(item, str))
    return patterns


def _identity(record: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "phase", "family", "model", "models", "dataset", "datasets", "task", "tasks",
        "domain", "domains", "environment", "method", "algorithm", "recipe_config",
        "repeat_index", "deployment_role",
    )
    return {key: record.get(key) for key in keys if key in record}


def _load_index(root: Path) -> Dict[str, Any]:
    path = root / ".training_control" / "metrics_index.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _assign(records: Sequence[Mapping[str, Any]], artifacts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    job_rows: Dict[str, Dict[str, Any]] = {}
    slug_to_id: Dict[str, str] = {}
    for record in records:
        job_id = str(record.get("id") or "")
        slug = base._slug(job_id)
        slug_to_id[slug] = job_id
        job_rows[job_id] = {
            "job_id": job_id,
            "identity": _identity(record),
            "artifact_patterns": _declared_artifact_patterns(record),
            "metric_artifacts": [],
        }

    unassigned: list[Mapping[str, Any]] = []
    for artifact in artifacts:
        path = str(artifact.get("path") or "").replace("\\", "/")
        owners: set[str] = set()
        for slug, job_id in slug_to_id.items():
            if f"/generic_jobs/{slug}/" in "/" + path.lstrip("/"):
                owners.add(job_id)
        for job_id, row in job_rows.items():
            for pattern in row["artifact_patterns"]:
                normalized = str(pattern).lstrip("./")
                if path == normalized or fnmatch.fnmatch(path, normalized) or fnmatch.fnmatch(path, "*/" + normalized):
                    owners.add(job_id)
        if not owners:
            unassigned.append(dict(artifact))
            continue
        for job_id in sorted(owners):
            job_rows[job_id]["metric_artifacts"].append(dict(artifact))

    return {
        "schema": METRICS_LEDGER_SCHEMA,
        "jobs": [job_rows[job_id] for job_id in sorted(job_rows)],
        "unassigned_metric_artifacts": unassigned,
        "assigned_metric_artifact_links": sum(len(row["metric_artifacts"]) for row in job_rows.values()),
        "jobs_with_metric_artifacts": sum(1 for row in job_rows.values() if row["metric_artifacts"]),
    }


def _execute(root: Path, profile: Dict[str, Any], records: list[Dict[str, Any]], forwarded: Sequence[str]) -> int:
    assert _ORIGINAL_EXECUTE is not None
    try:
        return _ORIGINAL_EXECUTE(root, profile, records, forwarded)
    finally:
        index = _load_index(root)
        artifacts = index.get("artifacts", []) if isinstance(index.get("artifacts"), list) else []
        ledger = _assign(records, artifacts)
        ledger["repository"] = profile.get("repository") or root.name
        ledger["global_metric_artifact_count"] = int(index.get("metric_artifact_count") or len(artifacts))
        _atomic_json(root / ".training_control" / "job_metrics_ledger.json", ledger)


def install() -> None:
    global _ORIGINAL_EXECUTE
    if getattr(dag._execute, "_training_control_metrics_v2", False):
        return
    _ORIGINAL_EXECUTE = dag._execute
    _execute._training_control_metrics_v2 = True  # type: ignore[attr-defined]
    dag._execute = _execute


__all__ = ["METRICS_LEDGER_SCHEMA", "install"]
