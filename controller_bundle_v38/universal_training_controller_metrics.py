#!/usr/bin/env python3
"""Persistent lifecycle/metrics manifests around the OPF-backed DAG executor.

This layer does not parse or alter model mathematics and does not schedule
resources.  It records the concrete repository orchestration plan before launch
and indexes metric-like artifacts after each invocation (including interrupted
runs), while the literal OPF_ADP scheduler remains responsible for execution.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller as base
import universal_training_controller_dag as dag

METRICS_MANIFEST_SCHEMA = 1
METRIC_NAME_RE = re.compile(
    r"(?:metric|metrics|history|score|scores|eval|evaluation|validation|test|loss|"
    r"accuracy|precision|recall|f1|auc|bleu|rouge|wer|cer|reward|return|summary|report)",
    re.I,
)
METRIC_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".npz", ".npy"}
MAX_HASH_BYTES = 8 * 1024 * 1024


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_atomic(path, (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _job_manifest(root: Path, profile: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "phase", "family", "device_capable", "depends_on", "is_training_job",
        "resume_strategy", "checkpoint_contract", "early_stopping",
        "early_stopping_applicable", "early_stopping_exception_reason",
        "task", "model", "datasets", "dataset", "losses", "deployment_role",
        "artifact_outputs", "artifact_dependencies", "completion_artifacts",
        "checkpoint_artifacts", "recipe_config", "repeat_index",
    )
    jobs = []
    for record in records:
        row: Dict[str, Any] = {
            "id": str(record.get("id") or ""),
            "command": [str(x) for x in (record.get("command") or [])],
        }
        for key in keys:
            if key in record:
                row[key] = record.get(key)
        jobs.append(row)
    lifecycle = profile.get("lifecycle_orchestration")
    return {
        "schema": METRICS_MANIFEST_SCHEMA,
        "repository": profile.get("repository") or root.name,
        "job_count": len(jobs),
        "lifecycle": lifecycle if isinstance(lifecycle, Mapping) else {},
        "jobs": jobs,
    }


def _metric_file_row(root: Path, path: Path) -> Dict[str, Any]:
    stat = path.stat()
    row: Dict[str, Any] = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= MAX_HASH_BYTES:
        try:
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        except Exception:
            pass
    return row


def _metrics_index(root: Path, forwarded: Sequence[str]) -> Dict[str, Any]:
    results_value = dag._arg_value(base._ensure_opf_cli(forwarded), "--results-dir", "Results/training_control")
    results_dir = Path(str(results_value))
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    search_roots = [root / ".training_control", results_dir]
    rows: list[Dict[str, Any]] = []
    seen: set[Path] = set()
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            if path.suffix.lower() not in METRIC_SUFFIXES:
                continue
            if not METRIC_NAME_RE.search(path.name):
                continue
            try:
                rows.append(_metric_file_row(root, path))
            except Exception:
                continue
    rows.sort(key=lambda x: str(x.get("path")))
    return {
        "schema": METRICS_MANIFEST_SCHEMA,
        "results_dir": results_dir.resolve().relative_to(root.resolve()).as_posix()
        if results_dir.resolve().is_relative_to(root.resolve()) else str(results_dir.resolve()),
        "metric_artifact_count": len(rows),
        "artifacts": rows,
    }


def _execute(root: Path, profile: Dict[str, Any], records: list[Dict[str, Any]], forwarded: Sequence[str]) -> int:
    control = root / ".training_control"
    _atomic_json(control / "orchestration_manifest.json", _job_manifest(root, profile, records))
    try:
        return _ORIGINAL_EXECUTE(root, profile, records, forwarded)
    finally:
        # Persist whatever metrics/history exists even for interrupted or failed
        # runs. OPF owns process failure/retry semantics; this is indexing only.
        try:
            payload = _metrics_index(root, forwarded)
            payload["repository"] = profile.get("repository") or root.name
            _atomic_json(control / "metrics_index.json", payload)
        except Exception as exc:
            _atomic_json(
                control / "metrics_index.json",
                {
                    "schema": METRICS_MANIFEST_SCHEMA,
                    "repository": profile.get("repository") or root.name,
                    "metric_artifact_count": 0,
                    "artifacts": [],
                    "index_error": str(exc),
                },
            )


def install() -> None:
    dag._execute = _execute


_ORIGINAL_EXECUTE = dag._execute

__all__ = ["METRICS_MANIFEST_SCHEMA", "install"]
