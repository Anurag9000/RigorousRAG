#!/usr/bin/env python3
"""Global dependency-safe job slicing above the literal OPF scheduler.

The original DAG executor rejected ``--job-start-index`` and ``--job-limit`` as
soon as dependencies existed.  Those knobs are part of the required OPF control
surface, so this layer restores them without allowing prerequisite bypass:

1. apply the requested slice to the *global compiled job order*;
2. compute the transitive prerequisite closure of those target jobs;
3. execute only that closure through the ordinary DAG executor;
4. remove the OPF slice flags before each DAG wave so the same slice is not
   applied independently inside every wave.

Deferred producer-only stages are intentionally unsliced because their artifacts
may be required merely to enumerate the final experiment universe; the slice is
applied to the expanded/final graph instead.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller as base
import universal_training_controller_dag as dag

DAG_SLICING_SCHEMA = 1
_ORIGINAL_EXECUTE = None


def _int_arg(argv: Sequence[str], flag: str, default: int) -> int:
    raw = dag._arg_value(argv, flag)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception as exc:
        raise SystemExit(f"{flag} must be an integer") from exc


def _producer_stage(forwarded: Sequence[str]) -> bool:
    normalized = base._ensure_opf_cli(forwarded)
    run_root = str(dag._arg_value(normalized, "--run-root", "") or "").replace("\\", "/")
    return "/deferred_stages/producers" in "/" + run_root.lstrip("/")


def _dependency_list(record: Mapping[str, Any]) -> list[str]:
    raw = record.get("depends_on", []) or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(value) for value in raw]


def _select(records: Sequence[Mapping[str, Any]], start: int, limit: int) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if start < 0:
        raise SystemExit("--job-start-index must be >= 0")
    if limit < 0:
        raise SystemExit("--job-limit must be >= 0")
    ordered = [dict(row) for row in records]
    targets = ordered[start:]
    if limit > 0:
        targets = targets[:limit]
    target_ids = [str(row.get("id")) for row in targets]
    if not targets:
        return [], {
            "schema": DAG_SLICING_SCHEMA,
            "start_index": start,
            "job_limit": limit,
            "target_job_ids": [],
            "prerequisite_job_ids": [],
            "selected_job_ids": [],
        }

    index = {str(row.get("id")): row for row in ordered}
    closure = set(target_ids)
    stack = list(target_ids)
    while stack:
        job_id = stack.pop()
        row = index.get(job_id)
        if row is None:
            raise SystemExit(f"selected job disappeared from compiled graph: {job_id}")
        for dep in _dependency_list(row):
            if dep not in index:
                raise SystemExit(f"selected job {job_id} depends on unknown job {dep}")
            if dep not in closure:
                closure.add(dep)
                stack.append(dep)

    selected = [dict(row) for row in ordered if str(row.get("id")) in closure]
    prerequisites = [str(row.get("id")) for row in selected if str(row.get("id")) not in set(target_ids)]
    return selected, {
        "schema": DAG_SLICING_SCHEMA,
        "start_index": start,
        "job_limit": limit,
        "target_job_ids": target_ids,
        "prerequisite_job_ids": prerequisites,
        "selected_job_ids": [str(row.get("id")) for row in selected],
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_atomic(path, (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _execute(root: Path, profile: Dict[str, Any], records: list[Dict[str, Any]], forwarded: Sequence[str]) -> int:
    assert _ORIGINAL_EXECUTE is not None
    if _producer_stage(forwarded):
        return _ORIGINAL_EXECUTE(root, profile, records, forwarded)

    has_start = dag._has_arg(forwarded, "--job-start-index")
    has_limit = dag._has_arg(forwarded, "--job-limit")
    if not has_start and not has_limit:
        return _ORIGINAL_EXECUTE(root, profile, records, forwarded)

    start = _int_arg(forwarded, "--job-start-index", 0)
    limit = _int_arg(forwarded, "--job-limit", 0)
    selected, state = _select(records, start, limit)
    state["repository"] = profile.get("repository") or root.name
    state["original_job_count"] = len(records)
    state["selected_job_count"] = len(selected)
    _atomic_json(root / ".training_control" / "dag_slice.json", state)
    if not selected:
        print("[training-control] global job slice selected zero jobs; nothing to schedule")
        return 0

    clean = dag._drop_arg(forwarded, "--job-start-index")
    clean = dag._drop_arg(clean, "--job-limit")
    return _ORIGINAL_EXECUTE(root, profile, selected, clean)


def install() -> None:
    global _ORIGINAL_EXECUTE
    if getattr(dag._execute, "_training_control_dag_slicing", False):
        return
    _ORIGINAL_EXECUTE = dag._execute
    _execute._training_control_dag_slicing = True  # type: ignore[attr-defined]
    dag._execute = _execute


__all__ = ["DAG_SLICING_SCHEMA", "_select", "install"]
