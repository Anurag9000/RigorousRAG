#!/usr/bin/env python3
"""Dependency-aware executor above the literal pinned OPF_ADP scheduler.

The OPF scheduler itself is never forked or edited. Repository jobs are first
validated as a DAG. Each dependency-ready topological wave is then passed to an
ordinary invocation of the pinned ``opf_massive_suite_runner``. The OPF runner
therefore remains solely responsible for resource admission, pressure pause /
resume, retries, OOM fallback, process accounting, logging and concurrency.

A separate run root is used for every wave so OPF's manifest identity and
checkpoint/resume state remain stable across relaunches. A failed prerequisite
prevents dependent waves from launching.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import universal_training_controller as base
import universal_training_controller_current as current

DAG_EXECUTOR_SCHEMA = 1


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_atomic(path, (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode())


def _enforced_job_dag(jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    report = _ORIGINAL_JOB_DAG(jobs)
    report["runtime_dependency_enforced"] = True
    return report


def _install() -> None:
    current._job_dag = _enforced_job_dag
    current._install_enhancements()


def _dependency_waves(jobs: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    by_id = {str(job.get("id")): job for job in jobs}
    ids = list(by_id)
    deps: Dict[str, set[str]] = {}
    children: Dict[str, set[str]] = {job_id: set() for job_id in ids}
    for job_id, job in by_id.items():
        raw = job.get("depends_on", []) or []
        if isinstance(raw, str):
            raw = [raw]
        wanted = {str(item) for item in raw}
        unknown = sorted(item for item in wanted if item not in by_id or item == job_id)
        if unknown:
            raise SystemExit(f"Invalid dependency declaration for {job_id}: {unknown}")
        deps[job_id] = wanted
        for dep in wanted:
            children[dep].add(job_id)

    remaining = set(ids)
    completed: set[str] = set()
    waves: List[List[Dict[str, Any]]] = []
    while remaining:
        ready_ids = [job_id for job_id in ids if job_id in remaining and deps[job_id] <= completed]
        if not ready_ids:
            raise SystemExit(f"Training job DAG contains a cycle among: {sorted(remaining)}")
        waves.append([by_id[job_id] for job_id in ready_ids])
        completed.update(ready_ids)
        remaining.difference_update(ready_ids)
    return waves


def _arg_value(argv: Sequence[str], flag: str, default: str | None = None) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return str(argv[index + 1])
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def _has_arg(argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def _drop_arg(argv: Sequence[str], flag: str) -> List[str]:
    out: List[str] = []
    skip = False
    for arg in argv:
        if skip:
            skip = False
            continue
        if arg == flag:
            skip = True
            continue
        if arg.startswith(flag + "="):
            continue
        out.append(arg)
    return out


def _set_arg(argv: Sequence[str], flag: str, value: str) -> List[str]:
    return [*_drop_arg(argv, flag), flag, value]


def _wave_cli(forwarded: Sequence[str], wave_index: int) -> Tuple[List[str], Path, Path]:
    cli = base._ensure_opf_cli(forwarded)
    results_dir = Path(str(_arg_value(cli, "--results-dir", "Results/training_control")))
    requested_root = _arg_value(cli, "--run-root")
    run_root_base = Path(str(requested_root)) if requested_root else results_dir / "run_root"
    wave_root = run_root_base / "dag_waves" / f"wave_{wave_index:04d}"
    cli = _set_arg(cli, "--run-root", str(wave_root))
    return cli, results_dir, wave_root


def _wave_summary(wave_root: Path) -> Dict[str, Any]:
    path = wave_root / "final_report.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _global_state(
    *,
    root: Path,
    profile: Mapping[str, Any],
    waves: Sequence[Sequence[Dict[str, Any]]],
    completed_waves: Sequence[int],
    failed_wave: int | None,
    wave_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema": DAG_EXECUTOR_SCHEMA,
        "repository": profile.get("repository") or root.name,
        "opf_reference_commit": current.OPF_REFERENCE_COMMIT,
        "wave_count": len(waves),
        "waves": [[str(job.get("id")) for job in wave] for wave in waves],
        "completed_waves": list(completed_waves),
        "failed_wave": failed_wave,
        "wave_reports": [dict(item) for item in wave_reports],
    }


def _execute(root: Path, profile: Dict[str, Any], records: List[Dict[str, Any]], forwarded: Sequence[str]) -> int:
    waves = _dependency_waves(records)
    dag_path = root / ".training_control" / "job_dag.json"
    state_path = root / ".training_control" / "dag_state.json"
    dag_report = current._job_dag(records)
    dag_report["waves"] = [[str(job.get("id")) for job in wave] for wave in waves]
    _atomic_json(dag_path, dag_report)

    if len(waves) <= 1:
        # Preserve the original single-invocation behavior when there are no
        # dependency boundaries. This is important for exact OPF parity.
        cache = base._prepare_opf_runtime(root)
        opf = base._import_opf_scheduler(cache)
        opf.build_suite_jobs = base._builder_factory(root, profile, records, opf)
        saved_argv = list(sys.argv)
        try:
            sys.argv = [str(Path(__file__).resolve()), *base._ensure_opf_cli(forwarded)]
            opf.main()
        finally:
            sys.argv = saved_argv
        return 0

    if any(_has_arg(forwarded, flag) for flag in ("--job-start-index", "--job-limit")):
        raise SystemExit(
            "--job-start-index/--job-limit are intentionally disabled for dependency DAGs; "
            "slicing individual OPF waves could bypass prerequisites."
        )

    cache = base._prepare_opf_runtime(root)
    opf = base._import_opf_scheduler(cache)
    completed_waves: List[int] = []
    wave_reports: List[Dict[str, Any]] = []
    dry_run = _has_arg(forwarded, "--dry-run")

    for wave_index, wave_records in enumerate(waves):
        cli, results_dir, wave_root = _wave_cli(forwarded, wave_index)
        opf.build_suite_jobs = base._builder_factory(root, profile, wave_records, opf)
        saved_argv = list(sys.argv)
        try:
            sys.argv = [str(Path(__file__).resolve()), *cli]
            opf.main()
        finally:
            sys.argv = saved_argv

        if dry_run:
            report = {
                "wave": wave_index,
                "run_root": str(wave_root),
                "results_dir": str(results_dir),
                "job_ids": [str(job.get("id")) for job in wave_records],
                "dry_run": True,
            }
        else:
            summary = _wave_summary(wave_root)
            if not summary:
                raise SystemExit(f"OPF wave {wave_index} returned without {wave_root / 'final_report.json'}")
            report = {
                "wave": wave_index,
                "run_root": str(wave_root),
                "results_dir": str(results_dir),
                "job_ids": [str(job.get("id")) for job in wave_records],
                "completed_jobs": int(summary.get("completed_jobs") or 0),
                "failed_jobs": int(summary.get("failed_jobs") or 0),
                "retried_jobs": int(summary.get("retried_jobs") or 0),
                "total_jobs": int(summary.get("total_jobs") or len(wave_records)),
            }
        wave_reports.append(report)

        failed = 0 if dry_run else int(report.get("failed_jobs") or 0)
        if failed:
            _atomic_json(
                state_path,
                _global_state(
                    root=root, profile=profile, waves=waves, completed_waves=completed_waves,
                    failed_wave=wave_index, wave_reports=wave_reports,
                ),
            )
            raise SystemExit(
                f"Dependency wave {wave_index} has {failed} failed job(s); downstream jobs were not launched. "
                f"Resume the same central runner after fixing the failing trainer."
            )
        completed_waves.append(wave_index)
        _atomic_json(
            state_path,
            _global_state(
                root=root, profile=profile, waves=waves, completed_waves=completed_waves,
                failed_wave=None, wave_reports=wave_reports,
            ),
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _install()
    root = base._repo_root()
    profile = base._load_profile()
    flags, forwarded = base._custom_flags(list(sys.argv[1:] if argv is None else argv))
    if flags["skip_setup"]:
        os.environ["TRAINING_CONTROL_SKIP_SETUP"] = "1"
    records = base._job_records(root, profile)
    report = base._coverage_report(root, profile, records)
    report_path = root / ".training_control" / "coverage_report.json"
    _atomic_json(report_path, report)
    if flags["audit"] or flags["list_jobs"]:
        print(json.dumps(report, indent=2, sort_keys=True))
    if flags["audit"]:
        return 0 if report["coverage_ok"] else 2
    strict = bool(profile.get("strict_coverage", True)) and not flags["allow_uncovered"]
    if strict and not report["coverage_ok"]:
        raise SystemExit(
            f"Training coverage audit failed. See {report_path}. "
            "Use --allow-uncovered-training only for diagnosis."
        )
    if not records:
        print("[training-control] no trainable jobs declared/discovered; nothing to schedule")
        return 0
    base._run_setup(root, profile)
    return _execute(root, profile, records, forwarded)


_ORIGINAL_JOB_DAG = current._job_dag

if __name__ == "__main__":
    raise SystemExit(main())
