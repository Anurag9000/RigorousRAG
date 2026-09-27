#!/usr/bin/env python3
"""Fail-closed post-producer job expansion above the literal OPF scheduler.

Some scientifically valid experiment dimensions are not knowable until a
non-training producer finishes (for example discovered domains, tasks, shards or
datasets).  This layer lets a repository declare deterministic *deferred job
expanders* without hiding the resulting experiments inside one opaque child.

Safety invariants:

* only non-training, durably restartable prerequisite jobs may run before the
  complete repository training audit;
* producer artifacts, the expander source, descriptor and generated job catalog
  are fingerprinted and frozen atomically;
* any materialization drift fails closed on restart;
* the complete expanded graph is passed through the ordinary source coverage,
  exact-resume, early-stopping, registry and DAG audits before any training job
  launches;
* every generated concrete job is then scheduled by the unchanged pinned
  OPF_ADP scheduler through the existing DAG executor.

No CPU/GPU/RAM/VRAM admission or process-management logic lives here.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import universal_training_controller as base
import universal_training_controller_dag as dag
import universal_training_controller_training_contracts as training_contracts

DEFERRED_SCHEMA = 1
STATE_NAME = "deferred_expansions.json"
EXECUTION_STATE_NAME = "deferred_execution_state.json"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    base._write_atomic(path, (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _repository_path(root: Path, value: str, *, must_exist: bool = True) -> Path:
    relative = str(value or "").strip()
    if not relative:
        raise SystemExit("deferred expander path may not be empty")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except Exception as exc:
        raise SystemExit(f"deferred path escapes repository root: {relative}") from exc
    if must_exist and not path.is_file():
        raise SystemExit(f"deferred path does not exist: {path}")
    return path


def _descriptors(profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = profile.get("deferred_job_expanders", []) or []
    if not isinstance(raw, list):
        raise SystemExit("profile.deferred_job_expanders must be a list")
    descriptors: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise SystemExit(f"deferred_job_expanders[{index}] must be an object")
        descriptor = dict(item)
        expander_id = str(descriptor.get("id") or "").strip()
        if not expander_id:
            raise SystemExit(f"deferred_job_expanders[{index}].id is required")
        if expander_id in seen:
            raise SystemExit(f"duplicate deferred expander id: {expander_id}")
        seen.add(expander_id)
        if not str(descriptor.get("path") or "").strip():
            raise SystemExit(f"deferred expander {expander_id} requires path")
        depends = descriptor.get("depends_on", []) or []
        artifacts = descriptor.get("artifact_inputs", []) or []
        args = descriptor.get("args", []) or []
        kwargs = descriptor.get("kwargs", {}) or {}
        if isinstance(depends, str):
            depends = [depends]
        if isinstance(artifacts, str):
            artifacts = [artifacts]
        if not isinstance(depends, list) or not all(str(value).strip() for value in depends):
            raise SystemExit(f"deferred expander {expander_id}.depends_on must be a string list")
        if not isinstance(artifacts, list) or not all(str(value).strip() for value in artifacts):
            raise SystemExit(f"deferred expander {expander_id}.artifact_inputs must be a string list")
        if not isinstance(args, list) or not isinstance(kwargs, Mapping):
            raise SystemExit(f"deferred expander {expander_id} args/kwargs are malformed")
        descriptor["depends_on"] = [str(value) for value in depends]
        descriptor["artifact_inputs"] = [str(value) for value in artifacts]
        descriptor["args"] = list(args)
        descriptor["kwargs"] = dict(kwargs)
        descriptor["function"] = str(descriptor.get("function") or "iter_jobs")
        descriptors.append(descriptor)
    return descriptors


def _by_id(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for record in records:
        job_id = str(record.get("id") or "")
        if not job_id:
            raise SystemExit("compiled job has no id")
        if job_id in result:
            raise SystemExit(f"duplicate compiled job id: {job_id}")
        result[job_id] = record
    return result


def _dependency_list(record: Mapping[str, Any]) -> List[str]:
    raw = record.get("depends_on", []) or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item) for item in raw]


def _producer_closure(records: Sequence[Mapping[str, Any]], descriptors: Sequence[Mapping[str, Any]]) -> List[str]:
    index = _by_id(records)
    requested = {str(dep) for descriptor in descriptors for dep in descriptor.get("depends_on", []) or []}
    unknown = sorted(requested - set(index))
    if unknown:
        raise SystemExit(f"deferred expander prerequisites are not compiled jobs: {unknown}")
    closure: set[str] = set()
    stack = list(sorted(requested))
    while stack:
        job_id = stack.pop()
        if job_id in closure:
            continue
        closure.add(job_id)
        for dependency in _dependency_list(index[job_id]):
            if dependency not in index:
                raise SystemExit(f"producer prerequisite {job_id} has unknown dependency {dependency}")
            stack.append(dependency)
    ordered = [str(record.get("id")) for record in records if str(record.get("id")) in closure]
    return ordered


def _validate_producers(records: Sequence[Mapping[str, Any]], producer_ids: Sequence[str]) -> None:
    index = _by_id(records)
    producers = [index[job_id] for job_id in producer_ids]
    # Reuse the ordinary DAG validator to reject unknown dependencies/cycles.
    dag._dependency_waves([dict(record) for record in producers])
    failures: List[str] = []
    for job in producers:
        job_id = str(job.get("id"))
        if training_contracts._is_training_job(job):
            failures.append(f"{job_id}: training job cannot be a deferred-expansion prerequisite")
            continue
        strategy = str(job.get("resume_strategy") or "").strip().lower()
        contract = job.get("checkpoint_contract")
        declared_exact = isinstance(contract, Mapping) and contract.get("exact_resume") is True
        if strategy == "restart_exact":
            if not all(job.get(key) is True for key in ("deterministic", "idempotent", "atomic_outputs")):
                failures.append(f"{job_id}: restart_exact producer lacks deterministic/idempotent/atomic_outputs")
        elif strategy in {"exact_checkpoint", "framework_exact_checkpoint"}:
            if not declared_exact and strategy not in {"exact_checkpoint", "framework_exact_checkpoint"}:
                failures.append(f"{job_id}: exact producer lacks exact checkpoint contract")
        else:
            failures.append(f"{job_id}: producer durability is not exact/restart_exact ({strategy or 'missing'})")
    if failures:
        raise SystemExit("Deferred producer preflight failed:\n  - " + "\n  - ".join(failures))


def _load_expander(root: Path, descriptor: Mapping[str, Any]):
    path = _repository_path(root, str(descriptor.get("path")))
    module_name = "_training_control_deferred_" + hashlib.sha1(str(path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import deferred expander {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    function_name = str(descriptor.get("function") or "iter_jobs")
    function = getattr(module, function_name, None)
    if not callable(function):
        raise SystemExit(f"deferred expander {path} has no callable {function_name}()")
    return path, function


def _normalize_generated_job(root: Path, raw: Mapping[str, Any], descriptor: Mapping[str, Any], index: int) -> Dict[str, Any]:
    item = dict(raw)
    command_value = item.get("command") or item.get("entrypoint")
    if not command_value:
        raise SystemExit(f"deferred expander {descriptor['id']} job {index} has no command/entrypoint")
    job_id = str(item.get("id") or item.get("name") or "").strip()
    if not job_id:
        raise SystemExit(f"deferred expander {descriptor['id']} job {index} has no stable id")
    dependencies = item.get("depends_on", []) or []
    if isinstance(dependencies, str):
        dependencies = [dependencies]
    dependencies = [str(value) for value in dependencies]
    for dependency in descriptor.get("depends_on", []) or []:
        if str(dependency) not in dependencies:
            dependencies.append(str(dependency))
    record: Dict[str, Any] = {
        "id": job_id,
        "command": base._normalize_command(root, command_value),
        "device_capable": bool(item.get("device_capable", True)),
        "phase": str(item.get("phase") or "training"),
        "family": str(item.get("family") or "deferred"),
        "repeat_index": int(item.get("repeat_index", 0)),
        "depends_on": dependencies,
        "deferred_expander_id": str(descriptor["id"]),
    }
    for key, value in item.items():
        if key not in {"id", "name", "command", "entrypoint", "device_capable", "phase", "family", "repeat_index", "depends_on"}:
            record[key] = value
    return record


def _materialize_one(root: Path, descriptor: Mapping[str, Any]) -> Dict[str, Any]:
    source_path, function = _load_expander(root, descriptor)
    artifact_rows: List[Dict[str, str]] = []
    for relative in descriptor.get("artifact_inputs", []) or []:
        path = _repository_path(root, str(relative))
        artifact_rows.append({"path": str(relative), "sha256": _sha256_file(path)})
    args = list(descriptor.get("args", []) or [])
    kwargs = dict(descriptor.get("kwargs", {}) or {})
    generated_raw = list(function(*args, **kwargs))
    if not generated_raw:
        raise SystemExit(f"deferred expander {descriptor['id']} generated zero jobs")
    records: List[Dict[str, Any]] = []
    raw_for_state: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_commands: set[tuple[str, ...]] = set()
    for index, raw in enumerate(generated_raw):
        if not isinstance(raw, Mapping):
            raise SystemExit(f"deferred expander {descriptor['id']} item {index} is not an object")
        record = _normalize_generated_job(root, raw, descriptor, index)
        job_id = str(record["id"])
        command_key = tuple(str(value) for value in record["command"])
        if job_id in seen_ids:
            raise SystemExit(f"deferred expander {descriptor['id']} generated duplicate id {job_id}")
        if command_key in seen_commands:
            raise SystemExit(f"deferred expander {descriptor['id']} generated duplicate command for {job_id}")
        seen_ids.add(job_id); seen_commands.add(command_key)
        records.append(record)
        raw_for_state.append(dict(raw))
    descriptor_fingerprint = _sha256_bytes(_canonical_bytes(dict(descriptor)))
    source_fingerprint = _sha256_file(source_path)
    generated_fingerprint = _sha256_bytes(_canonical_bytes(raw_for_state))
    fingerprint_payload = {
        "descriptor_sha256": descriptor_fingerprint,
        "source_sha256": source_fingerprint,
        "artifacts": artifact_rows,
        "generated_sha256": generated_fingerprint,
    }
    return {
        "id": str(descriptor["id"]),
        "descriptor": dict(descriptor),
        "descriptor_sha256": descriptor_fingerprint,
        "source": str(source_path.relative_to(root)),
        "source_sha256": source_fingerprint,
        "artifacts": artifact_rows,
        "generated_sha256": generated_fingerprint,
        "materialization_sha256": _sha256_bytes(_canonical_bytes(fingerprint_payload)),
        "generated_job_count": len(records),
        "generated_job_ids": [str(record["id"]) for record in records],
        "generated_jobs": raw_for_state,
        "records": records,
    }


def _load_previous_state(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot parse previous deferred expansion state: {path}") from exc
    if not isinstance(payload, dict) or int(payload.get("schema", -1)) != DEFERRED_SCHEMA:
        raise SystemExit(f"unsupported deferred expansion state: {path}")
    return payload


def _freeze_materializations(root: Path, profile: Mapping[str, Any], descriptors: Sequence[Mapping[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    materialized = [_materialize_one(root, descriptor) for descriptor in descriptors]
    all_records: List[Dict[str, Any]] = []
    ids: set[str] = set()
    for row in materialized:
        for record in row.pop("records"):
            job_id = str(record["id"])
            if job_id in ids:
                raise SystemExit(f"duplicate generated job id across deferred expanders: {job_id}")
            ids.add(job_id); all_records.append(record)
    state_path = root / ".training_control" / STATE_NAME
    previous = _load_previous_state(state_path)
    frozen_rows = [dict(row) for row in materialized]
    state = {
        "schema": DEFERRED_SCHEMA,
        "repository": profile.get("repository") or root.name,
        "expanders": frozen_rows,
    }
    if previous is not None:
        old = {str(row.get("id")): row for row in previous.get("expanders", []) if isinstance(row, Mapping)}
        new = {str(row.get("id")): row for row in frozen_rows}
        drift: List[str] = []
        if set(old) != set(new):
            drift.append(f"expander ids changed: {sorted(old)} -> {sorted(new)}")
        for expander_id in sorted(set(old) & set(new)):
            if old[expander_id].get("materialization_sha256") != new[expander_id].get("materialization_sha256"):
                drift.append(expander_id)
        if drift and os.environ.get("TRAINING_CONTROL_ALLOW_DEFERRED_REBUILD", "").strip().lower() not in {"1", "true", "yes", "on"}:
            raise SystemExit(
                "Deferred job materialization drifted from the frozen prior run. "
                "Refusing to mix experiment universes. Drift: " + ", ".join(drift) + ". "
                "Start from a clean training-control state or explicitly set "
                "TRAINING_CONTROL_ALLOW_DEFERRED_REBUILD=1 after reviewing downstream artifacts."
            )
    _atomic_json(state_path, state)
    return all_records, frozen_rows


def _combine_records(static: Sequence[Mapping[str, Any]], generated: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    combined = [dict(record) for record in static]
    known_ids = {str(record.get("id")) for record in combined}
    commands = {tuple(str(part) for part in record.get("command", [])) for record in combined}
    for record in generated:
        job_id = str(record.get("id"))
        command = tuple(str(part) for part in record.get("command", []))
        if job_id in known_ids:
            raise SystemExit(f"deferred job id collides with static job: {job_id}")
        if command in commands:
            raise SystemExit(f"deferred job command duplicates a static job: {job_id}")
        known_ids.add(job_id); commands.add(command); combined.append(dict(record))
    # Full dependency validation is deliberately after expansion.
    dag._dependency_waves(combined)
    return combined


def _without_completed(records: Sequence[Mapping[str, Any]], completed_ids: Iterable[str]) -> List[Dict[str, Any]]:
    completed = {str(value) for value in completed_ids}
    remaining: List[Dict[str, Any]] = []
    for raw in records:
        job_id = str(raw.get("id"))
        if job_id in completed:
            continue
        record = copy.deepcopy(dict(raw))
        record["depends_on"] = [dep for dep in _dependency_list(record) if dep not in completed]
        remaining.append(record)
    if remaining:
        dag._dependency_waves(remaining)
    return remaining


def _stage_forwarded(forwarded: Sequence[str], stage: str) -> List[str]:
    normalized = base._ensure_opf_cli(forwarded)
    results_dir = Path(str(dag._arg_value(normalized, "--results-dir", "Results/training_control")))
    requested = dag._arg_value(normalized, "--run-root")
    base_root = Path(str(requested)) if requested else results_dir / "run_root"
    return dag._set_arg(list(forwarded), "--run-root", str(base_root / "deferred_stages" / stage))


def _artifacts_ready(root: Path, descriptors: Sequence[Mapping[str, Any]]) -> bool:
    return all((root / str(path)).is_file() for descriptor in descriptors for path in descriptor.get("artifact_inputs", []) or [])


def _report_with_deferred(report: Dict[str, Any], *, rows: Sequence[Mapping[str, Any]], pending: bool) -> Dict[str, Any]:
    report = dict(report)
    report["deferred_expansion_schema"] = DEFERRED_SCHEMA
    report["deferred_expansions_pending"] = bool(pending)
    report["deferred_expanders"] = [
        {
            "id": row.get("id"),
            "generated_job_count": row.get("generated_job_count"),
            "generated_job_ids": row.get("generated_job_ids", []),
            "materialization_sha256": row.get("materialization_sha256"),
        }
        for row in rows
    ]
    if pending:
        report["coverage_ok"] = False
    return report


def main(argv: Sequence[str] | None = None) -> int:
    root = base._repo_root()
    profile = base._load_profile()
    descriptors = _descriptors(profile)
    if not descriptors:
        return dag.main(argv)

    # Install DAG-enforced coverage semantics before compiling any records.
    dag._install()
    flags, forwarded = base._custom_flags(list(sys.argv[1:] if argv is None else argv))
    if flags["skip_setup"]:
        os.environ["TRAINING_CONTROL_SKIP_SETUP"] = "1"
    static_records = base._job_records(root, profile)
    producer_ids = _producer_closure(static_records, descriptors)
    _validate_producers(static_records, producer_ids)
    producer_set = set(producer_ids)
    producer_records = [dict(record) for record in static_records if str(record.get("id")) in producer_set]

    audit_only = bool(flags["audit"])
    list_only = bool(flags["list_jobs"])
    dry_run = dag._has_arg(forwarded, "--dry-run")
    ready_before_execution = _artifacts_ready(root, descriptors)
    if (audit_only or list_only or dry_run) and not ready_before_execution:
        report = base._coverage_report(root, profile, static_records)
        report = _report_with_deferred(report, rows=[{"id": d["id"]} for d in descriptors], pending=True)
        report["deferred_producer_job_ids"] = producer_ids
        report_path = root / ".training_control" / "coverage_report.json"
        _atomic_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if audit_only:
            return 2
        # Listing/dry-run must never execute producers merely to discover jobs.
        return 0

    if not ready_before_execution:
        base._run_setup(root, profile)
        if producer_records:
            result = dag._execute(root, profile, producer_records, _stage_forwarded(forwarded, "producers"))
            if result != 0:
                return int(result)
    if not _artifacts_ready(root, descriptors):
        raise SystemExit("Deferred producers completed but required enumeration artifacts are still missing")

    generated_records, frozen_rows = _freeze_materializations(root, profile, descriptors)
    full_records = _combine_records(static_records, generated_records)
    full_report = base._coverage_report(root, profile, full_records)
    full_report = _report_with_deferred(full_report, rows=frozen_rows, pending=False)
    full_report["deferred_producer_job_ids"] = producer_ids
    report_path = root / ".training_control" / "coverage_report.json"
    _atomic_json(report_path, full_report)
    if audit_only or list_only:
        print(json.dumps(full_report, indent=2, sort_keys=True))
    if audit_only:
        return 0 if full_report.get("coverage_ok") else 2

    strict = bool(profile.get("strict_coverage", True)) and not flags["allow_uncovered"]
    if strict and not full_report.get("coverage_ok"):
        raise SystemExit(
            f"Expanded training coverage audit failed. See {report_path}. "
            "No training job has been launched."
        )
    if list_only:
        return 0
    if not ready_before_execution:
        # setup already ran before producer execution
        pass
    else:
        base._run_setup(root, profile)

    remaining = _without_completed(full_records, producer_ids)
    execution_state = {
        "schema": DEFERRED_SCHEMA,
        "repository": profile.get("repository") or root.name,
        "producer_job_ids": producer_ids,
        "generated_job_ids": [str(record.get("id")) for record in generated_records],
        "remaining_job_count": len(remaining),
        "coverage_ok": bool(full_report.get("coverage_ok")),
    }
    _atomic_json(root / ".training_control" / EXECUTION_STATE_NAME, execution_state)
    if not remaining:
        print("[training-control] deferred producers completed and expansion generated no remaining jobs")
        return 0
    return dag._execute(root, profile, remaining, _stage_forwarded(forwarded, "expanded"))
