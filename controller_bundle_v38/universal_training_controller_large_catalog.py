#!/usr/bin/env python3
"""Scale strict source/resume audits to very large explicit job catalogs.

This layer changes no discovery criteria, contract criteria, DAG semantics or OPF
scheduling.  It removes repeated work only:

* metadata lookup is indexed once instead of scanning profile.jobs per job;
* repository import/execution graphs are built once;
* graph closure is cached per unique direct source signature (many experiment
  jobs differ only in CLI hyperparameters but execute the same trainer);
* exact-resume source contracts are evaluated once per reachable-source
  signature;
* per-job resume rows retain the strict booleans needed by downstream audits,
  while verbose source evidence is emitted once in ``resume_contract_groups``.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import universal_training_controller_current as current

LARGE_CATALOG_SCHEMA = 1
_METADATA_CACHE_KEY = "_training_control_metadata_index_v1"
_CONTRACT_CACHE: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
_CONTRACT_GROUPS: Dict[Tuple[str, str], Dict[str, Any]] = {}
_ORIGINAL_METADATA = None
_ORIGINAL_CHECKPOINT = None
_ORIGINAL_REPORT = None


def _metadata_index(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cached = profile.get(_METADATA_CACHE_KEY)
    if isinstance(cached, dict):
        return cached
    index: Dict[str, Dict[str, Any]] = {}
    configured = profile.get("job_metadata", {})
    if isinstance(configured, dict):
        for job_id, value in configured.items():
            if isinstance(value, dict):
                index[str(job_id)] = dict(value)
    excluded = {"id", "name", "command", "entrypoint", "device_capable", "phase", "family", "repeat_index"}
    for collection_name in ("jobs", "extra_jobs"):
        collection = profile.get(collection_name, []) or []
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("id") or item.get("name") or item.get("entrypoint") or "")
            if not candidate_id:
                continue
            row = index.setdefault(candidate_id, {})
            row.update({key: value for key, value in item.items() if key not in excluded})
    profile[_METADATA_CACHE_KEY] = index
    return index


def _metadata_for_record(profile: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(_metadata_index(profile).get(str(record.get("id")), {}))


def _direct_sources(root: Path, job: Mapping[str, Any], module_index: Mapping[str, str]) -> Tuple[str, ...]:
    covered: Set[str] = set()
    command = [str(value) for value in job.get("command", [])]
    for part in command:
        try:
            candidate = Path(part)
            if candidate.is_absolute():
                rel = candidate.resolve().relative_to(root).as_posix()
                covered.add(rel)
            elif (root / candidate).is_file():
                covered.add((root / candidate).resolve().relative_to(root).as_posix())
        except Exception:
            continue
    for index, part in enumerate(command[:-1]):
        if part == "-m":
            covered.update(current._resolve_module(module_index, command[index + 1]))
    return tuple(sorted(
        rel for rel in covered
        if (root / rel).is_file() and Path(rel).suffix.lower() in current.SOURCE_SUFFIXES
    ))


def _closure(
    direct: Tuple[str, ...],
    import_graph: Mapping[str, Set[str]],
    execution_graph: Mapping[str, Set[str]],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    executed: Set[str] = set()
    queue = deque(direct)
    while queue:
        rel = queue.popleft()
        if rel in executed:
            continue
        executed.add(rel)
        queue.extend(execution_graph.get(rel, set()) - executed)
    reachable: Set[str] = set(executed)
    queue = deque(executed)
    while queue:
        rel = queue.popleft()
        for nxt in import_graph.get(rel, set()) | execution_graph.get(rel, set()):
            if nxt not in reachable:
                reachable.add(nxt)
                queue.append(nxt)
    return tuple(sorted(executed)), tuple(sorted(reachable))


def _reachability(root: Path, jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    module_index = current._python_module_index(root)
    import_graph: Dict[str, Set[str]] = {}
    execution_graph: Dict[str, Set[str]] = {}
    for path in current._iter_sources(root):
        rel = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".py":
            import_graph[rel] = current._python_import_edges(root, rel, module_index)
        if path.suffix.lower() in current.EXEC_SCRIPT_SUFFIXES:
            execution_graph[rel] = current._text_execution_edges(root, rel, module_index)

    closure_cache: Dict[Tuple[str, ...], Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
    per_job: Dict[str, Dict[str, List[str]]] = {}
    all_direct: Set[str] = set()
    all_executed: Set[str] = set()
    all_reachable: Set[str] = set()
    for job in jobs:
        direct = _direct_sources(root, job, module_index)
        all_direct.update(direct)
        if direct not in closure_cache:
            closure_cache[direct] = _closure(direct, import_graph, execution_graph)
        executed, reachable = closure_cache[direct]
        all_executed.update(executed)
        all_reachable.update(reachable)
        per_job[str(job.get("id"))] = {
            "direct": list(direct),
            "executed": list(executed),
            "reachable": list(reachable),
        }
    return {
        "direct_job_sources": sorted(all_direct),
        "executed_sources": sorted(all_executed),
        "reachable_sources": sorted(all_reachable),
        "per_job": per_job,
        "unique_direct_source_signatures": len(closure_cache),
        "python_import_edge_count": sum(len(value) for value in import_graph.values()),
        "local_execution_edge_count": sum(len(value) for value in execution_graph.values()),
    }


def _contract_signature(paths: Iterable[str]) -> str:
    payload = json.dumps(sorted(set(paths)), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _checkpoint_contract_for_paths(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    normalized = tuple(sorted(set(str(path) for path in paths)))
    key = (str(root.resolve()), normalized)
    cached = _CONTRACT_CACHE.get(key)
    if cached is None:
        assert _ORIGINAL_CHECKPOINT is not None
        cached = _ORIGINAL_CHECKPOINT(root, normalized)
        _CONTRACT_CACHE[key] = cached
    signature = _contract_signature(normalized)
    _CONTRACT_GROUPS[(str(root.resolve()), signature)] = {
        "signature": signature,
        "reachable_sources": list(normalized),
        "checkpoint_contract": cached,
    }
    return cached


def _job_resume_evidence(root: Path, job: Dict[str, Any], reachability: Mapping[str, Any]) -> Dict[str, Any]:
    job_id = str(job.get("id"))
    paths = reachability.get("per_job", {}).get(job_id, {}).get("reachable", [])
    contract = current._checkpoint_contract_for_paths(root, paths)
    declared = str(job.get("resume_strategy") or "").strip().lower()
    declared_contract = job.get("checkpoint_contract")
    declared_exact = isinstance(declared_contract, dict) and declared_contract.get("exact_resume") is True
    if declared in {"exact_checkpoint", "framework_exact_checkpoint"}:
        declared_exact = True
    native = bool(contract.get("native_resume_detected"))
    if declared in {"native_checkpoint", "framework_checkpoint", "exact_checkpoint", "framework_exact_checkpoint"}:
        native = True
    exact = bool(contract.get("exact_resume_detected") or declared_exact)
    early = bool(contract.get("early_stopping") or job.get("early_stopping") is True)
    return {
        "job_id": job_id,
        "resume_strategy": declared or ("native_checkpoint_detected" if native else "unproven"),
        "native_resume_proven": native,
        "exact_resume_proven": exact,
        "early_stopping_present": early,
        "contract_signature": _contract_signature(paths),
    }


def _coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    assert _ORIGINAL_REPORT is not None
    report = _ORIGINAL_REPORT(root, profile, jobs)
    root_key = str(root.resolve())
    groups = {
        signature: payload
        for (group_root, signature), payload in _CONTRACT_GROUPS.items()
        if group_root == root_key
    }
    report.update({
        "large_catalog_audit_schema": LARGE_CATALOG_SCHEMA,
        "resume_contract_group_count": len(groups),
        "resume_contract_groups": groups,
    })
    return report


def install() -> None:
    global _ORIGINAL_METADATA, _ORIGINAL_CHECKPOINT, _ORIGINAL_REPORT
    if getattr(current._metadata_for_record, "_training_control_large_catalog", False):
        return
    _ORIGINAL_METADATA = current._metadata_for_record
    _ORIGINAL_CHECKPOINT = current._checkpoint_contract_for_paths
    _ORIGINAL_REPORT = current._enhanced_coverage_report
    _metadata_for_record._training_control_large_catalog = True  # type: ignore[attr-defined]
    current._metadata_for_record = _metadata_for_record
    current._reachability = _reachability
    current._checkpoint_contract_for_paths = _checkpoint_contract_for_paths
    current._job_resume_evidence = _job_resume_evidence
    current._enhanced_coverage_report = _coverage_report
