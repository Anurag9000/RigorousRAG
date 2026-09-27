#!/usr/bin/env python3
"""Fail-closed member-level accounting for repository experiment registries.

The workload-closure layer proves that registry *files* are reachable.  That is
necessary but not sufficient for exhaustive orchestration: a reachable
``BACKBONES`` dictionary may contain three implemented models while every job
selects only one.  This layer therefore accounts for statically enumerable
members of model/architecture/dataset/task/method/algorithm/pipeline/benchmark/
environment/domain registries.

A member is considered centrally reachable only when one of the following is
true:

* a concrete compiled job names the member in its command or metadata;
* a compiled job that reaches the registry uses an explicit all-selector
  (``--all``, ``--all-models``, ``--all-datasets`` ...); or
* reachable repository source genuinely iterates the registry (rather than only
  exposing it as an argparse ``choices`` collection).

Repositories with dynamic/non-string registries remain visible as
``non_enumerable_registry_surfaces``.  Explicit exemptions require a reason and
may cover a whole ``path:symbol`` registry or a declared subset of members.
Nothing in this module schedules CPU/GPU/RAM/VRAM resources; it only strengthens
source/workload closure before the literal pinned OPF_ADP scheduler runs.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller_current as current
import universal_training_controller_workload_closure as workload

REGISTRY_MEMBER_SCHEMA = 1
ALL_FLAGS = {
    "--all", "--all-models", "--all-model", "--all-backbones", "--all-architectures",
    "--all-datasets", "--all-dataset", "--all-tasks", "--all-methods", "--all-algorithms",
    "--all-pipelines", "--all-benchmarks", "--all-environments", "--all-domains",
}


def _flatten(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        out: list[str] = []
        for key, item in value.items():
            out.extend(_flatten(key))
            out.extend(_flatten(item))
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [str(value)]


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _job_strings(job: Mapping[str, Any]) -> tuple[set[str], str]:
    values = [str(value) for value in _flatten(job)]
    normalized = {_norm(value) for value in values if _norm(value)}
    corpus = "\n".join(values).lower()
    return normalized, corpus


def _member_named(member: str, normalized: set[str], corpus: str) -> bool:
    target = _norm(member)
    if not target:
        return False
    if target in normalized:
        return True
    # Preserve compound identifiers as a unit.  Boundary matching prevents
    # e.g. model ``bert`` from being credited merely because ``roberta`` occurs.
    pattern = re.compile(r"(?<![a-z0-9])" + re.escape(str(member).lower()) + r"(?![a-z0-9])")
    return bool(pattern.search(corpus))


def _all_selector(job: Mapping[str, Any]) -> bool:
    command = [str(value).lower() for value in (job.get("command") or [])]
    for token in command:
        if token in ALL_FLAGS:
            return True
        if token.startswith("--all-"):
            return True
        if "=" in token and token.split("=", 1)[0] in ALL_FLAGS and token.split("=", 1)[1] not in {"0", "false", "no", "off"}:
            return True
    return False


def _looped_symbols(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
    except Exception:
        return set()
    result: set[str] = set()

    def root_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"items", "keys", "values"}:
            return root_name(node.func.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"sorted", "list", "tuple", "set", "iter", "enumerate"} and node.args:
            return root_name(node.args[0])
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            name = root_name(node.iter)
            if name:
                result.add(name)
        elif isinstance(node, ast.comprehension):
            name = root_name(node.iter)
            if name:
                result.add(name)
    return result


def _exemptions(profile: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw = profile.get("registry_member_coverage_exemptions", {}) or {}
    if not isinstance(raw, Mapping):
        raise SystemExit("registry_member_coverage_exemptions must be an object")
    result: Dict[str, Dict[str, Any]] = {}
    for key, value in raw.items():
        registry = str(key)
        if isinstance(value, str):
            if not value.strip():
                raise SystemExit(f"registry member exemption {registry} requires a reason")
            result[registry] = {"all": True, "reason": value.strip(), "members": []}
            continue
        if not isinstance(value, Mapping):
            raise SystemExit(f"registry member exemption {registry} must be string/object")
        reason = str(value.get("reason") or "").strip()
        if not reason:
            raise SystemExit(f"registry member exemption {registry} requires a reason")
        members = value.get("members", []) or []
        if isinstance(members, str):
            members = [members]
        if not isinstance(members, list):
            raise SystemExit(f"registry member exemption {registry}.members must be a list")
        result[registry] = {
            "all": bool(value.get("all", False)),
            "reason": reason,
            "members": [str(item) for item in members],
        }
    return result


def _inventory(root: Path, profile: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    base = workload._inventory(root, jobs)
    registries = list(base.get("registry_surfaces") or [])
    reach = current._reachability(root, jobs)
    reachable = {str(value) for value in (reach.get("reachable_sources") or [])}
    per_job = reach.get("per_job") if isinstance(reach.get("per_job"), Mapping) else {}
    exemptions = _exemptions(profile)

    job_evidence: Dict[str, tuple[set[str], str]] = {
        str(job.get("id")): _job_strings(job) for job in jobs
    }
    loop_cache: Dict[str, set[str]] = {}
    rows: list[Dict[str, Any]] = []
    non_enumerable: list[Dict[str, Any]] = []
    uncovered_rows: list[Dict[str, Any]] = []

    for registry in registries:
        path = str(registry.get("path") or "")
        symbol = str(registry.get("symbol") or "")
        members = [str(value) for value in (registry.get("members") or [])]
        key = f"{path}:{symbol}"
        if not members:
            non_enumerable.append({"registry": key, "path": path, "symbol": symbol})
            continue

        explicit: set[str] = set()
        generic_all_jobs: list[str] = []
        iterating_sources: list[str] = []
        for job in jobs:
            job_id = str(job.get("id"))
            normalized, corpus = job_evidence[job_id]
            for member in members:
                if _member_named(member, normalized, corpus):
                    explicit.add(member)
            reachable_for_job = set()
            if isinstance(per_job, Mapping) and isinstance(per_job.get(job_id), Mapping):
                reachable_for_job = {str(value) for value in (per_job[job_id].get("reachable") or [])}
            if path in reachable_for_job and _all_selector(job):
                generic_all_jobs.append(job_id)

        # Registry iteration is stronger than mere import/argparse choices: a
        # reachable loop over the registry means one concrete workflow naturally
        # enumerates every member itself.
        for source in sorted(reachable):
            source_path = root / source
            if not source_path.is_file() or source_path.suffix.lower() != ".py":
                continue
            if source not in loop_cache:
                loop_cache[source] = _looped_symbols(source_path)
            if symbol in loop_cache[source]:
                iterating_sources.append(source)

        covered = set(explicit)
        if generic_all_jobs or iterating_sources:
            covered.update(members)
        exemption = exemptions.get(key)
        exempted: set[str] = set()
        exemption_reason = None
        if exemption:
            exemption_reason = exemption["reason"]
            if exemption["all"]:
                exempted.update(members)
            else:
                requested = set(exemption["members"])
                unknown = sorted(requested - set(members))
                if unknown:
                    raise SystemExit(f"registry exemption {key} names unknown members: {unknown}")
                exempted.update(requested)
        missing = sorted(set(members) - covered - exempted)
        row = {
            "registry": key,
            "path": path,
            "symbol": symbol,
            "member_count": len(members),
            "members": sorted(members),
            "explicitly_named_members": sorted(explicit),
            "generic_all_jobs": sorted(set(generic_all_jobs)),
            "iterating_sources": sorted(set(iterating_sources)),
            "exempted_members": sorted(exempted),
            "exemption_reason": exemption_reason,
            "uncovered_members": missing,
            "complete": not missing,
        }
        rows.append(row)
        if missing:
            uncovered_rows.append(row)

    return {
        "schema": REGISTRY_MEMBER_SCHEMA,
        "registries": rows,
        "non_enumerable_registry_surfaces": non_enumerable,
        "uncovered_registries": uncovered_rows,
        "uncovered_registry_member_count": sum(len(row["uncovered_members"]) for row in uncovered_rows),
        "complete": not uncovered_rows,
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs):
        report = original_report(root, profile, jobs)
        inventory = _inventory(Path(root), profile, jobs)
        require = bool(profile.get("require_registry_member_accounting", profile.get("strict_coverage", True)))
        controls = dict(report.get("strict_controls") or {})
        controls["require_registry_member_accounting"] = require
        report.update(
            {
                "registry_member_closure_schema": REGISTRY_MEMBER_SCHEMA,
                "registry_member_inventory": inventory,
                "uncovered_registry_members": inventory["uncovered_registries"],
                "strict_registry_member_pass": bool(inventory["complete"]),
                "strict_controls": controls,
            }
        )
        if require and not inventory["complete"]:
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["REGISTRY_MEMBER_SCHEMA", "_inventory", "install"]
