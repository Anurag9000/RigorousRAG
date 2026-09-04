#!/usr/bin/env python3
"""Static argparse-subcommand discovery below PEP 621 console entrypoints.

A console command whose training work is exposed through an argparse subcommand
must not be scheduled twice: once as the parent and once as the concrete
subcommand.  This layer therefore treats a parent training console entrypoint as
*delegated* (not exempted) when one of its concrete training subcommands has been
materialized as a job.  Every individual training subcommand remains audited by
this module, so delegation cannot hide an unscheduled sibling subcommand.
"""
from __future__ import annotations

import ast
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

import universal_training_controller as base
import universal_training_controller_console as console
import universal_training_controller_current as current

SUBCOMMAND_AUDIT_SCHEMA = 2


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _subcommands(root: Path, rel: str | None) -> List[Dict[str, Any]]:
    if not rel:
        return []
    text = current._read_text(root / rel)
    try:
        tree = ast.parse(text, filename=rel)
    except (SyntaxError, ValueError):
        return []
    parsers: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not current._call_name(value.func).endswith("add_parser") or not value.args:
            continue
        name = _literal(value.args[0])
        if not isinstance(name, str) or not name:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                parsers[target.id] = name
    contracts: Dict[str, Dict[str, Any]] = {
        name: {"required_options": set(), "required_positionals": set(), "dynamic_argument_definitions": 0}
        for name in parsers.values()
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call = current._call_name(node.func)
        if not call.endswith(".add_argument"):
            continue
        owner = call.rsplit(".", 1)[0]
        if owner not in parsers:
            continue
        contract = contracts[parsers[owner]]
        if not node.args:
            contract["dynamic_argument_definitions"] += 1
            continue
        first = _literal(node.args[0])
        if not isinstance(first, str):
            contract["dynamic_argument_definitions"] += 1
            continue
        flags = [x for x in (_literal(a) for a in node.args) if isinstance(x, str)]
        kw = {x.arg: _literal(x.value) for x in node.keywords if x.arg}
        if first.startswith("-"):
            if kw.get("required") is True:
                contract["required_options"].add(next((x for x in flags if x.startswith("--")), first))
        elif kw.get("nargs") not in {"?", "*"}:
            contract["required_positionals"].add(first)
    result = []
    for name in sorted(contracts):
        item = contracts[name]
        result.append({
            "name": name,
            "required_options": sorted(item["required_options"]),
            "required_positionals": sorted(item["required_positionals"]),
            "dynamic_argument_definitions": int(item["dynamic_argument_definitions"]),
        })
    return result


def _explicit_args(profile: Mapping[str, Any], key: str) -> List[str] | None:
    mapping = profile.get("console_subcommand_args", {})
    if not isinstance(mapping, Mapping) or key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    raise SystemExit(f"console_subcommand_args[{key!r}] must be string/list/null")


def _default_args(profile: Mapping[str, Any], contract: Mapping[str, Any]) -> List[str] | None:
    if contract.get("required_positionals"):
        return None
    defaults = profile.get("console_argument_defaults", {})
    if not isinstance(defaults, Mapping):
        return None
    args: List[str] = []
    for option in contract.get("required_options", []) or []:
        if option not in defaults:
            return None
        value = defaults[option]
        if isinstance(value, bool):
            if value:
                args.append(str(option))
            continue
        if value is None:
            return None
        args.extend([str(option), str(value)])
    return args


def _ignored(profile: Mapping[str, Any], key: str) -> bool:
    raw = profile.get("ignore_console_subcommands", []) or []
    return key in raw if isinstance(raw, Mapping) else key in {str(x) for x in raw}


def _inventory(root: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for script_name, target in sorted(console._read_scripts(root).items()):
        module, func = console._target(target)
        rel = console._source(root, module)
        for contract in _subcommands(root, rel):
            name = contract["name"]
            key = f"{script_name}:{name}"
            training = bool(console.TRAINING_NAME.search(name))
            if not training:
                continue
            explicit = _explicit_args(profile, key)
            defaults = _default_args(profile, contract)
            ignored = _ignored(profile, key)
            configured = bool(ignored or explicit is not None or defaults is not None)
            entries.append({
                "key": key, "console_script": script_name, "subcommand": name,
                "target": target, "module": module, "callable": func,
                "source": rel, "training_surface": True, "ignored": ignored,
                "explicit_args": explicit, "default_args": defaults,
                "configured": configured, **contract,
            })
    return {
        "schema": SUBCOMMAND_AUDIT_SCHEMA,
        "registered_training_subcommand_count": len(entries),
        "training_subcommands": entries,
        "unresolved_targets": [x["key"] for x in entries if not x["source"] and not x["ignored"]],
        "unconfigured_training_subcommands": [x["key"] for x in entries if not x["configured"] and not x["ignored"]],
    }


def _entry_code(module: str, func: str) -> str:
    return (
        "import sys;sys.path[:0]=['src','lib','python'];"
        f"from {module} import {func} as _entry;"
        "_r=_entry();raise SystemExit(_r if isinstance(_r,int) else 0)"
    )


def _jobs(root: Path, profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not bool(profile.get("auto_console_subcommand_jobs", False)):
        return []
    result: List[Dict[str, Any]] = []
    for item in _inventory(root, profile)["training_subcommands"]:
        if item["ignored"] or not item["source"]:
            continue
        args = item["explicit_args"] if item["explicit_args"] is not None else item["default_args"]
        if args is None:
            continue
        meta: Dict[str, Any] = {
            "console_script": item["console_script"],
            "console_subcommand": item["subcommand"],
            "entrypoint_source": item["source"],
            "registry": "pyproject.toml:[project.scripts]/argparse",
        }
        custom = profile.get("console_subcommand_metadata", {})
        if isinstance(custom, Mapping) and isinstance(custom.get(item["key"]), Mapping):
            meta.update(dict(custom[item["key"]]))
        if "depends_on" not in meta:
            deps = profile.get("console_subcommand_default_depends_on", []) or []
            meta["depends_on"] = [str(x) for x in ([deps] if isinstance(deps, str) else deps)]
        result.append({
            "id": f"console:{item['key']}",
            "command": [sys.executable, "-c", _entry_code(item["module"], item["callable"]), item["subcommand"], *args],
            "device_capable": True, "phase": "training", "family": "console-subcommand", "repeat_index": 0,
            **meta,
        })
    return result


def _delegated_console_scripts(jobs: Sequence[Mapping[str, Any]]) -> Dict[str, list[str]]:
    """Return parent console scripts satisfied by concrete scheduled subcommands."""
    delegated: Dict[str, set[str]] = {}
    for job in jobs:
        script = str(job.get("console_script") or "").strip()
        subcommand = str(job.get("console_subcommand") or "").strip()
        if not script or not subcommand:
            continue
        delegated.setdefault(script, set()).add(subcommand)
    return {name: sorted(values) for name, values in sorted(delegated.items())}


def _with_delegated_parent_ignores(profile: Mapping[str, Any], delegated: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """Suppress duplicate parent scheduling only while the lower report is built.

    The resulting report is normalized immediately afterwards so delegated
    parents are reported as satisfied-by-subcommand, not as user exemptions.
    """
    augmented = dict(profile)
    raw = profile.get("ignore_console_scripts", []) or []
    if isinstance(raw, Mapping):
        existing = {str(key) for key in raw}
    elif isinstance(raw, str):
        existing = {raw}
    else:
        existing = {str(value) for value in raw}
    augmented["ignore_console_scripts"] = sorted(existing | set(delegated))
    return augmented


def _normalize_delegated_parent_report(report: Dict[str, Any], delegated: Mapping[str, Sequence[str]]) -> None:
    if not delegated:
        report["console_entrypoints_satisfied_by_subcommands"] = {}
        return
    registry = report.get("console_registry")
    if not isinstance(registry, dict):
        report["console_entrypoints_satisfied_by_subcommands"] = dict(delegated)
        return
    entries = registry.get("entries") or []
    explicit_ignored_raw = report.get("strict_controls", {}).get("explicit_ignore_console_scripts", [])
    explicit_ignored = {str(x) for x in explicit_ignored_raw} if isinstance(explicit_ignored_raw, (list, tuple, set)) else set()
    for item in entries:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name not in delegated:
            continue
        # The temporary lower-layer ignore is an implementation detail, not an
        # exemption.  The parent is covered by a concrete scheduled child.
        if name not in explicit_ignored:
            item["ignored"] = False
        item["configured"] = True
        item["satisfied_by_subcommand"] = True
        item["satisfied_by_subcommands"] = list(delegated[name])
    training_entries = [x for x in entries if isinstance(x, dict) and x.get("training_surface")]
    registry["training_entries"] = training_entries
    registry["unconfigured_training_entrypoints"] = [
        x["name"] for x in training_entries
        if not x.get("configured") and not x.get("ignored") and str(x.get("name") or "") not in delegated
    ]
    report["console_entrypoints_satisfied_by_subcommands"] = {
        name: list(commands) for name, commands in delegated.items()
    }
    scheduled_parents = {str(x) for x in report.get("compiled_console_training_jobs", []) or []}
    active_parents = {
        str(x.get("name")) for x in training_entries
        if x.get("name") and not x.get("ignored")
    }
    report["unscheduled_registered_training_entrypoints"] = sorted(active_parents - scheduled_parents - set(delegated))
    missing = [
        str(x) for x in report.get("missing_console_job_materialization", []) or []
        if str(x) not in delegated
    ]
    report["missing_console_job_materialization"] = sorted(set(missing))
    unresolved = [
        str(x) for x in registry.get("unresolved_targets", []) or []
        if str(x) not in delegated
    ]
    registry["unresolved_targets"] = sorted(set(unresolved))
    parent_ok = not registry["unresolved_targets"] and not registry["unconfigured_training_entrypoints"] and not report["missing_console_job_materialization"]
    report["strict_registered_training_entrypoints_pass"] = parent_ok


def install() -> None:
    original_jobs = current._enhanced_job_records
    original_report = current._enhanced_coverage_report

    def job_records(root: Path, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        return base._dedupe_jobs([*original_jobs(root, profile), *_jobs(root, profile)])

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        delegated = _delegated_console_scripts(jobs)
        report_profile = _with_delegated_parent_ignores(profile, delegated) if delegated else profile
        report = original_report(root, report_profile, jobs)
        _normalize_delegated_parent_report(report, delegated)

        inv = _inventory(root, profile)
        compiled = sorted(
            f"{job.get('console_script')}:{job.get('console_subcommand')}"
            for job in jobs if job.get("console_subcommand")
        )
        scheduled = set(compiled)
        active = {x["key"] for x in inv["training_subcommands"] if not x["ignored"]}
        expected = {x["key"] for x in inv["training_subcommands"] if not x["ignored"] and x["source"] and x["configured"]}
        auto = bool(profile.get("auto_console_subcommand_jobs", False))
        require_schedule = bool(profile.get("require_registered_training_subcommand_scheduling", False))
        missing = sorted(expected - scheduled) if auto else (sorted(expected) if require_schedule else [])
        strict = bool(profile.get("require_registered_training_subcommands", True))
        ok = not inv["unresolved_targets"] and not inv["unconfigured_training_subcommands"] and not missing
        report.update({
            "console_subcommand_registry_schema": SUBCOMMAND_AUDIT_SCHEMA,
            "console_subcommand_registry": inv,
            "compiled_console_training_subcommands": compiled,
            "unscheduled_registered_training_subcommands": sorted(active - scheduled),
            "missing_console_subcommand_job_materialization": missing,
            "strict_registered_training_subcommands_pass": ok,
            "strict_controls": {
                **dict(report.get("strict_controls") or {}),
                "require_registered_training_subcommands": strict,
                "require_registered_training_subcommand_scheduling": require_schedule,
                "auto_console_subcommand_jobs": auto,
            },
        })
        if strict and not ok:
            report["coverage_ok"] = False
        return report

    current._enhanced_job_records = job_records
    current._enhanced_coverage_report = coverage_report
