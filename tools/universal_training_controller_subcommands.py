#!/usr/bin/env python3
"""Static argparse-subcommand discovery below PEP 621 console entrypoints."""
from __future__ import annotations

import ast
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

import universal_training_controller as base
import universal_training_controller_console as console
import universal_training_controller_current as current

SUBCOMMAND_AUDIT_SCHEMA = 1


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


def install() -> None:
    original_jobs = current._enhanced_job_records
    original_report = current._enhanced_coverage_report

    def job_records(root: Path, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        return base._dedupe_jobs([*original_jobs(root, profile), *_jobs(root, profile)])

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
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
