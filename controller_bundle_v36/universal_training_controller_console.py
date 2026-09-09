#!/usr/bin/env python3
"""PEP 621 console-script discovery and strict training-job materialization.

This layer closes the filename-discovery blind spot without changing OPF_ADP
scheduling: package-registered training CLIs become audited execution roots and,
when provably invokable, concrete jobs handed to the existing DAG/OPF stack.
"""
from __future__ import annotations

import ast
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import universal_training_controller as base
import universal_training_controller_current as current

CONSOLE_AUDIT_SCHEMA = 1
TRAINING_NAME = re.compile(
    r"(?:^|[-_.])(?:train|training|pretrain|finetune|fine[-_.]?tune|fit|adapt|distill|"
    r"specialists?|federated|sweep|search|optimi[sz]e|boosting|self[-_.]?supervised|"
    r"contrastive|qat)(?:[-_.]|$)", re.I
)


def _read_scripts(root: Path) -> Dict[str, str]:
    path = root / "pyproject.toml"
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8", errors="ignore")
    try:
        import tomllib
        doc = tomllib.loads(raw)
        scripts = (doc.get("project") or {}).get("scripts") or {}
        if isinstance(scripts, dict):
            return {str(k): str(v) for k, v in scripts.items() if isinstance(v, str)}
    except Exception:
        pass
    answer: Dict[str, str] = {}
    inside = False
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            inside = s == "[project.scripts]"
            continue
        if not inside or not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip().strip("'\"")
        value = value.strip().strip("'\"")
        if key and value:
            answer[key] = value
    return answer


def _target(target: str) -> Tuple[str, str]:
    value = target.split("[", 1)[0].strip()
    module, sep, func = value.partition(":")
    return module.strip(), (func.strip() if sep and func.strip() else "main")


def _source(root: Path, module: str) -> str | None:
    index = current._python_module_index(root)
    if module in index:
        return index[module]
    resolved = current._resolve_module(index, module)
    if len(resolved) == 1:
        return next(iter(resolved))
    suffix = Path(*module.split(".")).with_suffix(".py")
    for prefix in (Path(), Path("src"), Path("lib"), Path("python")):
        rel = (prefix / suffix).as_posix()
        if (root / rel).is_file():
            return rel
    package = Path(*module.split(".")) / "__init__.py"
    for prefix in (Path(), Path("src"), Path("lib"), Path("python")):
        rel = (prefix / package).as_posix()
        if (root / rel).is_file():
            return rel
    return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _cli_contract(root: Path, rel: str | None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "has_argparse": False,
        "required_options": [],
        "required_positionals": [],
        "dynamic_argument_definitions": 0,
    }
    if not rel:
        return out
    text = current._read_text(root / rel)
    try:
        tree = ast.parse(text, filename=rel)
    except (SyntaxError, ValueError):
        return out
    options: Set[str] = set()
    positionals: Set[str] = set()
    dynamic = 0
    has = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = current._call_name(node.func)
        if name.endswith("ArgumentParser"):
            has = True
        if not name.endswith("add_argument"):
            continue
        has = True
        if not node.args:
            dynamic += 1
            continue
        first = _literal(node.args[0])
        if not isinstance(first, str):
            dynamic += 1
            continue
        flags = [x for x in (_literal(a) for a in node.args) if isinstance(x, str)]
        kw = {x.arg: _literal(x.value) for x in node.keywords if x.arg}
        if first.startswith("-"):
            if kw.get("required") is True:
                options.add(next((x for x in flags if x.startswith("--")), first))
        elif kw.get("nargs") not in {"?", "*"}:
            positionals.add(first)
    out.update(
        has_argparse=has,
        required_options=sorted(options),
        required_positionals=sorted(positionals),
        dynamic_argument_definitions=dynamic,
    )
    return out


def _training(root: Path, name: str, rel: str | None) -> bool:
    if TRAINING_NAME.search(name):
        return True
    if not rel:
        return False
    text = current._read_text(root / rel)
    return bool(text and current._matches_any(text, current.TRAIN_PATTERNS))


def _profile_args(profile: Mapping[str, Any], name: str) -> List[str] | None:
    mapping = profile.get("console_script_args", {})
    if not isinstance(mapping, Mapping) or name not in mapping:
        return None
    value = mapping[name]
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    raise SystemExit(f"console_script_args[{name!r}] must be string/list/null")


def _ignored(profile: Mapping[str, Any], name: str) -> bool:
    raw = profile.get("ignore_console_scripts", []) or []
    return name in raw if isinstance(raw, Mapping) else name in {str(x) for x in raw}


def _inventory(root: Path, profile: Mapping[str, Any]) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for name, target in sorted(_read_scripts(root).items()):
        module, func = _target(target)
        rel = _source(root, module)
        contract = _cli_contract(root, rel)
        training = _training(root, name, rel)
        args = _profile_args(profile, name)
        ignored = _ignored(profile, name)
        default_ok = not contract["required_options"] and not contract["required_positionals"]
        entries.append({
            "name": name, "target": target, "module": module, "callable": func,
            "source": rel, "training_surface": training, "ignored": ignored,
            "explicit_args": args, "runnable_by_default": default_ok,
            "configured": bool(ignored or args is not None or default_ok), **contract,
        })
    training_entries = [x for x in entries if x["training_surface"]]
    return {
        "schema": CONSOLE_AUDIT_SCHEMA,
        "registered_script_count": len(entries),
        "registered_training_script_count": len(training_entries),
        "entries": entries,
        "training_entries": training_entries,
        "unresolved_targets": [x["name"] for x in training_entries if not x["source"] and not x["ignored"]],
        "unconfigured_training_entrypoints": [x["name"] for x in training_entries if not x["configured"] and not x["ignored"]],
    }


def _entry_code(module: str, func: str) -> str:
    return (
        "import sys;sys.path[:0]=['src','lib','python'];"
        f"from {module} import {func} as _entry;"
        "_r=_entry();raise SystemExit(_r if isinstance(_r,int) else 0)"
    )


def _jobs(root: Path, profile: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not bool(profile.get("auto_console_training_jobs", False)):
        return []
    result: List[Dict[str, Any]] = []
    for item in _inventory(root, profile)["training_entries"]:
        if item["ignored"] or not item["source"]:
            continue
        args = item["explicit_args"]
        if args is None:
            if not item["runnable_by_default"]:
                continue
            args = []
        meta: Dict[str, Any] = {
            "console_script": item["name"],
            "entrypoint_source": item["source"],
            "registry": "pyproject.toml:[project.scripts]",
        }
        custom = profile.get("console_script_metadata", {})
        if isinstance(custom, Mapping) and isinstance(custom.get(item["name"]), Mapping):
            meta.update(dict(custom[item["name"]]))
        if "depends_on" not in meta:
            deps = profile.get("console_script_default_depends_on", []) or []
            meta["depends_on"] = [str(x) for x in ([deps] if isinstance(deps, str) else deps)]
        record = {
            "id": f"console:{item['name']}",
            "command": [sys.executable, "-c", _entry_code(item["module"], item["callable"]), *args],
            "device_capable": True, "phase": "training", "family": "console-script", "repeat_index": 0,
            **meta,
        }
        result.append(record)
    return result


_ORIGINAL_JOBS = current._enhanced_job_records
_ORIGINAL_REPORT = current._enhanced_coverage_report
_ORIGINAL_PATHS = current._command_repo_paths


def _job_records(root: Path, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    return base._dedupe_jobs([*_ORIGINAL_JOBS(root, profile), *_jobs(root, profile)])


def _command_repo_paths(root: Path, jobs: Sequence[Dict[str, Any]]) -> Set[str]:
    paths = set(_ORIGINAL_PATHS(root, jobs))
    for job in jobs:
        rel = job.get("entrypoint_source")
        if rel and (root / str(rel)).is_file():
            paths.add(str(rel).replace("\\", "/"))
    return paths


def _coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    report = _ORIGINAL_REPORT(root, profile, jobs)
    inventory = _inventory(root, profile)
    compiled = sorted(str(x["console_script"]) for x in jobs if x.get("console_script"))
    scheduled = set(compiled)
    active = {x["name"] for x in inventory["training_entries"] if not x["ignored"]}
    expected = {
        x["name"] for x in inventory["training_entries"]
        if not x["ignored"] and x["source"] and (x["runnable_by_default"] or x["explicit_args"] is not None)
    }
    auto = bool(profile.get("auto_console_training_jobs", False))
    require_schedule = bool(profile.get("require_registered_training_scheduling", False))
    unresolved = sorted(set(inventory["unresolved_targets"]))
    unconfigured = sorted(set(inventory["unconfigured_training_entrypoints"]))
    missing = sorted(expected - scheduled) if auto else (sorted(expected) if require_schedule else [])
    strict = bool(profile.get("require_registered_training_entrypoints", True))
    console_ok = not unresolved and not unconfigured and not missing
    report.update({
        "console_registry_schema": CONSOLE_AUDIT_SCHEMA,
        "console_registry": inventory,
        "compiled_console_training_jobs": compiled,
        "unscheduled_registered_training_entrypoints": sorted(active - scheduled),
        "missing_console_job_materialization": missing,
        "strict_registered_training_entrypoints_pass": console_ok,
        "strict_controls": {
            **dict(report.get("strict_controls") or {}),
            "require_registered_training_entrypoints": strict,
            "require_registered_training_scheduling": require_schedule,
            "auto_console_training_jobs": auto,
        },
    })
    if strict and not console_ok:
        report["coverage_ok"] = False
    return report


def install() -> None:
    current._enhanced_job_records = _job_records
    current._enhanced_coverage_report = _coverage_report
    current._command_repo_paths = _command_repo_paths
