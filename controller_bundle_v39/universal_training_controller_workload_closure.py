#!/usr/bin/env python3
"""Fail-closed workload-surface accounting beyond learner source files.

The existing semantic inventory proves that concrete learner/training code is
reachable from central-runner jobs.  This layer extends the same idea to the
repository-authored *experiment universe*: strong model/dataset/task registries
and training recipe configs must also be reachable from at least one compiled
job.  It does not create arbitrary Cartesian products; concrete job records and
repo-authored configs remain the compatibility authority.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller_current as current

WORKLOAD_CLOSURE_SCHEMA = 1
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
SKIP_DIRS = {
    ".git", ".training_control", "__pycache__", ".venv", "venv", "env", "node_modules",
    "results", "result", "outputs", "output", "checkpoints", "artifacts", "dist", "build",
    "docs", "doc", "tests", "test",
}
REGISTRY_RE = re.compile(
    r"(?:^|_)(MODEL|MODELS|ARCH|ARCHS|ARCHITECTURE|ARCHITECTURES|DATASET|DATASETS|"
    r"TASK|TASKS|METHOD|METHODS|ALGORITHM|ALGORITHMS|EXPERIMENT|EXPERIMENTS|PIPELINE|PIPELINES|"
    r"BENCHMARK|BENCHMARKS|ENVIRONMENT|ENVIRONMENTS|DOMAIN|DOMAINS)(?:_|$)", re.I,
)
CONFIG_HINT_RE = re.compile(r"(?:config|recipe|experiment|train|training|model|dataset|task|benchmark|method|algorithm)", re.I)
TRAINING_KEYS = {
    "training", "trainer", "optimizer", "learning_rate", "lr", "epochs", "max_epochs",
    "batch_size", "loss", "scheduler", "checkpoint", "early_stopping", "patience",
}
IDENTITY_KEYS = {
    "model", "model_name", "architecture", "arch", "dataset", "dataset_name", "task", "task_name",
    "objective", "benchmark", "method", "algorithm", "environment", "domain",
}


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name.lower() not in SKIP_DIRS)
        base = Path(dirpath)
        for filename in sorted(filenames):
            yield base / filename


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _registry_findings(path: Path, rel: str) -> list[Dict[str, Any]]:
    if path.suffix.lower() != ".py":
        return []
    text = _read(path)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return []
    rows: list[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            continue
        for target in targets:
            if not isinstance(target, ast.Name) or not REGISTRY_RE.search(target.id):
                continue
            members: list[str] = []
            if isinstance(value, ast.Dict):
                for key in value.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, (str, int, float)):
                        members.append(str(key.value))
            else:
                for item in value.elts:
                    if isinstance(item, ast.Constant) and isinstance(item.value, (str, int, float)):
                        members.append(str(item.value))
            rows.append({
                "path": rel,
                "symbol": target.id,
                "line": int(getattr(node, "lineno", 0)),
                "members": sorted(set(members)),
            })
    return rows


def _config_keys(path: Path) -> set[str]:
    text = _read(path)
    if not text:
        return set()
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return {str(key).strip().lower() for key in payload.keys()}
    keys = set()
    for match in re.finditer(r"(?mi)^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]", text):
        keys.add(match.group(1).split(".", 1)[0].strip().lower())
    return keys


def _strong_training_config(path: Path, rel: str) -> Dict[str, Any] | None:
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        return None
    # Configs become contractual only when both location/name and content signal
    # an experiment recipe. This avoids treating arbitrary application JSON/YAML
    # as training work.
    if not CONFIG_HINT_RE.search(rel):
        return None
    keys = _config_keys(path)
    training_hits = sorted(keys & TRAINING_KEYS)
    identity_hits = sorted(keys & IDENTITY_KEYS)
    if len(training_hits) < 1 or len(identity_hits) < 1:
        return None
    return {
        "path": rel,
        "sha256": _sha256(path),
        "training_keys": training_hits,
        "identity_keys": identity_hits,
    }


def _command_paths(root: Path, jobs: Sequence[Mapping[str, Any]]) -> set[str]:
    result: set[str] = set()
    for job in jobs:
        for token in job.get("command", []) or []:
            text = str(token)
            if "=" in text and text.startswith("--"):
                text = text.split("=", 1)[1]
            try:
                path = Path(text)
                candidate = path if path.is_absolute() else root / path
                candidate = candidate.resolve()
                rel = candidate.relative_to(root.resolve()).as_posix()
                if candidate.is_file():
                    result.add(rel)
            except Exception:
                continue
        for key in ("recipe_config", "config", "config_path", "config_file"):
            value = job.get(key)
            if value:
                try:
                    candidate = (root / str(value)).resolve()
                    if candidate.is_file():
                        result.add(candidate.relative_to(root.resolve()).as_posix())
                except Exception:
                    pass
    return result


def _source_referenced_configs(root: Path, reachable_sources: Iterable[str], config_paths: set[str]) -> set[str]:
    result: set[str] = set()
    by_name: Dict[str, list[str]] = {}
    for rel in config_paths:
        by_name.setdefault(Path(rel).name, []).append(rel)
    for source in reachable_sources:
        path = root / source
        if not path.is_file():
            continue
        text = _read(path)
        if not text:
            continue
        for rel in config_paths:
            if rel in text:
                result.add(rel)
        for name, rels in by_name.items():
            if name in text and len(rels) == 1:
                result.add(rels[0])
    return result


def _inventory(root: Path, jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    registries: list[Dict[str, Any]] = []
    configs: list[Dict[str, Any]] = []
    for path in _iter_files(root):
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        registries.extend(_registry_findings(path, rel))
        row = _strong_training_config(path, rel)
        if row is not None:
            configs.append(row)

    reach = current._reachability(root, jobs)
    reachable_sources = {str(x) for x in (reach.get("reachable_sources") or [])}
    direct_paths = _command_paths(root, jobs)
    config_paths = {str(row["path"]) for row in configs}
    referenced_configs = _source_referenced_configs(root, reachable_sources, config_paths)
    reachable_configs = config_paths & (direct_paths | referenced_configs)
    registry_paths = {str(row["path"]) for row in registries}
    reachable_registries = registry_paths & reachable_sources

    return {
        "schema": WORKLOAD_CLOSURE_SCHEMA,
        "registry_surfaces": sorted(registries, key=lambda row: (str(row["path"]), str(row["symbol"]))),
        "training_config_surfaces": sorted(configs, key=lambda row: str(row["path"])),
        "reachable_registry_paths": sorted(reachable_registries),
        "reachable_training_config_paths": sorted(reachable_configs),
        "unaccounted_registry_paths": sorted(registry_paths - reachable_registries),
        "unaccounted_training_config_paths": sorted(config_paths - reachable_configs),
        "direct_job_file_paths": sorted(direct_paths),
        "source_referenced_training_configs": sorted(referenced_configs),
        "reachability": reach,
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs):
        report = original_report(root, profile, jobs)
        inventory = _inventory(Path(root), jobs)
        require = bool(profile.get("require_workload_surface_accounting", profile.get("strict_coverage", True)))
        config_matrix = profile.get("config_matrix_autofill") if isinstance(profile.get("config_matrix_autofill"), Mapping) else {}
        ambiguous = dict(config_matrix.get("ambiguous_config_directories") or {}) if isinstance(config_matrix, Mapping) else {}
        unaccounted_registries = list(inventory["unaccounted_registry_paths"])
        unaccounted_configs = list(inventory["unaccounted_training_config_paths"])
        closure_ok = not unaccounted_registries and not unaccounted_configs and not ambiguous
        controls = dict(report.get("strict_controls") or {})
        controls["require_workload_surface_accounting"] = require
        report.update({
            "workload_closure_schema": WORKLOAD_CLOSURE_SCHEMA,
            "workload_surface_inventory": inventory,
            "unaccounted_workload_registry_paths": unaccounted_registries,
            "unaccounted_training_config_paths": unaccounted_configs,
            "ambiguous_training_config_directories": ambiguous,
            "strict_workload_surface_pass": closure_ok,
            "strict_controls": controls,
        })
        if require and not closure_ok:
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["WORKLOAD_CLOSURE_SCHEMA", "_inventory", "install"]
