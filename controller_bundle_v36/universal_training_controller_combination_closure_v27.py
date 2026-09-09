#!/usr/bin/env python3
"""Fail-closed coverage for repository-authored scientific combinations.

Earlier controller generations prove that every declared scientific *member* is
reachable.  That is necessary but not sufficient: a repository can advertise
several concrete model/dataset/task/loss/etc. recipes while each individual name
appears somewhere in the central job universe and one real compatible recipe is
still absent.

v27 therefore inventories only combinations the repository itself declares.  It
does **not** form a Cartesian product.  Literal experiment/recipe/matrix rows in
Python and structured config files are treated as compatibility authority.  A
row is covered when one concrete compiled job represents all of its scientific
values together, or when the job directly selects the config file containing the
row.  Python matrix rows can additionally be covered by a reachable source that
actually iterates the matrix symbol.

This is an accounting layer only.  It contains no CPU/GPU/RAM/VRAM admission,
pause/resume, retry, OOM fallback, device selection, or process scheduling code.
Ready jobs continue to execute exclusively through the byte-pinned OPF_ADP
scheduler.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:  # Python 3.11+
    import tomllib
except Exception:  # pragma: no cover - older interpreters remain supported
    tomllib = None  # type: ignore[assignment]

import universal_training_controller_current as current
import universal_training_controller_registry_member_closure as registry_members
import universal_training_controller_workload_closure as workload

COMBINATION_CLOSURE_SCHEMA = 1
CONFIG_SUFFIXES = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}

_KEY_ALIASES = {
    "model": "model", "model_name": "model", "architecture": "model", "arch": "model",
    "backbone": "backbone", "encoder": "encoder", "decoder": "decoder", "head": "head",
    "learner": "learner", "trainer": "trainer",
    "loss": "loss", "loss_name": "loss", "objective": "loss", "criterion": "loss",
    "dataset": "dataset", "dataset_name": "dataset", "datamodule": "dataset",
    "data_module": "dataset", "data_source": "dataset", "corpus": "dataset", "benchmark": "dataset",
    "task": "task", "task_name": "task", "target": "task", "label_space": "task",
    "method": "method", "algorithm": "method", "strategy": "method", "approach": "method",
    "policy": "method", "agent": "method",
    "optimizer": "optimizer", "optimiser": "optimizer",
    "scheduler": "scheduler", "lr_scheduler": "scheduler",
    "sampler": "sampler", "augmentation": "augmentation", "transform": "augmentation",
    "preprocessor": "preprocessor", "tokenizer": "preprocessor", "feature": "feature",
    "ensemble": "ensemble", "fusion": "ensemble", "cascade": "ensemble",
    "stacker": "ensemble", "blender": "ensemble",
    "experiment": "pipeline", "pipeline": "pipeline", "workflow": "pipeline",
    "stage": "pipeline", "recipe": "pipeline",
    "environment": "environment", "domain": "environment", "scenario": "environment", "regime": "environment",
    "evaluator": "evaluation", "metric": "evaluation", "metrics": "evaluation", "scorer": "evaluation",
}
_PRIMARY_DIMENSIONS = {"model", "backbone", "encoder", "decoder", "head", "loss", "dataset", "task", "method", "ensemble", "pipeline", "environment"}
_MATRIX_NAME_RE = re.compile(r"(?:matrix|matrices|combination|combinations|combo|combos|recipe|recipes|experiment|experiments|suite|suites|variant|variants|workload|workloads|job|jobs|run|runs)", re.I)


def _norm_key(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _dimension(value: str) -> str | None:
    return _KEY_ALIASES.get(_norm_key(value))


def _scalar(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        return text if text else None
    return None


def _canonical_values(mapping: Mapping[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in mapping.items():
        dim = _dimension(str(key))
        scalar = _scalar(value)
        if dim and scalar is not None:
            # If aliases collide in one row, retain the first value only when the
            # aliases agree; conflicting aliases are not a single concrete tuple.
            prior = result.get(dim)
            if prior is None:
                result[dim] = scalar
            elif prior != scalar:
                return {}
    if len(result) < 2 or not (_PRIMARY_DIMENSIONS & set(result)):
        return {}
    return dict(sorted(result.items()))


def _fingerprint(values: Mapping[str, str]) -> str:
    return "|".join(f"{key}={values[key]}" for key in sorted(values))


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _rows_from_value(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _rows_from_value(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _rows_from_value(child)


def _python_combinations(path: Path, rel: str) -> list[Dict[str, Any]]:
    if path.suffix.lower() != ".py":
        return []
    text = workload._read(path)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return []
    rows: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        literal = _literal(node.value)
        if not isinstance(literal, (Mapping, list, tuple)):
            continue
        symbols = [target.id for target in targets if isinstance(target, ast.Name)]
        symbol = symbols[0] if symbols else ""
        # Generic local dictionaries are noisy.  Top-level scientific registries
        # are already contractual through v25/v26; matrix-ish names additionally
        # make nested literal recipe tables contractual here.
        strong_symbol = bool(symbol and (_MATRIX_NAME_RE.search(symbol) or workload.REGISTRY_RE.search(symbol)))
        for candidate in _rows_from_value(literal):
            values = _canonical_values(candidate)
            if not values:
                continue
            if not strong_symbol and len(values) < 3:
                continue
            fp = _fingerprint(values)
            key = (symbol, fp, int(getattr(node, "lineno", 0)))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "path": rel,
                "line": int(getattr(node, "lineno", 0)),
                "symbol": symbol,
                "values": values,
                "fingerprint": fp,
                "surface_kind": "python_literal_matrix",
            })
    return rows


def _walk_structured(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_structured(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_structured(child)


def _strip_scalar(text: str) -> Any:
    text = text.strip().rstrip(",")
    if not text or text in {"{}", "[]", "|", ">"}:
        return None
    if (text[0:1] == text[-1:] and text[0:1] in {"'", '"'}) and len(text) >= 2:
        return text[1:-1]
    low = text.lower()
    if low in {"true", "false", "null", "none", "~"}:
        return None
    if text.startswith("[") or text.startswith("{"):
        return None
    return text.split(" #", 1)[0].strip()


def _yaml_like_combinations(text: str, rel: str) -> list[Dict[str, Any]]:
    """Conservative YAML/INI fallback without requiring third-party parsers.

    Only contiguous scalar mapping rows are emitted.  Lists of selectors are not
    expanded, so this function cannot invent compatibility products.
    """
    rows: list[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    current_indent: int | None = None
    current_line = 0

    def emit() -> None:
        nonlocal current
        values = _canonical_values(current)
        if values:
            rows.append({
                "path": rel,
                "line": current_line,
                "symbol": "",
                "values": values,
                "fingerprint": _fingerprint(values),
                "surface_kind": "structured_config",
            })
        current = {}

    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        body = raw.strip()
        new_item = body.startswith("-")
        if new_item:
            body = body[1:].strip()
            if current and current_indent is not None and indent <= current_indent:
                emit()
            if not current:
                current_indent = indent
                current_line = number
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]\s*(.*?)\s*$", body)
        if not match:
            continue
        key, raw_value = match.group(1), match.group(2)
        if current and current_indent is not None and indent < current_indent:
            emit()
            current_indent = indent
            current_line = number
        value = _strip_scalar(raw_value)
        if _dimension(key) and value is not None:
            current[key] = value
            if current_indent is None:
                current_indent = indent
                current_line = number
    if current:
        emit()
    return rows


def _config_combinations(path: Path, rel: str) -> list[Dict[str, Any]]:
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        return []
    text = workload._read(path)
    if not text:
        return []
    payload: Any = None
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
    elif suffix == ".toml" and tomllib is not None:
        try:
            payload = tomllib.loads(text)
        except Exception:
            payload = None

    rows: list[Dict[str, Any]] = []
    if payload is not None:
        for candidate in _walk_structured(payload):
            values = _canonical_values(candidate)
            if not values:
                continue
            rows.append({
                "path": rel,
                "line": 0,
                "symbol": "",
                "values": values,
                "fingerprint": _fingerprint(values),
                "surface_kind": "structured_config",
            })
    else:
        rows.extend(_yaml_like_combinations(text, rel))

    dedup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        dedup.setdefault(str(row["fingerprint"]), row)
    return list(dedup.values())


def _inventory_surfaces(root: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in workload._iter_files(root):
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        candidates = _python_combinations(path, rel)
        if path.suffix.lower() in CONFIG_SUFFIXES:
            # Config files become combination-contract surfaces only when the
            # existing workload layer already considers their location/name
            # scientifically plausible, or when the row has >=3 dimensions.
            candidates += _config_combinations(path, rel)
        for row in candidates:
            values = row.get("values") if isinstance(row.get("values"), Mapping) else {}
            if path.suffix.lower() in CONFIG_SUFFIXES and len(values) == 2 and not workload.CONFIG_HINT_RE.search(rel):
                continue
            key = (rel, str(row.get("symbol") or ""), str(row.get("fingerprint") or ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return sorted(rows, key=lambda row: (str(row["path"]), int(row.get("line") or 0), str(row["fingerprint"])))


def _job_covers_values(job: Mapping[str, Any], values: Mapping[str, str]) -> bool:
    normalized, corpus = registry_members._job_strings(job)
    return all(registry_members._member_named(value, normalized, corpus) for value in values.values())


def _direct_config_jobs(root: Path, jobs: Sequence[Mapping[str, Any]]) -> Dict[str, list[str]]:
    result: Dict[str, list[str]] = {}
    for job in jobs:
        job_id = str(job.get("id") or "")
        paths = workload._command_paths(root, [job])
        for rel in paths:
            result.setdefault(rel, []).append(job_id)
    return result


def _iterating_sources(root: Path, reachable: Iterable[str]) -> Dict[str, set[str]]:
    result: Dict[str, set[str]] = {}
    for rel in reachable:
        path = root / rel
        if path.is_file() and path.suffix.lower() == ".py":
            result[rel] = registry_members._looped_symbols(path)
    return result


def _combination_inventory(root: Path, jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    surfaces = _inventory_surfaces(root)
    reach = current._reachability(root, jobs)
    reachable = {str(value) for value in (reach.get("reachable_sources") or [])}
    per_job = reach.get("per_job") if isinstance(reach.get("per_job"), Mapping) else {}
    direct_configs = _direct_config_jobs(root, jobs)
    looped = _iterating_sources(root, reachable)

    rows: list[Dict[str, Any]] = []
    uncovered: list[Dict[str, Any]] = []
    for surface in surfaces:
        path = str(surface["path"])
        symbol = str(surface.get("symbol") or "")
        values = dict(surface.get("values") or {})
        covering: set[str] = set(direct_configs.get(path, []))
        value_jobs: set[str] = set()
        iterator_jobs: set[str] = set()
        for job in jobs:
            job_id = str(job.get("id") or "")
            if _job_covers_values(job, values):
                covering.add(job_id)
                value_jobs.add(job_id)
            reachable_for_job: set[str] = set()
            row = per_job.get(job_id) if isinstance(per_job, Mapping) else None
            if isinstance(row, Mapping):
                reachable_for_job = {str(value) for value in (row.get("reachable") or [])}
            if symbol and path in reachable_for_job:
                # The matrix symbol may be iterated either in its own source or
                # in another source reachable from the same job.
                if any(symbol in looped.get(source, set()) for source in reachable_for_job):
                    covering.add(job_id)
                    iterator_jobs.add(job_id)

        row = dict(surface)
        row.update({
            "covering_jobs": sorted(covering),
            "value_matching_jobs": sorted(value_jobs),
            "direct_config_jobs": sorted(set(direct_configs.get(path, []))),
            "iterator_jobs": sorted(iterator_jobs),
            "complete": bool(covering),
        })
        rows.append(row)
        if not covering:
            uncovered.append(row)

    return {
        "schema": COMBINATION_CLOSURE_SCHEMA,
        "declared_combination_count": len(rows),
        "declared_combinations": rows,
        "uncovered_combination_count": len(uncovered),
        "uncovered_declared_combinations": uncovered,
        "complete": not uncovered,
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
        inventory = _combination_inventory(Path(root), jobs)
        require = bool(profile.get("require_declared_combination_accounting", profile.get("strict_coverage", True)))
        controls = dict(report.get("strict_controls") or {})
        controls["require_declared_combination_accounting"] = require
        report.update({
            "combination_closure_schema": COMBINATION_CLOSURE_SCHEMA,
            "declared_combination_inventory": inventory,
            "uncovered_declared_combinations": inventory["uncovered_declared_combinations"],
            "strict_declared_combination_pass": bool(inventory["complete"]),
            "strict_controls": controls,
        })
        if require and not inventory["complete"]:
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["COMBINATION_CLOSURE_SCHEMA", "_combination_inventory", "install"]
