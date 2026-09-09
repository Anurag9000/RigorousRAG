#!/usr/bin/env python3
"""Universal controller v30: declarative scientific source closure.

v25-v29 already close explicit scientific registries, selectors, component
configs, repository-authored combinations and the extended family/type/variant/
capability/regularizer ontology. v30 closes declaration forms that otherwise
remain easy to miss in real repositories:

* scientific Literal[...] choices used directly in function/method parameters
  or class fields, including nested Annotated/Optional/Union forms;
* Enum choices referenced through scientific annotations even when the Enum
  class itself has a generic name;
* statically recoverable registry mutation/merge forms (update, extend, append,
  add, setdefault, |=, unpacking and local aliases);
* structured JSON/TOML/YAML scientific choice lists/maps;
* Hydra-style conf/<component>/... config groups; and
* concrete implementation classes in scientific component modules that are
  never referenced by any centrally reachable source/config/job.

The layer is static, conservative and fail-closed. It never imports repository
application code, never invents a Cartesian product, and contains no resource
scheduler. Ready jobs continue to run only through the exact byte-pinned
OPF_ADP scheduler inherited by the surrounding controller stack.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

try:
    import tomllib
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

import universal_training_controller_current as current
import universal_training_controller_registry_member_closure as registry_members
import universal_training_controller_selector_closure_v26 as selectors
import universal_training_controller_workload_closure as workload

DECLARATION_CLOSURE_SCHEMA = 1
CONFIG_SUFFIXES = {".json", ".toml", ".yaml", ".yml", ".ini", ".cfg"}
_STRONG_BASES = {
    "Module", "LightningModule", "DataModule", "LightningDataModule",
    "Dataset", "IterableDataset", "Metric", "Optimizer", "LRScheduler",
    "_LRScheduler", "Env", "BaseEstimator", "TransformerMixin",
}
_BEHAVIOR_METHODS = {
    "forward", "training_step", "fit", "partial_fit", "predict",
    "__getitem__", "__iter__", "step", "update", "compute", "loss",
}
_CONFIG_ROOT_PARTS = {
    "conf", "config", "configs", "recipe", "recipes", "experiment", "experiments",
}


def _call_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        left = _call_name(node.value)
        return f"{left}.{node.attr}" if left else node.attr
    if isinstance(node, ast.Subscript):
        return _call_name(node.value)
    return ""


def _root_name(node: ast.AST | None) -> str:
    name = _call_name(node)
    return name.split(".")[-1] if name else ""


def _scalar(node: ast.AST | None) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)) and not isinstance(node.value, bool):
        return [str(node.value)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
        value = node.operand.value
        if isinstance(node.op, ast.USub):
            value = -value
        return [str(value)]
    return []


def _resolve_members(node: ast.AST | None, aliases: Mapping[str, Sequence[str]]) -> list[str]:
    if node is None:
        return []
    literal = _scalar(node)
    if literal:
        return literal
    if isinstance(node, ast.Name):
        return list(aliases.get(node.id, ()))
    if isinstance(node, ast.Starred):
        return _resolve_members(node.value, aliases)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_resolve_members(item, aliases))
        return out
    if isinstance(node, ast.Dict):
        out: list[str] = []
        for key, value in zip(node.keys, node.values):
            if key is None:
                out.extend(_resolve_members(value, aliases))
            else:
                out.extend(_resolve_members(key, aliases))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.BitOr)):
        return _resolve_members(node.left, aliases) + _resolve_members(node.right, aliases)
    if isinstance(node, ast.Call) and _root_name(node.func).lower() in {"list", "tuple", "set", "frozenset", "dict"} and node.args:
        return _resolve_members(node.args[0], aliases)
    return []


def _local_aliases(tree: ast.AST) -> Dict[str, list[str]]:
    assignments: list[tuple[str, ast.AST | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                assignments.append((target.id, node.value))
    aliases: Dict[str, list[str]] = {}
    for _ in range(max(2, min(16, len(assignments) + 1))):
        changed = False
        for name, value in assignments:
            members = sorted(set(_resolve_members(value, aliases)))
            if members and members != aliases.get(name):
                aliases[name] = members
                changed = True
        if not changed:
            break
    return aliases


def _enum_members(tree: ast.AST, aliases: Mapping[str, Sequence[str]]) -> Dict[str, list[str]]:
    result: Dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(_root_name(base) in {"Enum", "StrEnum", "IntEnum", "Flag", "IntFlag"} for base in node.bases):
            continue
        members: list[str] = []
        for item in node.body:
            if not isinstance(item, (ast.Assign, ast.AnnAssign)):
                continue
            targets = item.targets if isinstance(item, ast.Assign) else [item.target]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    values = _resolve_members(item.value, aliases)
                    members.extend(values or [target.id])
        if members:
            result[node.name] = sorted(set(members))
    return result


def _annotation_members(node: ast.AST | None, aliases: Mapping[str, Sequence[str]], enums: Mapping[str, Sequence[str]]) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Name) and node.id in enums:
        return list(enums[node.id])
    if isinstance(node, ast.Attribute) and node.attr in enums:
        return list(enums[node.attr])
    if isinstance(node, ast.Subscript):
        if _root_name(node.value).lower() == "literal":
            return _resolve_members(node.slice, aliases)
        return _annotation_members(node.slice, aliases, enums)
    if isinstance(node, ast.Tuple):
        out: list[str] = []
        for item in node.elts:
            out.extend(_annotation_members(item, aliases, enums))
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_members(node.left, aliases, enums) + _annotation_members(node.right, aliases, enums)
    return []


def _merge_store(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    store: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        item = store.setdefault(symbol, {
            "path": str(row.get("path") or ""), "symbol": symbol,
            "line": int(row.get("line") or 0), "members": set(),
            "enumerable": bool(row.get("enumerable", True)),
            "assignment_kind": str(row.get("assignment_kind") or "declaration"),
            "surface_kinds": set(),
        })
        item["members"].update(str(value) for value in (row.get("members") or []) if str(value).strip())
        item["enumerable"] = bool(item["enumerable"] and row.get("enumerable", True))
        item["surface_kinds"].update(str(value) for value in (row.get("surface_kinds") or []))
        if row.get("assignment_kind"):
            item["surface_kinds"].add(str(row.get("assignment_kind")))
    return store


def _add_row(store: Dict[str, Dict[str, Any]], *, rel: str, symbol: str, line: int, members: Iterable[str], kind: str) -> None:
    values = {str(value) for value in members if str(value).strip()}
    if not values:
        return
    row = store.setdefault(symbol, {
        "path": rel, "symbol": symbol, "line": int(line or 0), "members": set(),
        "enumerable": True, "assignment_kind": kind, "surface_kinds": set(),
    })
    row["members"].update(values)
    row["surface_kinds"].add(kind)


def _python_findings(path: Path, rel: str) -> list[Dict[str, Any]]:
    text = workload._read(path)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return selectors._selector_registry_findings(path, rel)
    store = _merge_store(selectors._selector_registry_findings(path, rel))
    aliases = _local_aliases(tree)
    enums = _enum_members(tree, aliases)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and selectors._semantic_name(target.id):
                    _add_row(store, rel=rel, symbol=target.id, line=int(getattr(node, "lineno", 0)), members=_resolve_members(node.value, aliases), kind="static_alias_or_merge")
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and selectors._semantic_name(node.target.id):
            _add_row(store, rel=rel, symbol=node.target.id, line=int(getattr(node, "lineno", 0)), members=_resolve_members(node.value, aliases), kind="augmented_registry_merge")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = _root_name(node.func.value)
            tail = node.func.attr
            if receiver and selectors._semantic_name(receiver) and node.args:
                if tail in {"update", "extend"}:
                    _add_row(store, rel=rel, symbol=receiver, line=int(getattr(node, "lineno", 0)), members=_resolve_members(node.args[0], aliases), kind=f"registry_{tail}")
                elif tail in {"append", "add", "setdefault"}:
                    _add_row(store, rel=rel, symbol=receiver, line=int(getattr(node, "lineno", 0)), members=_resolve_members(node.args[0], aliases), kind=f"registry_{tail}")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if node.args.vararg is not None:
                args.append(node.args.vararg)
            if node.args.kwarg is not None:
                args.append(node.args.kwarg)
            for arg in args:
                if selectors._semantic_name(arg.arg):
                    _add_row(store, rel=rel, symbol=f"annotation:{node.name}:{arg.arg}", line=int(getattr(arg, "lineno", node.lineno)), members=_annotation_members(arg.annotation, aliases, enums), kind="parameter_annotation_choices")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and selectors._semantic_name(node.target.id):
            _add_row(store, rel=rel, symbol=f"annotation:field:{node.target.id}", line=int(getattr(node, "lineno", 0)), members=_annotation_members(node.annotation, aliases, enums), kind="field_annotation_choices")

    return [{
        "path": rel, "symbol": symbol, "line": int(row.get("line") or 0),
        "members": sorted(row["members"]), "enumerable": bool(row.get("enumerable", True)),
        "assignment_kind": str(row.get("assignment_kind") or "declaration"),
        "surface_kinds": sorted(row.get("surface_kinds") or []),
    } for symbol, row in sorted(store.items())]


def _declaration_findings(path: Path, rel: str) -> list[Dict[str, Any]]:
    return _python_findings(path, rel) if path.suffix.lower() == ".py" else []


def _config_candidate(rel: str) -> bool:
    parts = {part.lower() for part in Path(rel).parts}
    return bool(parts & _CONFIG_ROOT_PARTS or selectors._COMPONENT_DIR_RE.search(rel.lower()) or workload.CONFIG_HINT_RE.search(rel))


def _structured_payload(path: Path) -> Any:
    text = workload._read(path)
    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except Exception:
            return None
    if path.suffix.lower() == ".toml" and tomllib is not None:
        try:
            return tomllib.loads(text)
        except Exception:
            return None
    return None


def _structured_choice_rows(value: Any, rel: str, prefix: str = "") -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if not isinstance(value, Mapping):
        return rows
    for key, child in value.items():
        key_text = str(key)
        full = prefix + ("." if prefix else "") + key_text
        if selectors._semantic_name(key_text):
            members: list[str] = []
            if isinstance(child, (list, tuple, set)):
                members = [str(item) for item in child if isinstance(item, (str, int, float)) and not isinstance(item, bool)]
            elif isinstance(child, Mapping):
                members = [str(item) for item in child.keys() if isinstance(item, (str, int, float)) and not isinstance(item, bool)]
            if members:
                rows.append({"path": rel, "symbol": f"config:{full}", "members": sorted(set(members)), "kind": "structured_choices"})
        if isinstance(child, Mapping):
            rows.extend(_structured_choice_rows(child, rel, full))
    return rows


def _yaml_choice_rows(path: Path, rel: str) -> list[Dict[str, Any]]:
    text = workload._read(path)
    rows: list[Dict[str, Any]] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        match = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(.*?)\s*$", raw)
        index += 1
        if not match or not selectors._semantic_name(match.group(2)):
            continue
        indent = len(match.group(1)); key = match.group(2); raw_value = match.group(3)
        members: list[str] = []
        if raw_value.startswith("[") and raw_value.endswith("]"):
            for token in raw_value[1:-1].split(","):
                value = token.strip().strip("'\"")
                if value and re.fullmatch(r"[A-Za-z0-9_.:/+\-]+", value):
                    members.append(value)
        elif not raw_value:
            cursor = index
            while cursor < len(lines):
                child = lines[cursor]
                child_indent = len(child) - len(child.lstrip(" "))
                if child.strip() and child_indent <= indent:
                    break
                item = re.match(r"^\s*-\s*([A-Za-z0-9_.:/+\-]+)\s*(?:#.*)?$", child)
                if item:
                    members.append(item.group(1))
                cursor += 1
        if members:
            rows.append({"path": rel, "symbol": f"config:{key}", "members": sorted(set(members)), "kind": "yaml_choice_list"})
    return rows


def _structured_choices(root: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in workload._iter_files(root):
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        if not _config_candidate(rel):
            continue
        payload = _structured_payload(path)
        found = _structured_choice_rows(payload, rel) if payload is not None else []
        if payload is None and path.suffix.lower() in {".yaml", ".yml"}:
            found = _yaml_choice_rows(path, rel)
        for row in found:
            key = (str(row["path"]), str(row["symbol"]))
            if key not in seen:
                seen.add(key); rows.append(row)
    return sorted(rows, key=lambda row: (str(row["path"]), str(row["symbol"])))


def _enhanced_component_configs(root: Path) -> list[str]:
    original = getattr(_enhanced_component_configs, "_original", None)
    rows = set(original(root) if callable(original) else [])
    for path in workload._iter_files(root):
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        parts = [part.lower() for part in Path(rel).parts]
        if "conf" in parts and selectors._COMPONENT_DIR_RE.search(rel.lower()):
            rows.add(rel)
    return sorted(rows)


def _method_executable(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = list(node.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    for statement in body:
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is Ellipsis:
            continue
        return True
    return False


def _concrete_component_classes(root: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in workload._iter_files(root):
        if path.suffix.lower() != ".py":
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        text = workload._read(path)
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError:
            continue
        in_component_dir = bool(selectors._COMPONENT_DIR_RE.search(rel.lower()))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or node.name.startswith(("Base", "Abstract", "_")):
                continue
            bases = {_root_name(base) for base in node.bases}
            methods = {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and _method_executable(item)}
            strong_base = bool(bases & _STRONG_BASES)
            behavioral = bool(methods & _BEHAVIOR_METHODS)
            if not (strong_base or (in_component_dir and behavioral)):
                continue
            rows.append({
                "path": rel, "class": node.name, "line": int(node.lineno),
                "bases": sorted(value for value in bases if value),
                "behavior_methods": sorted(methods & _BEHAVIOR_METHODS),
                "component_directory": in_component_dir,
            })
    return sorted(rows, key=lambda row: (str(row["path"]), int(row["line"]), str(row["class"])))


def _flatten(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key); yield from _flatten(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _flatten(item)
    elif value is not None:
        yield str(value)


def _identifier_present(name: str, text: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", text))


def _class_inventory(root: Path, jobs: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> Dict[str, Any]:
    declarations = _concrete_component_classes(root)
    reach = current._reachability(root, jobs)
    reachable = {str(value) for value in (reach.get("reachable_sources") or [])}
    workload_inventory = report.get("workload_surface_inventory") if isinstance(report.get("workload_surface_inventory"), Mapping) else {}
    config_paths = {str(value) for value in list(workload_inventory.get("reachable_training_config_paths") or []) + list(report.get("reachable_scientific_component_config_paths") or [])}
    texts: Dict[str, str] = {}
    for rel in sorted(reachable | config_paths):
        path = root / rel
        if path.is_file():
            texts[rel] = workload._read(path)
    job_corpus = "\n".join(value for job in jobs for value in _flatten(job))
    rows: list[Dict[str, Any]] = []
    missing: list[Dict[str, Any]] = []
    for declaration in declarations:
        rel = str(declaration["path"]); name = str(declaration["class"])
        evidence: list[str] = []
        if _identifier_present(name, job_corpus):
            evidence.append("compiled_job")
        own = workload._read(root / rel)
        if rel in reachable and len(re.findall(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", own)) > 1:
            evidence.append("reachable_same_module_use")
        for source, text in texts.items():
            if source != rel and _identifier_present(name, text):
                evidence.append(f"reachable_reference:{source}"); break
        row = dict(declaration)
        row.update({"reachable_source": rel in reachable, "reference_evidence": evidence, "accounted": bool(rel in reachable and evidence)})
        rows.append(row)
        if not row["accounted"]:
            missing.append(row)
    return {"schema": DECLARATION_CLOSURE_SCHEMA, "concrete_scientific_declarations": rows, "unaccounted_concrete_scientific_declarations": missing, "complete": not missing}


def _choice_inventory(root: Path, jobs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = _structured_choices(root)
    direct = workload._command_paths(root, jobs)
    corpus = "\n".join(value for job in jobs for value in _flatten(job))
    has_all = any(registry_members._all_selector(job) for job in jobs)
    missing: list[Dict[str, Any]] = []
    enriched: list[Dict[str, Any]] = []
    for row in rows:
        members = [str(value) for value in row.get("members") or []]
        explicit = [member for member in members if registry_members._member_named(member, set(), corpus.lower())]
        covered = str(row["path"]) in direct or has_all or len(explicit) == len(members)
        item = dict(row); item.update({"explicitly_named_members": explicit, "direct_config_selected": str(row["path"]) in direct, "all_selector_present": has_all, "accounted": covered})
        enriched.append(item)
        if not covered:
            missing.append(item)
    return {"schema": DECLARATION_CLOSURE_SCHEMA, "structured_scientific_choices": enriched, "unaccounted_structured_scientific_choices": missing, "complete": not missing}


def install_primitives() -> None:
    workload._registry_findings = _declaration_findings
    if not hasattr(selectors, "_iter_component_configs_v26"):
        selectors._iter_component_configs_v26 = selectors._iter_component_configs  # type: ignore[attr-defined]
    _enhanced_component_configs._original = selectors._iter_component_configs_v26  # type: ignore[attr-defined]
    selectors._iter_component_configs = _enhanced_component_configs


def install_contract() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
        classes = _class_inventory(Path(root), jobs, report)
        choices = _choice_inventory(Path(root), jobs)
        require = bool(profile.get("require_declarative_scientific_source_accounting", profile.get("strict_coverage", True)))
        controls = dict(report.get("strict_controls") or {})
        controls["require_declarative_scientific_source_accounting"] = require
        complete = bool(classes["complete"] and choices["complete"])
        report.update({
            "declaration_closure_schema": DECLARATION_CLOSURE_SCHEMA,
            "declarative_scientific_class_inventory": classes,
            "structured_scientific_choice_inventory": choices,
            "unaccounted_concrete_scientific_declarations": classes["unaccounted_concrete_scientific_declarations"],
            "unaccounted_structured_scientific_choices": choices["unaccounted_structured_scientific_choices"],
            "strict_declarative_scientific_source_pass": complete,
            "strict_controls": controls,
        })
        if require and not complete:
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = ["DECLARATION_CLOSURE_SCHEMA", "install_primitives", "install_contract"]
