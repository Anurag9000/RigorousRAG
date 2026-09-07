#!/usr/bin/env python3
"""Fail-closed scientific selector/config closure for universal controller v26.

v25 accounts for explicit repository registries and full-recipe training configs.
Real repositories also declare scientific choices through CLI ``choices``, Enum or
Literal aliases, decorator/call registration, registry subscript assignment,
branch/match dispatch and component config groups.  Those surfaces are equally
contractual: if a repository advertises a model/loss/dataset/task/algorithm/etc.
choice, the centralized job universe must account for it.

This module is intentionally static and conservative.  It does not import or run
repository code, does not invent Cartesian products, and contains no resource
scheduling logic.  Repository-authored trainers/configs/catalogs remain the
compatibility authority; ready jobs still execute only through the literal pinned
OPF_ADP scheduler.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller_current as current
import universal_training_controller_registry_member_closure as registry_members
import universal_training_controller_scientific_surface_v25 as scientific_v25
import universal_training_controller_workload_closure as workload

SELECTOR_CLOSURE_SCHEMA = 1
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}

# Names are normalized through CamelCase -> snake_case before this regex is used.
_SELECTOR_TOKEN_RE = re.compile(
    r"(?:^|_)(?:model|models|arch|archs|architecture|architectures|backbone|backbones|"
    r"encoder|encoders|decoder|decoders|head|heads|learner|learners|trainer|trainers|"
    r"loss|losses|objective|objectives|criterion|criteria|"
    r"dataset|datasets|datamodule|datamodules|data_module|data_modules|data_source|data_sources|"
    r"corpus|corpora|benchmark|benchmarks|task|tasks|target|targets|label_space|label_spaces|"
    r"method|methods|algorithm|algorithms|strategy|strategies|approach|approaches|"
    r"policy|policies|agent|agents|optimizer|optimizers|optimiser|optimisers|"
    r"scheduler|schedulers|lr_scheduler|lr_schedulers|sampler|samplers|"
    r"augmentation|augmentations|transform|transforms|preprocessor|preprocessors|"
    r"tokenizer|tokenizers|feature|features|ensemble|ensembles|fusion|fusions|"
    r"cascade|cascades|stacker|stackers|blender|blenders|experiment|experiments|"
    r"pipeline|pipelines|workflow|workflows|stage|stages|recipe|recipes|"
    r"environment|environments|domain|domains|scenario|scenarios|regime|regimes|"
    r"evaluator|evaluators|metric|metrics|scorer|scorers)(?:_|$)",
    re.I,
)
_COMPONENT_DIR_RE = re.compile(
    r"(?:^|/)(?:models?|architectures?|archs?|backbones?|encoders?|decoders?|heads?|losses?|"
    r"objectives?|criteria|datasets?|datamodules?|data_modules?|corpora|benchmarks?|tasks?|targets?|"
    r"methods?|algorithms?|strategies|approaches|policies|agents|optimizers?|optimisers?|schedulers?|"
    r"samplers?|augmentations?|transforms?|preprocessors?|tokenizers?|features?|ensembles?|fusions?|"
    r"cascades?|stackers?|blenders?|experiments?|pipelines?|workflows?|stages?|recipes?|"
    r"environments?|domains?|scenarios?|regimes?|evaluators?|metrics?|scorers?)(?:/|$)",
    re.I,
)
_REGISTER_RE = re.compile(r"(?:^|_)(?:register|registration)(?:_|$)", re.I)


def _snake(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(name))
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def _semantic_name(name: str) -> bool:
    return bool(_SELECTOR_TOKEN_RE.search(_snake(name)))


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _root_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _root_name(node.value)
    return ""


def _literal_members(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
        return [str(node.value)]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_literal_members(item))
        return out
    if isinstance(node, ast.Dict):
        out: list[str] = []
        for key in node.keys:
            out.extend(_literal_members(key))
        return out
    if isinstance(node, ast.Call):
        name = _call_name(node.func).split(".")[-1].lower()
        if name in {"list", "tuple", "set", "frozenset", "choice"} and node.args:
            return _literal_members(node.args[0])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        value = node.operand.value
        if isinstance(value, (int, float)):
            return [str(-value if isinstance(node.op, ast.USub) else value)]
    return []


def _subscript_name(node: ast.Subscript) -> str:
    return _call_name(node.value).split(".")[-1]


def _literal_alias_members(node: ast.AST | None) -> list[str]:
    if not isinstance(node, ast.Subscript):
        return []
    if _subscript_name(node).lower() != "literal":
        return []
    return _literal_members(node.slice)


def _enum_base(node: ast.ClassDef) -> bool:
    return any(_call_name(base).split(".")[-1] in {"Enum", "StrEnum", "IntEnum", "Flag", "IntFlag"} for base in node.bases)


def _flag_dest(call: ast.Call) -> str:
    flags = [str(arg.value) for arg in call.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)]
    for keyword in call.keywords:
        if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return str(keyword.value.value)
    dashed = [flag for flag in flags if flag.startswith("--")]
    chosen = max(dashed or flags or [""], key=len)
    return chosen.lstrip("-").replace("-", "_")


def _choices(call: ast.Call) -> list[str]:
    for keyword in call.keywords:
        if keyword.arg == "choices":
            return _literal_members(keyword.value)
    return []


def _click_choices(call: ast.Call) -> list[str]:
    for keyword in call.keywords:
        if keyword.arg != "type" or not isinstance(keyword.value, ast.Call):
            continue
        if _call_name(keyword.value.func).split(".")[-1].lower() == "choice" and keyword.value.args:
            return _literal_members(keyword.value.args[0])
    return []


def _registration_category(name: str) -> str | None:
    normalized = _snake(name)
    if not _REGISTER_RE.search(normalized):
        return None
    stripped = normalized.replace("registration", "").replace("register", "")
    return stripped.strip("_") if _semantic_name(stripped) else None


def _merge_row(store: Dict[str, Dict[str, Any]], *, symbol: str, line: int, members: Iterable[str], kind: str, enumerable: bool = True) -> None:
    row = store.setdefault(symbol, {
        "symbol": symbol,
        "line": int(line or 0),
        "members": set(),
        "enumerable": bool(enumerable),
        "assignment_kind": kind,
        "surface_kinds": set(),
    })
    row["line"] = min(int(row.get("line") or line or 0), int(line or row.get("line") or 0))
    row["members"].update(str(value) for value in members if str(value).strip())
    row["enumerable"] = bool(row["enumerable"] and enumerable)
    row["surface_kinds"].add(kind)


def _selector_registry_findings(path: Path, rel: str) -> list[Dict[str, Any]]:
    if path.suffix.lower() != ".py":
        return []
    text = workload._read(path)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return []

    store: Dict[str, Dict[str, Any]] = {}
    # Preserve all v25 registry surfaces, but aggregate duplicate assignments.
    for row in scientific_v25._scientific_registry_findings(path, rel):
        _merge_row(
            store,
            symbol=str(row.get("symbol") or ""),
            line=int(row.get("line") or 0),
            members=[str(v) for v in (row.get("members") or [])],
            kind=str(row.get("assignment_kind") or "registry_assignment"),
            enumerable=bool(row.get("enumerable", False)),
        )

    for node in ast.walk(tree):
        # Enum scientific selectors.
        if isinstance(node, ast.ClassDef) and _semantic_name(node.name) and _enum_base(node):
            members: list[str] = []
            for item in node.body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)):
                    targets = item.targets if isinstance(item, ast.Assign) else [item.target]
                    value = item.value
                    literal = _literal_members(value)
                    for target in targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            members.extend(literal or [target.id])
            _merge_row(store, symbol=f"enum:{node.name}", line=node.lineno, members=members, kind="enum")

        # Scientific Literal aliases.
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and _semantic_name(target.id):
                    literal = _literal_alias_members(value)
                    if literal:
                        _merge_row(store, symbol=f"literal:{target.id}", line=node.lineno, members=literal, kind="typing_literal")
                # REGISTRY["member"] = implementation.
                if isinstance(target, ast.Subscript):
                    root = _root_name(target.value)
                    if _semantic_name(root):
                        member = _literal_members(target.slice)
                        if member:
                            _merge_row(store, symbol=root, line=node.lineno, members=member, kind="subscript_registration")

        if isinstance(node, ast.Call):
            call = _call_name(node.func)
            tail = call.split(".")[-1]
            # argparse-style add_argument(..., choices=[...]).
            if tail == "add_argument":
                dest = _flag_dest(node)
                choices = _choices(node)
                if dest and choices and _semantic_name(dest):
                    _merge_row(store, symbol=f"cli:{dest}", line=node.lineno, members=choices, kind="argparse_choices")
            # click.option(..., type=click.Choice([...])).
            elif tail == "option":
                dest = _flag_dest(node)
                choices = _click_choices(node)
                if dest and choices and _semantic_name(dest):
                    _merge_row(store, symbol=f"cli:{dest}", line=node.lineno, members=choices, kind="click_choices")

            # register_model("name") / MODEL_REGISTRY.register("name").
            category = _registration_category(tail)
            receiver = call.rsplit(".", 1)[0].split(".")[-1] if "." in call else ""
            if category and node.args:
                members = _literal_members(node.args[0])
                if members:
                    _merge_row(store, symbol=f"registration:{category}", line=node.lineno, members=members, kind="registration_call")
            elif tail.lower() in {"register", "add", "register_module"} and receiver and _semantic_name(receiver) and node.args:
                members = _literal_members(node.args[0])
                if members:
                    _merge_row(store, symbol=receiver, line=node.lineno, members=members, kind="registry_method_registration")

        # if model == "x" / model in {"x", "y"} scientific dispatch.
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and _semantic_name(node.left.id):
            members: list[str] = []
            for comparator in node.comparators:
                members.extend(_literal_members(comparator))
            if members:
                _merge_row(store, symbol=f"branch:{node.left.id}", line=node.lineno, members=members, kind="branch_dispatch")

        # match model: case "x" | "y": scientific dispatch.
        if isinstance(node, ast.Match) and isinstance(node.subject, ast.Name) and _semantic_name(node.subject.id):
            members: list[str] = []
            for case in node.cases:
                pattern = case.pattern
                if isinstance(pattern, ast.MatchValue):
                    members.extend(_literal_members(pattern.value))
                elif isinstance(pattern, ast.MatchOr):
                    for part in pattern.patterns:
                        if isinstance(part, ast.MatchValue):
                            members.extend(_literal_members(part.value))
            if members:
                _merge_row(store, symbol=f"match:{node.subject.id}", line=node.lineno, members=members, kind="match_dispatch")

    rows: list[Dict[str, Any]] = []
    for symbol, row in sorted(store.items()):
        rows.append({
            "path": rel,
            "symbol": symbol,
            "line": int(row["line"]),
            "members": sorted(row["members"]),
            "enumerable": bool(row["enumerable"]),
            "assignment_kind": str(row["assignment_kind"]),
            "surface_kinds": sorted(row["surface_kinds"]),
        })
    return rows


def _iter_component_configs(root: Path) -> list[str]:
    rows: list[str] = []
    for path in workload._iter_files(root):
        if path.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            continue
        low = rel.lower()
        if not ("config" in low or "recipe" in low or "experiment" in low):
            continue
        if _COMPONENT_DIR_RE.search(low) or _semantic_name(path.stem):
            rows.append(rel)
    return sorted(set(rows))


def _job_corpus(jobs: Sequence[Mapping[str, Any]]) -> str:
    values: list[str] = []
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                values.append(str(key))
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
        elif value is not None:
            values.append(str(value))
    for job in jobs:
        visit(job)
    return "\n".join(values).lower()


def _all_selector_present(jobs: Sequence[Mapping[str, Any]]) -> bool:
    return any(registry_members._all_selector(job) for job in jobs)


def _config_reference_closure(root: Path, seeds: set[str], candidates: set[str], jobs: Sequence[Mapping[str, Any]]) -> set[str]:
    reachable = set(seeds) & set(candidates)
    by_name: Dict[str, list[str]] = {}
    by_stem: Dict[str, list[str]] = {}
    for rel in candidates:
        by_name.setdefault(Path(rel).name.lower(), []).append(rel)
        by_stem.setdefault(Path(rel).stem.lower(), []).append(rel)

    corpus = _job_corpus(jobs)
    for rel in candidates:
        name = Path(rel).name.lower()
        stem = Path(rel).stem.lower()
        if rel.lower() in corpus or (len(by_name.get(name, [])) == 1 and name in corpus) or (len(by_stem.get(stem, [])) == 1 and re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?![a-z0-9])", corpus)):
            reachable.add(rel)

    changed = True
    while changed:
        changed = False
        for source in list(reachable):
            path = root / source
            if not path.is_file():
                continue
            text = workload._read(path).lower()
            for rel in candidates - reachable:
                name = Path(rel).name.lower()
                stem = Path(rel).stem.lower()
                if rel.lower() in text or (len(by_name.get(name, [])) == 1 and name in text) or (len(by_stem.get(stem, [])) == 1 and re.search(r"(?<![a-z0-9])" + re.escape(stem) + r"(?![a-z0-9])", text)):
                    reachable.add(rel)
                    changed = True
    # A repository-authored --all scientific enumerator is allowed to cover
    # component recipes because the trainer itself owns compatibility semantics.
    if _all_selector_present(jobs):
        reachable.update(candidates)
    return reachable


def _config_exemptions(profile: Mapping[str, Any]) -> Dict[str, str]:
    raw = profile.get("scientific_component_config_exemptions") or {}
    if not isinstance(raw, Mapping):
        raise SystemExit("scientific_component_config_exemptions must be an object")
    result: Dict[str, str] = {}
    for path, reason in raw.items():
        text = str(reason or "").strip()
        if not text:
            raise SystemExit(f"component config exemption {path!r} requires a reason")
        result[str(path)] = text
    return result


def install_primitives() -> None:
    workload._registry_findings = _selector_registry_findings


def install_contract() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        report = original_report(root, profile, jobs)
        candidates = set(_iter_component_configs(root))
        workload_inventory = report.get("workload_surface_inventory") if isinstance(report.get("workload_surface_inventory"), Mapping) else {}
        seeds = set(str(value) for value in (workload_inventory.get("reachable_training_config_paths") or []))
        seeds.update(str(value) for value in (workload_inventory.get("direct_job_file_paths") or []))
        seeds.update(str(value) for value in (workload_inventory.get("source_referenced_training_configs") or []))
        reachable = _config_reference_closure(root, seeds, candidates, jobs)
        exemptions = _config_exemptions(profile)
        unknown_exemptions = sorted(set(exemptions) - candidates)
        if unknown_exemptions:
            raise SystemExit(f"scientific component config exemptions name unknown paths: {unknown_exemptions}")
        missing = sorted(candidates - reachable - set(exemptions))
        require = bool(profile.get("require_scientific_component_config_accounting", profile.get("strict_coverage", True)))
        controls = dict(report.get("strict_controls") or {})
        controls["require_scientific_component_config_accounting"] = require
        report.update({
            "selector_closure_schema": SELECTOR_CLOSURE_SCHEMA,
            "scientific_component_config_paths": sorted(candidates),
            "reachable_scientific_component_config_paths": sorted(reachable),
            "scientific_component_config_exemptions": dict(sorted(exemptions.items())),
            "unaccounted_scientific_component_configs": missing,
            "strict_scientific_component_config_pass": not missing,
            "strict_controls": controls,
        })
        if require and missing:
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report


__all__ = [
    "SELECTOR_CLOSURE_SCHEMA", "install_primitives", "install_contract",
    "_selector_registry_findings", "_iter_component_configs",
]
