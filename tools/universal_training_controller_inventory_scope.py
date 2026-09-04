#!/usr/bin/env python3
"""Closed-world production scope for training/model surface inventory.

The universal source scanner intentionally uses broad lexical signatures to avoid
missing obscure training implementations. Test fixtures and the controller's own
implementation contain those same signatures by design, however, and are not
repository training surfaces. This layer removes only structurally proven
infrastructure paths from the production inventory and records every removal in
an explicit audit ledger.

The base scanner is deliberately framework-oriented (Torch optimizers, ``.fit``
calls, Trainer/RL APIs). Repository-owned numerical learners can legitimately
implement optimization with plain Python arithmetic and therefore never mention
those framework signatures. This layer adds a conservative AST-based
``quiet_learner`` detector. It requires both an explicit fitting/training API and
multiple optimization-state signals, with a parameter-update signal or a
resumable training-state type. The detected files are added to the ordinary
training-logic inventory, so they must become reachable from a scheduled job or
remain an audit failure.

Repositories may additionally classify *model-only* production files as frozen
inference/materialization surfaces through
``training_control/non_training_surface_accounting.json``. That contract is
fail-closed: entries use exact paths (no globs), a closed category vocabulary,
non-trivial reasons, must resolve to existing model surfaces, and are rejected if
the scanner sees executable training or training logic in the file. A malformed
or stale declaration forces the overall coverage audit to fail.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller_current as current

INVENTORY_SCOPE_SCHEMA = 3
ACCOUNTING_SCHEMA = "training-control/non-training-surface-accounting/v1"
ACCOUNTING_PATH = Path("training_control/non_training_surface_accounting.json")
ALLOWED_NONTRAINING_CATEGORIES = frozenset(
    {
        "frozen_model_backend",
        "inference_adapter",
        "post_training_scorer",
        "runtime_feature_provider",
        "scientific_inference_adapter",
        "serving_provider",
        "supervision_materializer",
        "training_data_materializer",
    }
)
_GLOB_CHARS = frozenset("*?[]{}")

# A quiet learner must expose an API whose name says it performs fitting/training
# and must also carry a sufficiently rich optimization vocabulary. These signals
# intentionally do not classify generic helpers merely because a filename uses
# the word "training".
_QUIET_API_RE = re.compile(
    r"^(?:fit(?:_|$)|train(?:_|$)|advance_.*(?:train|fit)|resume_.*(?:train|fit)|"
    r"(?:train|fit).*_resumable$)",
    re.I,
)
_QUIET_STATE_RE = re.compile(r"(?:Training|Fitting|Optimizer|Resume)State$", re.I)
_QUIET_PARAMETER_RE = re.compile(
    r"(?:weight|weights|theta|bias|parameter|parameters|coefficient|coefficients|gradient|gradients)",
    re.I,
)
_QUIET_OPTIMIZATION_SIGNALS = {
    "epochs": re.compile(r"\bepochs?\b", re.I),
    "batch": re.compile(r"\bbatch(?:_size|_index|es)?\b", re.I),
    "learning_rate": re.compile(r"\blearning_rate\b|\blr\b", re.I),
    "gradient": re.compile(r"\bgrad(?:ient)?s?\b", re.I),
    "validation": re.compile(r"\bvalidation(?:_loss)?\b|\bvalidation_", re.I),
    "best_state": re.compile(r"\bbest_(?:loss|epoch|weights?|theta|bias|metric)\b", re.I),
    "early_stopping": re.compile(r"\bpatience\b|\bmin_delta\b|\bstale_epochs?\b|\bbad_epochs?\b", re.I),
    "resume_cursor": re.compile(r"\b(?:resume|cursor|next_batch|batch_index|random_state|rng_state)\b", re.I),
}


def _exclusion_reason(rel: str) -> str | None:
    normalized = rel.replace("\\", "/").strip("/")
    path = Path(normalized)
    parts = {part.casefold() for part in path.parts[:-1]}
    if "tests" in parts or "test" in parts:
        return "test_source"
    if normalized == "run_all_training.py":
        return "launcher_infrastructure"
    if normalized in {
        "tools/account_wide_training_control_audit.py",
        "tools/account_wide_training_control_audit_v2.py",
    }:
        return "account_audit_infrastructure"
    if normalized.startswith("tools/universal_training_controller") and normalized.endswith(".py"):
        return "training_controller_infrastructure"
    return None


def _target_mentions_parameter(node: ast.AST) -> bool:
    try:
        rendered = ast.unparse(node)
    except Exception:
        rendered = ""
    return bool(_QUIET_PARAMETER_RE.search(rendered))


def _contains_parameter_update(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            if _target_mentions_parameter(node.target):
                return True
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is None or not isinstance(value, (ast.BinOp, ast.Call, ast.ListComp, ast.DictComp)):
                continue
            if any(_target_mentions_parameter(target) for target in targets):
                try:
                    rendered = ast.unparse(value)
                except Exception:
                    rendered = ""
                if _QUIET_PARAMETER_RE.search(rendered) or "learning_rate" in rendered or "gradient" in rendered:
                    return True
    return False


def _quiet_learner_evidence(rel: str, text: str) -> dict[str, Any] | None:
    if not rel.endswith(".py") or not text.strip():
        return None
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return None

    functions = sorted(
        {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _QUIET_API_RE.search(node.name)
        }
    )
    states = sorted(
        {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and _QUIET_STATE_RE.search(node.name)
        }
    )
    if not functions:
        return None

    signals = sorted(
        name for name, pattern in _QUIET_OPTIMIZATION_SIGNALS.items() if pattern.search(text)
    )
    parameter_update = _contains_parameter_update(tree)
    # Requiring at least four independent training signals suppresses ordinary
    # serializer/provider APIs named fit/train. A resumable training-state class
    # is itself strong evidence; otherwise we require an explicit arithmetic
    # parameter update in addition to the vocabulary.
    if len(signals) < 4:
        return None
    if not parameter_update and not states:
        return None
    if not ({"epochs", "batch", "validation", "gradient"} & set(signals)):
        return None

    return {
        "path": rel,
        "apis": functions,
        "state_types": states,
        "signals": signals,
        "parameter_update": parameter_update,
    }


def _augment_quiet_learners(root: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(report)
    existing_logic = {
        str(path).replace("\\", "/") for path in result.get("training_logic_surfaces", []) or []
    }
    existing_exec = {
        str(path).replace("\\", "/") for path in result.get("executable_training_candidates", []) or []
    }
    rows_by_path: dict[str, dict[str, Any]] = {
        str(row.get("path") or "").replace("\\", "/"): dict(row)
        for row in result.get("training_files", []) or []
        if isinstance(row, Mapping) and row.get("path")
    }
    evidence: list[dict[str, Any]] = []

    for path in current._iter_sources(root):
        if path.suffix.lower() != ".py":
            continue
        rel = path.relative_to(root).as_posix()
        if _exclusion_reason(rel) is not None:
            continue
        text = current._read_text(path)
        detected = _quiet_learner_evidence(rel, text)
        if detected is None:
            continue
        evidence.append(detected)
        existing_logic.add(rel)
        executable = current._is_executable_script(path, text)
        if executable:
            existing_exec.add(rel)
        row = rows_by_path.get(rel, {"path": rel})
        row.update(
            {
                "training_logic": True,
                "model_surface": bool(row.get("model_surface", False)),
                "executable": bool(executable or row.get("executable", False)),
                "checkpoint_write": bool(row.get("checkpoint_write", False)),
                "checkpoint_read": bool(row.get("checkpoint_read", False)),
                "resume_token": bool(row.get("resume_token", False) or current.RESUME_TOKEN.search(text)),
                "early_stopping": bool(row.get("early_stopping", False) or current.EARLY_STOP_TOKEN.search(text)),
                "quiet_learner": True,
            }
        )
        rows_by_path[rel] = row

    result["training_logic_surfaces"] = sorted(existing_logic)
    result["executable_training_candidates"] = sorted(existing_exec)
    result["training_files"] = sorted(rows_by_path.values(), key=lambda row: str(row.get("path") or ""))
    result["quiet_training_logic_surfaces"] = sorted(row["path"] for row in evidence)
    result["quiet_learner_evidence"] = sorted(evidence, key=lambda row: row["path"])
    result["quiet_learner_count"] = len(evidence)
    return result


def _normalize_exact_path(root: Path, value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str) or not value.strip():
        return None, "path must be a non-empty string"
    raw = value.strip().replace("\\", "/")
    if any(char in raw for char in _GLOB_CHARS):
        return None, f"glob/meta characters are forbidden in path {raw!r}"
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None, f"path must be repository-relative and non-traversing: {raw!r}"
    normalized = candidate.as_posix().lstrip("./")
    if not normalized or normalized == ".":
        return None, f"invalid repository path {raw!r}"
    resolved = (root / normalized).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, f"path escapes repository root: {raw!r}"
    if not resolved.is_file():
        return None, f"accounted source does not exist as a file: {normalized}"
    return normalized, None


def _load_nontraining_accounting(
    root: Path,
    *,
    model_surfaces: Sequence[str],
    training_logic_surfaces: Sequence[str],
    executable_training_candidates: Sequence[str],
) -> tuple[list[dict[str, str]], list[str]]:
    source = root / ACCOUNTING_PATH
    if not source.is_file():
        return [], []
    errors: list[str] = []
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"{ACCOUNTING_PATH.as_posix()} is not valid JSON: {exc}"]
    if not isinstance(value, Mapping):
        return [], [f"{ACCOUNTING_PATH.as_posix()} must contain a JSON object"]
    if value.get("schema") != ACCOUNTING_SCHEMA:
        errors.append(
            f"accounting schema must be {ACCOUNTING_SCHEMA!r}, got {value.get('schema')!r}"
        )
    raw_surfaces = value.get("surfaces")
    if not isinstance(raw_surfaces, list):
        return [], [*errors, "accounting surfaces must be a JSON array"]

    models = {str(path).replace("\\", "/") for path in model_surfaces}
    training_logic = {str(path).replace("\\", "/") for path in training_logic_surfaces}
    executable = {str(path).replace("\\", "/") for path in executable_training_candidates}
    seen: set[str] = set()
    accepted: list[dict[str, str]] = []
    for index, row in enumerate(raw_surfaces):
        prefix = f"surfaces[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        if set(row) != {"path", "category", "reason"}:
            errors.append(f"{prefix} must contain exactly path/category/reason")
            continue
        path, path_error = _normalize_exact_path(root, row.get("path"))
        if path_error:
            errors.append(f"{prefix}: {path_error}")
            continue
        assert path is not None
        category = row.get("category")
        if category not in ALLOWED_NONTRAINING_CATEGORIES:
            errors.append(
                f"{prefix}: category {category!r} is not in the closed non-training vocabulary"
            )
            continue
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 24:
            errors.append(f"{prefix}: reason must be a specific explanation of at least 24 characters")
            continue
        if path in seen:
            errors.append(f"{prefix}: duplicate accounted path {path}")
            continue
        seen.add(path)
        if path in training_logic or path in executable:
            errors.append(
                f"{prefix}: {path} contains executable/training logic and cannot be declared non-training"
            )
            continue
        if path not in models:
            errors.append(
                f"{prefix}: {path} is stale/not currently detected as a production model surface"
            )
            continue
        accepted.append(
            {"path": path, "category": str(category), "reason": reason.strip()}
        )
    return sorted(accepted, key=lambda row: row["path"]), errors


def _filter_inventory(root: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    report = _augment_quiet_learners(root, report)
    result = dict(report)
    excluded: dict[str, str] = {}

    def keep(path: str) -> bool:
        reason = _exclusion_reason(str(path))
        if reason is not None:
            excluded[str(path).replace("\\", "/")] = reason
            return False
        return True

    result["training_files"] = [
        row for row in report.get("training_files", [])
        if isinstance(row, dict) and keep(str(row.get("path") or ""))
    ]
    for key in (
        "executable_training_candidates",
        "model_surfaces",
        "training_logic_surfaces",
        "quiet_training_logic_surfaces",
    ):
        result[key] = sorted({
            str(path).replace("\\", "/")
            for path in report.get(key, []) or []
            if keep(str(path))
        })

    accounted, accounting_errors = _load_nontraining_accounting(
        root,
        model_surfaces=result.get("model_surfaces", []),
        training_logic_surfaces=result.get("training_logic_surfaces", []),
        executable_training_candidates=result.get("executable_training_candidates", []),
    )
    accounted_paths = {row["path"] for row in accounted}
    if not accounting_errors:
        result["model_surfaces"] = [
            path for path in result.get("model_surfaces", []) if path not in accounted_paths
        ]
        result["training_files"] = [
            row for row in result.get("training_files", [])
            if not (
                isinstance(row, dict)
                and str(row.get("path") or "").replace("\\", "/") in accounted_paths
                and not row.get("training_logic")
                and not row.get("executable")
            )
        ]

    result["inventory_scope_schema"] = INVENTORY_SCOPE_SCHEMA
    result["excluded_nontraining_sources"] = [
        {"path": path, "reason": excluded[path]}
        for path in sorted(excluded)
    ]
    result["excluded_nontraining_source_count"] = len(excluded)
    result["non_training_surface_accounting_schema"] = ACCOUNTING_SCHEMA
    result["non_training_surface_accounting"] = accounted
    result["non_training_surface_accounting_count"] = len(accounted)
    result["non_training_surface_accounting_errors"] = accounting_errors
    result["non_training_surface_accounting_pass"] = not accounting_errors
    return result


def install() -> None:
    original_inventory = current._training_inventory
    original_report = current._enhanced_coverage_report

    def training_inventory(root):
        return _filter_inventory(root, original_inventory(root))

    def coverage_report(root, profile, jobs):
        report = original_report(root, profile, jobs)
        inventory = report.get("inventory") or {}
        errors = (
            list(inventory.get("non_training_surface_accounting_errors", []))
            if isinstance(inventory, Mapping)
            else ["training inventory is missing from coverage report"]
        )
        report["non_training_surface_accounting_errors"] = errors
        report["non_training_surface_accounting_pass"] = not errors
        report["quiet_training_logic_surfaces"] = (
            list(inventory.get("quiet_training_logic_surfaces", []))
            if isinstance(inventory, Mapping)
            else []
        )
        report["quiet_learner_count"] = (
            int(inventory.get("quiet_learner_count", 0) or 0)
            if isinstance(inventory, Mapping)
            else 0
        )
        if errors:
            report["coverage_ok"] = False
        return report

    current._training_inventory = training_inventory
    current._enhanced_coverage_report = coverage_report


__all__ = [
    "ACCOUNTING_PATH",
    "ACCOUNTING_SCHEMA",
    "ALLOWED_NONTRAINING_CATEGORIES",
    "INVENTORY_SCOPE_SCHEMA",
    "_augment_quiet_learners",
    "_exclusion_reason",
    "_filter_inventory",
    "_load_nontraining_accounting",
    "_quiet_learner_evidence",
    "install",
]
