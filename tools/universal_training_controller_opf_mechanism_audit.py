#!/usr/bin/env python3
"""Prove that every repository controller delegates scheduling to literal OPF_ADP.

The universal controller is intentionally an adapter around the *unchanged* pinned
OPF scheduler.  This layer makes that claim mechanically auditable rather than a
comment:

* all pinned OPF runtime blobs are re-hashed with Git's blob algorithm;
* the scheduler's complete top-level function/class inventory and AST are
  fingerprinted;
* every OPF CLI option string visible in the pinned scheduler source is recorded;
* the universal DAG adapter source is parsed and every assignment to the imported
  ``opf`` module is enumerated.  Only ``opf.build_suite_jobs`` may be replaced;
  pressure/resource/retry/pause/resume/reporting functions may not be monkey-patched;
* the current repository's training-relevant source set is content-hashed so a
  coverage certificate is tied to the exact trainer/model tree it audited.

This module changes no scheduling, admission, retry, pause/resume, resource or
process-control behavior.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import universal_training_controller as base
import universal_training_controller_current as current
import universal_training_controller_dag as dag

MECHANISM_AUDIT_SCHEMA = 1
_ALLOWED_OPF_ASSIGNMENTS = {"build_suite_jobs"}
_CLI_RE = re.compile(r"add_argument\(\s*(['\"])(--?[A-Za-z0-9][A-Za-z0-9_-]*)\1")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ast_digest(text: str) -> str:
    tree = ast.parse(text)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return _sha256(normalized.encode("utf-8"))


def _top_level_symbols(text: str) -> Dict[str, list[str]]:
    tree = ast.parse(text)
    functions: list[str] = []
    classes: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {"functions": sorted(functions), "classes": sorted(classes)}


def _opf_attribute_assignments() -> list[str]:
    source = inspect.getsource(dag)
    tree = ast.parse(source)
    assigned: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            else:
                targets.append(node.target)
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "opf":
                assigned.add(target.attr)
    return sorted(assigned)


def _training_snapshot(root: Path, report: Mapping[str, Any]) -> Dict[str, Any]:
    inventory = report.get("inventory") or {}
    reachability = report.get("reachability") or {}
    paths: set[str] = set()
    if isinstance(inventory, Mapping):
        for key in (
            "executable_training_candidates",
            "model_surfaces",
            "training_logic_surfaces",
        ):
            value = inventory.get(key, []) or []
            if isinstance(value, list):
                paths.update(str(item) for item in value)
    if isinstance(reachability, Mapping):
        for key in ("direct_job_sources", "executed_sources", "reachable_sources"):
            value = reachability.get(key, []) or []
            if isinstance(value, list):
                paths.update(str(item) for item in value)

    rows: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for relative in sorted(paths):
        path = root / relative
        if not path.is_file():
            rows.append({"path": relative, "missing": True})
            aggregate.update(f"MISSING\0{relative}\0".encode("utf-8"))
            continue
        data = path.read_bytes()
        digest = _sha256(data)
        rows.append({"path": relative, "bytes": len(data), "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
    return {
        "schema": 1,
        "path_count": len(rows),
        "digest": aggregate.hexdigest(),
        "files": rows,
    }


def _mechanism_certificate(root: Path) -> Dict[str, Any]:
    cache = base._prepare_opf_runtime(root)
    observed_blobs: Dict[str, str | None] = {}
    blob_errors: list[str] = []
    for relative, expected in current.OPF_RUNTIME_BLOBS.items():
        path = cache / relative
        try:
            actual = base._git_blob_sha(path.read_bytes())
        except Exception as exc:
            actual = None
            blob_errors.append(f"{relative}: {exc}")
        observed_blobs[relative] = actual
        if actual != expected:
            blob_errors.append(f"{relative}: {actual} != {expected}")

    scheduler_path = cache / "utils" / "opf_massive_suite_runner.py"
    scheduler_text = scheduler_path.read_text(encoding="utf-8")
    scheduler_symbols = _top_level_symbols(scheduler_text)
    cli_options = sorted({match.group(2) for match in _CLI_RE.finditer(scheduler_text)})
    opf_assignments = _opf_attribute_assignments()
    forbidden_assignments = sorted(set(opf_assignments) - _ALLOWED_OPF_ASSIGNMENTS)

    return {
        "schema": MECHANISM_AUDIT_SCHEMA,
        "reference_repository": current.OPF_REFERENCE_REPOSITORY,
        "reference_commit": current.OPF_REFERENCE_COMMIT,
        "expected_runtime_blobs": dict(current.OPF_RUNTIME_BLOBS),
        "observed_runtime_blobs": observed_blobs,
        "runtime_blob_errors": blob_errors,
        "scheduler_git_blob_sha": base._git_blob_sha(scheduler_path.read_bytes()),
        "scheduler_sha256": _sha256(scheduler_path.read_bytes()),
        "scheduler_ast_sha256": _ast_digest(scheduler_text),
        "scheduler_top_level_symbols": scheduler_symbols,
        "scheduler_cli_options": cli_options,
        "scheduler_cli_option_count": len(cli_options),
        "adapter_opf_attribute_assignments": opf_assignments,
        "allowed_adapter_assignments": sorted(_ALLOWED_OPF_ASSIGNMENTS),
        "forbidden_adapter_assignments": forbidden_assignments,
        "only_job_catalog_builder_is_replaced": not forbidden_assignments and set(opf_assignments) <= _ALLOWED_OPF_ASSIGNMENTS,
        "literal_scheduler_source_modified": False,
        "pass": not blob_errors and not forbidden_assignments and set(opf_assignments) <= _ALLOWED_OPF_ASSIGNMENTS,
    }


def install() -> None:
    original_report = current._enhanced_coverage_report

    def coverage_report(root: Path, profile: Dict[str, Any], jobs: Sequence[Dict[str, Any]]):
        report = original_report(root, profile, jobs)
        certificate = _mechanism_certificate(root)
        snapshot = _training_snapshot(root, report)
        require_mechanism_parity = bool(profile.get("require_literal_opf_mechanism_parity", True))
        controls = dict(report.get("strict_controls") or {})
        controls["require_literal_opf_mechanism_parity"] = require_mechanism_parity
        report.update(
            {
                "opf_mechanism_audit_schema": MECHANISM_AUDIT_SCHEMA,
                "opf_mechanism_certificate": certificate,
                "training_surface_snapshot": snapshot,
                "strict_literal_opf_mechanism_parity_pass": bool(certificate.get("pass")),
                "strict_controls": controls,
            }
        )
        if require_mechanism_parity and not certificate.get("pass"):
            report["coverage_ok"] = False
        return report

    current._enhanced_coverage_report = coverage_report
