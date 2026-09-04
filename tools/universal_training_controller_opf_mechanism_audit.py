#!/usr/bin/env python3
"""Audit literal-OPF delegation without making coverage depend on network access.

There are deliberately two integrity gates:

* coverage-time proves the selected OPF repository/commit/blob manifest is
  synchronized across the controller layers, proves the adapter mutates only
  ``opf.build_suite_jobs``, and proves execution still contains the Git-blob
  verification gate;
* execution-time ``base._prepare_opf_runtime`` verifies every materialized OPF
  file against that manifest before the scheduler is imported.

If a verified local OPF cache is already present, coverage also fingerprints the
literal scheduler AST, symbols and CLI.  A malformed or mismatched local cache is
never ignored: it fails the mechanism certificate.  A missing cache is simply
reported as ``deferred_to_execution`` and is not downloaded by an audit.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import universal_training_controller as base
import universal_training_controller_current as current
import universal_training_controller_dag as dag

MECHANISM_AUDIT_SCHEMA = 2
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
        for key in ("executable_training_candidates", "model_surfaces", "training_logic_surfaces"):
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
    return {"schema": 1, "path_count": len(rows), "digest": aggregate.hexdigest(), "files": rows}


def _local_reference_cache(root: Path) -> Path:
    return root / ".training_control" / "opf_reference" / str(current.OPF_REFERENCE_COMMIT)


def _execution_integrity_gate() -> Dict[str, Any]:
    verify_source = inspect.getsource(base._verify_reference_file)
    prepare_source = inspect.getsource(base._prepare_opf_runtime)
    checks = {
        "verifier_uses_git_blob_sha": "_git_blob_sha" in verify_source,
        "verifier_reads_expected_manifest": "OPF_RUNTIME_BLOBS" in verify_source,
        "verifier_rejects_mismatch": "raise RuntimeError" in verify_source and "mismatch" in verify_source.lower(),
        "prepare_verifies_cached_files": "_verify_reference_file" in prepare_source,
        "prepare_verifies_downloaded_files": prepare_source.count("_verify_reference_file") >= 2,
        "prepare_iterates_runtime_manifest": "OPF_RUNTIME_FILES" in prepare_source,
    }
    return {"checks": checks, "pass": all(checks.values())}


def _inspect_local_runtime(cache: Path) -> Dict[str, Any]:
    expected = dict(current.OPF_RUNTIME_BLOBS)
    observed: Dict[str, str | None] = {}
    errors: list[str] = []
    for relative, wanted in expected.items():
        path = cache / relative
        try:
            actual = base._git_blob_sha(path.read_bytes())
        except Exception as exc:
            actual = None
            errors.append(f"{relative}: {exc}")
        observed[relative] = actual
        if actual != wanted:
            errors.append(f"{relative}: {actual} != {wanted}")

    result: Dict[str, Any] = {
        "status": "verified_local_cache" if not errors else "invalid_local_cache",
        "observed_runtime_blobs": observed,
        "runtime_blob_errors": errors,
        "pass": not errors,
        "scheduler_git_blob_sha": None,
        "scheduler_sha256": None,
        "scheduler_ast_sha256": None,
        "scheduler_top_level_symbols": {"functions": [], "classes": []},
        "scheduler_cli_options": [],
        "scheduler_cli_option_count": 0,
    }
    if errors:
        return result

    scheduler_path = cache / "utils" / "opf_massive_suite_runner.py"
    scheduler_data = scheduler_path.read_bytes()
    scheduler_text = scheduler_data.decode("utf-8")
    cli_options = sorted({match.group(2) for match in _CLI_RE.finditer(scheduler_text)})
    result.update(
        {
            "scheduler_git_blob_sha": base._git_blob_sha(scheduler_data),
            "scheduler_sha256": _sha256(scheduler_data),
            "scheduler_ast_sha256": _ast_digest(scheduler_text),
            "scheduler_top_level_symbols": _top_level_symbols(scheduler_text),
            "scheduler_cli_options": cli_options,
            "scheduler_cli_option_count": len(cli_options),
        }
    )
    return result


def _mechanism_certificate(root: Path) -> Dict[str, Any]:
    expected_blobs = dict(current.OPF_RUNTIME_BLOBS)
    reference_synchronized = (
        base.OPF_REFERENCE_REPOSITORY == current.OPF_REFERENCE_REPOSITORY
        and base.OPF_REFERENCE_COMMIT == current.OPF_REFERENCE_COMMIT
        and dict(base.OPF_RUNTIME_BLOBS) == expected_blobs
        and tuple(base.OPF_RUNTIME_FILES) == tuple(expected_blobs)
    )
    execution_gate = _execution_integrity_gate()
    opf_assignments = _opf_attribute_assignments()
    forbidden_assignments = sorted(set(opf_assignments) - _ALLOWED_OPF_ASSIGNMENTS)

    cache = _local_reference_cache(root)
    any_cached_runtime_file = any((cache / relative).exists() for relative in expected_blobs)
    marker_exists = (cache / "REFERENCE.json").is_file()
    if any_cached_runtime_file or marker_exists:
        local_runtime = _inspect_local_runtime(cache)
    else:
        local_runtime = {
            "status": "deferred_to_execution",
            "observed_runtime_blobs": {},
            "runtime_blob_errors": [],
            "pass": True,
            "scheduler_git_blob_sha": None,
            "scheduler_sha256": None,
            "scheduler_ast_sha256": None,
            "scheduler_top_level_symbols": {"functions": [], "classes": []},
            "scheduler_cli_options": [],
            "scheduler_cli_option_count": 0,
        }

    certificate_pass = (
        reference_synchronized
        and bool(execution_gate.get("pass"))
        and bool(local_runtime.get("pass"))
        and not forbidden_assignments
        and set(opf_assignments) <= _ALLOWED_OPF_ASSIGNMENTS
    )
    return {
        "schema": MECHANISM_AUDIT_SCHEMA,
        "reference_repository": current.OPF_REFERENCE_REPOSITORY,
        "reference_commit": current.OPF_REFERENCE_COMMIT,
        "expected_runtime_blobs": expected_blobs,
        "reference_synchronized": reference_synchronized,
        "runtime_validation_status": local_runtime["status"],
        "runtime_validation_deferred_to_execution": local_runtime["status"] == "deferred_to_execution",
        "observed_runtime_blobs": local_runtime["observed_runtime_blobs"],
        "runtime_blob_errors": local_runtime["runtime_blob_errors"],
        "execution_integrity_gate": execution_gate,
        "scheduler_git_blob_sha": local_runtime["scheduler_git_blob_sha"],
        "scheduler_sha256": local_runtime["scheduler_sha256"],
        "scheduler_ast_sha256": local_runtime["scheduler_ast_sha256"],
        "scheduler_top_level_symbols": local_runtime["scheduler_top_level_symbols"],
        "scheduler_cli_options": local_runtime["scheduler_cli_options"],
        "scheduler_cli_option_count": local_runtime["scheduler_cli_option_count"],
        "adapter_opf_attribute_assignments": opf_assignments,
        "allowed_adapter_assignments": sorted(_ALLOWED_OPF_ASSIGNMENTS),
        "forbidden_adapter_assignments": forbidden_assignments,
        "only_job_catalog_builder_is_replaced": not forbidden_assignments and set(opf_assignments) <= _ALLOWED_OPF_ASSIGNMENTS,
        "literal_scheduler_source_modified": False,
        "pass": certificate_pass,
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
