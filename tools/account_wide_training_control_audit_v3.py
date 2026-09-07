#!/usr/bin/env python3
"""Account-wide source/wiring certificate for selector-complete controller v26.

Accepts either a direct immutable v26 bootstrap pin or the immutable v26
preservation adapter. Local audits additionally require v23-v26 workload,
registry-member, dynamic-registry, source-contract and component-config closure.
No model training, dataset download or runtime/fault-injection test is performed.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "a0222d8753cddddb87e24c57175487d38158b663"
CANONICAL_BOOTSTRAP_BLOB = "a31a729adf4a2ce926cc4277eee06a0b7c7a2f1e"
CANONICAL_ADAPTER_COMMIT = "019e3542943de48fb5b55dd44dd75d7572db023d"
CANONICAL_ADAPTER_BLOB = "4d1d4b154abfdd25e902e959d6492fb614dc9299"


def _launcher_remote_audit(repo_row: dict[str, Any]) -> dict[str, Any]:
    full_name = str(repo_row.get("full_name") or "")
    default_branch = str(repo_row.get("default_branch") or "")
    branches = audit._branch_names(full_name)
    main_sha = audit._main_sha(full_name)
    errors: list[str] = []
    launcher_blob = None
    launcher_text = ""
    if not main_sha:
        errors.append("missing main branch")
    else:
        try:
            launcher = audit._raw(full_name, "run_all_training.py", "main")
            launcher_blob = audit._git_blob_sha(launcher)
            launcher_text = launcher.decode("utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"missing/unreadable run_all_training.py on main: {exc}")

    direct = bool(
        launcher_text
        and CANONICAL_BOOTSTRAP_BLOB in launcher_text
        and "universal_training_controller_entry.py" in launcher_text
    )
    adapter = bool(
        launcher_text
        and CANONICAL_ADAPTER_BLOB in launcher_text
        and "repo_training_launcher_adapter.py" in launcher_text
    )
    if launcher_text and not (direct or adapter):
        errors.append("launcher pins neither canonical v26 bootstrap nor canonical v26 preservation adapter")
    if direct and "require_literal_opf_mechanism_parity" not in launcher_text:
        errors.append("direct launcher does not explicitly require literal OPF mechanism parity")
    if adapter:
        required = (
            "TRAINING_LAUNCHER_BASE_REPOSITORY",
            "TRAINING_LAUNCHER_BASE_COMMIT",
            "TRAINING_LAUNCHER_BASE_BLOB",
        )
        missing = [token for token in required if token not in launcher_text]
        if missing:
            errors.append(f"preservation adapter launcher missing immutable base fields: {missing}")
        has_catalog = "TRAINING_LAUNCHER_FINAL_CATALOG" in launcher_text
        has_catalog_blob = "TRAINING_LAUNCHER_FINAL_CATALOG_BLOB" in launcher_text
        if has_catalog != has_catalog_blob:
            errors.append("catalog override is not blob-pinned")

    return {
        "repository": full_name,
        "default_branch": default_branch,
        "default_branch_is_main": default_branch == "main",
        "main_sha": main_sha,
        "branches": branches,
        "extra_branches": [name for name in branches if name != "main"],
        "launcher_blob": launcher_blob,
        "launcher_mode": "direct" if direct else "preservation_adapter" if adapter else "unknown",
        "canonical_bootstrap_commit": CANONICAL_BOOTSTRAP_COMMIT,
        "canonical_bootstrap_blob": CANONICAL_BOOTSTRAP_BLOB,
        "canonical_adapter_commit": CANONICAL_ADAPTER_COMMIT,
        "canonical_adapter_blob": CANONICAL_ADAPTER_BLOB,
        "errors": errors,
        "pass": not errors,
    }


def _strict_local_report(report: dict[str, Any]) -> list[str]:
    errors = list(audit._strict_local_report(report))
    for key in (
        "unaccounted_workload_registry_paths",
        "unaccounted_training_config_paths",
        "uncovered_registry_members",
        "unaccounted_dynamic_scientific_registries",
        "unaccounted_scientific_component_configs",
        "jobs_with_missing_local_targets",
        "training_jobs_without_source_exact_resume",
        "training_jobs_without_source_semantic_early_stopping",
    ):
        value = report.get(key, []) or []
        if value:
            errors.append(f"{key}={len(value) if isinstance(value, list) else value}")
    for key in (
        "strict_workload_surface_pass",
        "strict_registry_member_pass",
        "strict_dynamic_scientific_registry_pass",
        "strict_scientific_component_config_pass",
        "strict_source_contracts_pass",
    ):
        if key in report and not bool(report.get(key)):
            errors.append(f"{key}=false")
    return list(dict.fromkeys(errors))


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    audit._launcher_remote_audit = _launcher_remote_audit
    audit._strict_local_report = _strict_local_report
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
