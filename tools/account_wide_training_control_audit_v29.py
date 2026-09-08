#!/usr/bin/env python3
"""Account-wide source certificate for exhaustive training-control v29.

This layers on the established estate auditor and upgrades the remote/local
contract to the v29 architecture.  Every owner repository except OPF_ADP must
expose a root launcher wired to the immutable v29 bootstrap or preservation
adapter; local source reports must prove literal OPF parity, real local command
and config targets, source-proven exact resume and semantic early stopping,
full model/workload/registry/config/selector/combination closure, and the v29
scientific ontology covering families/types/variants/capabilities/regularizers.

The certificate is a source/audit program only.  It does not train models,
download datasets, run inference, or replace runtime validation.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base

CANONICAL_BOOTSTRAP_COMMIT = "5fecd0732a6f57741a1ea61b94f196965af5411f"
CANONICAL_BOOTSTRAP_BLOB = "2c76bc3c4b70c511fc4e94aaa1bd0dd145bbadb4"
CANONICAL_ADAPTER_COMMIT = "f899ac190085228cc95f47725a36bfd105822374"
CANONICAL_ADAPTER_BLOB = "82a10e0a6e221dbf90de92d56a8b87d0f270935f"
CANONICAL_HOST = "Anurag9000/RigorousRAG"
CERTIFICATE_SCHEMA = 29

_BASE_STRICT_LOCAL_REPORT = base._strict_local_report


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    full_name = str(repo_row.get("full_name") or "")
    default_branch = str(repo_row.get("default_branch") or "")
    branches = base._branch_names(full_name)
    main_sha = base._main_sha(full_name)
    errors: list[str] = []
    launcher_blob = None
    launcher_text = ""
    wiring_mode = "unknown"

    if not main_sha:
        errors.append("missing main branch")
    else:
        try:
            launcher = base._raw(full_name, "run_all_training.py", "main")
            launcher_blob = base._git_blob_sha(launcher)
            launcher_text = launcher.decode("utf-8", errors="replace")
        except Exception as exc:
            errors.append(f"missing/unreadable run_all_training.py on main: {exc}")

    if launcher_text:
        direct = CANONICAL_BOOTSTRAP_BLOB in launcher_text and "universal_training_controller_entry.py" in launcher_text
        adapter = CANONICAL_ADAPTER_BLOB in launcher_text and "repo_training_launcher_adapter.py" in launcher_text
        local_host = full_name == CANONICAL_HOST and "tools" in launcher_text and "universal_training_controller_entry.py" in launcher_text
        if direct:
            wiring_mode = "direct_immutable_bootstrap"
        elif adapter:
            wiring_mode = "immutable_preservation_adapter"
        elif local_host:
            wiring_mode = "canonical_local_bootstrap"
            try:
                bootstrap = base._raw(full_name, "tools/universal_training_controller_entry.py", "main")
                observed = base._git_blob_sha(bootstrap)
                if observed != CANONICAL_BOOTSTRAP_BLOB:
                    errors.append(f"canonical host bootstrap blob mismatch: {observed} != {CANONICAL_BOOTSTRAP_BLOB}")
            except Exception as exc:
                errors.append(f"canonical host bootstrap unavailable: {exc}")
        else:
            errors.append("launcher pins neither canonical v29 bootstrap nor canonical v29 preservation adapter")

    if default_branch != "main":
        errors.append(f"default branch is {default_branch!r}, expected 'main'")
    extra = [name for name in branches if name != "main"]
    if extra:
        errors.append(f"non-main branches remain: {extra}")

    return {
        "repository": full_name,
        "default_branch": default_branch,
        "default_branch_is_main": default_branch == "main",
        "main_sha": main_sha,
        "branches": branches,
        "extra_branches": extra,
        "launcher_blob": launcher_blob,
        "wiring_mode": wiring_mode,
        "canonical_bootstrap_commit": CANONICAL_BOOTSTRAP_COMMIT,
        "canonical_bootstrap_blob": CANONICAL_BOOTSTRAP_BLOB,
        "canonical_adapter_commit": CANONICAL_ADAPTER_COMMIT,
        "canonical_adapter_blob": CANONICAL_ADAPTER_BLOB,
        "errors": errors,
        "pass": not errors,
    }


def _nonempty(report: dict[str, Any], key: str) -> bool:
    value = report.get(key)
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def _strict(report: dict[str, Any]) -> list[str]:
    errors = list(_BASE_STRICT_LOCAL_REPORT(report))

    if "strict_source_contracts_pass" in report and not bool(report.get("strict_source_contracts_pass")):
        errors.append("source-proven command/resume/early-stopping contracts failed")

    for boolean_key, label in (
        ("strict_registry_member_pass", "registry-member accounting failed"),
        ("strict_dynamic_scientific_registry_pass", "dynamic scientific registry accounting failed"),
        ("strict_scientific_component_config_pass", "scientific component config accounting failed"),
        ("strict_declared_combination_pass", "declared scientific combination accounting failed"),
    ):
        if boolean_key in report and not bool(report.get(boolean_key)):
            errors.append(label)

    for key in (
        "jobs_with_missing_local_targets",
        "training_jobs_without_source_exact_resume",
        "training_jobs_without_source_semantic_early_stopping",
        "uncovered_registry_members",
        "unaccounted_dynamic_scientific_registries",
        "unaccounted_scientific_component_configs",
        "uncovered_declared_combinations",
    ):
        if _nonempty(report, key):
            value = report.get(key)
            errors.append(f"{key}={len(value) if hasattr(value, '__len__') else value}")

    inventory = report.get("declared_combination_inventory") or {}
    if isinstance(inventory, dict) and int(inventory.get("uncovered_combination_count") or 0):
        errors.append(f"declared_combination_inventory.uncovered={inventory.get('uncovered_combination_count')}")

    member_inventory = report.get("registry_member_inventory") or {}
    if isinstance(member_inventory, dict) and int(member_inventory.get("uncovered_registry_member_count") or 0):
        errors.append(f"registry_member_inventory.uncovered={member_inventory.get('uncovered_registry_member_count')}")

    controls = report.get("strict_controls") or {}
    if isinstance(controls, dict):
        required_v29 = (
            "require_existing_job_targets",
            "require_source_proven_training_exact_resume",
            "require_source_proven_training_early_stopping",
            "require_registry_member_accounting",
            "require_dynamic_registry_accounting",
            "require_scientific_component_config_accounting",
            "require_declared_combination_accounting",
            "require_scientific_ontology_accounting",
        )
        for key in required_v29:
            if key in controls and controls.get(key) is False:
                errors.append(f"strict control disabled: {key}")

    ontology = report.get("scientific_ontology_added_dimensions") or []
    required_dims = {"family", "model_type", "variant", "capability", "regularizer"}
    if ontology and not required_dims.issubset({str(value) for value in ontology}):
        errors.append("v29 scientific ontology dimensions incomplete")

    return sorted(set(errors))


def main() -> int:
    base.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    base.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    base.SCHEMA = CERTIFICATE_SCHEMA
    base._launcher_remote_audit = _remote
    base._strict_local_report = _strict
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
