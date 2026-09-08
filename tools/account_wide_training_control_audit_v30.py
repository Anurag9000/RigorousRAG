#!/usr/bin/env python3
"""Account-wide source certificate for exhaustive training-control v30.

Every owner repository except OPF_ADP must expose a root launcher wired to the
immutable v30 bootstrap or v30 preservation adapter. Local source reports must
also pass every v29 scientific/source contract plus v30 declarative scientific
closure: typed selectors, registry mutations/merges, structured config choices,
Hydra component configs and concrete implementation declarations.

This certificate is static source/audit infrastructure only. It does not train,
download datasets, run inference, or replace runtime validation.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v29 as prior

CANONICAL_BOOTSTRAP_COMMIT = "df5cd8e3b78451c3e9a134cb685d2f90902d8925"
CANONICAL_BOOTSTRAP_BLOB = "ecf809be32092aac6e585e15edb6b5a6f90798bc"
CANONICAL_ADAPTER_COMMIT = "d9c350e3c2d6b9d1ee418de3a9b4786ed5246a4c"
CANONICAL_ADAPTER_BLOB = "c96f5f2f6a754f4ea4125bd38334c2ee6e76ad4b"
CERTIFICATE_SCHEMA = 30


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    prior.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    prior.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    prior.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    prior.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    return prior._remote(repo_row)


def _strict(report: dict[str, Any]) -> list[str]:
    errors = list(prior._strict(report))
    if "strict_declarative_scientific_source_pass" in report and not bool(report.get("strict_declarative_scientific_source_pass")):
        errors.append("declarative scientific source accounting failed")
    for key in (
        "unaccounted_concrete_scientific_declarations",
        "unaccounted_structured_scientific_choices",
    ):
        value = report.get(key)
        if isinstance(value, (list, tuple, set, dict)) and value:
            errors.append(f"{key}={len(value)}")
    controls = report.get("strict_controls") or {}
    if isinstance(controls, dict) and controls.get("require_declarative_scientific_source_accounting") is False:
        errors.append("strict control disabled: require_declarative_scientific_source_accounting")
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
