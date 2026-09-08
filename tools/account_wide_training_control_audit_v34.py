#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v34.

Every live owner repository except OPF_ADP must expose a root launcher wired to
the immutable v34 bootstrap or v34 preservation adapter. Local reports must pass
all v33 scientific/source contracts and the retained-trainable-source closure:
no real retained trainer/model source may be hidden behind broad ignore/dynamic
coverage or manual/reference/research classification.

This performs source auditing only. It does not download datasets or execute
model training/inference and is not a runtime test substitute.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v33 as prior

CANONICAL_BOOTSTRAP_COMMIT = "142587b31ea4097e237cbe4ad756bb595c1591f7"
CANONICAL_BOOTSTRAP_BLOB = "0c7c6307ca97fbc502b511f845059f345712fc7d"
CANONICAL_ADAPTER_COMMIT = "a709cf605b818a61fdcce0e57c11a98459f6a9a1"
CANONICAL_ADAPTER_BLOB = "f174e72eebf00efb7f9de6e785210a9418b47450"
CERTIFICATE_SCHEMA = 34
RETAINED_TRAINING_CLOSURE_SCHEMA = 1


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    prior.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    prior.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    prior.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    prior.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    return prior._remote(repo_row)


def _strict(report: dict[str, Any]) -> list[str]:
    errors = list(prior._strict(report))
    if report.get("retained_training_closure_schema") != RETAINED_TRAINING_CLOSURE_SCHEMA:
        errors.append(
            "retained_training_closure_schema="
            f"{report.get('retained_training_closure_schema')!r}"
        )
    if report.get("retained_training_cartesian_products_invented") is not False:
        errors.append("v34 retained-source Cartesian-product guard missing")
    if not bool(report.get("strict_retained_training_closure_pass")):
        errors.append("retained trainable-source closure failed")
    for key in (
        "malformed_retained_training_surface_exclusions",
        "unreachable_retained_executable_trainers",
        "unreachable_retained_training_logic_surfaces",
        "unreachable_retained_model_surfaces",
        "unreachable_retained_trainable_sources",
        "trainable_sources_hidden_by_ignore_patterns",
        "trainable_sources_hidden_by_dynamic_cover_patterns",
    ):
        value = report.get(key, []) or []
        if value:
            errors.append(f"{key}={len(value) if isinstance(value, list) else value}")
    controls = report.get("strict_controls") or {}
    if not isinstance(controls, dict) or not bool(
        controls.get("require_all_retained_trainable_source_reachability")
    ):
        errors.append(
            "strict control disabled or absent: "
            "require_all_retained_trainable_source_reachability"
        )
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
