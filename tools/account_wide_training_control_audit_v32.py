#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v32.

Every live owner repository except OPF_ADP must expose a root launcher wired to
the immutable v32 bootstrap or v32 preservation adapter. Local source reports
must pass every v31 contract and explicitly enable the full v32 scientific-choice
accounting contract. This performs source auditing only; it does not download
datasets, execute training/inference, or substitute runtime validation.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v31 as prior

CANONICAL_BOOTSTRAP_COMMIT = "e16946686d09a2b2c3afdba5a18162d6f0aaeaf9"
CANONICAL_BOOTSTRAP_BLOB = "e794a90b6d681a7b275f92deadce82861f303bbf"
CANONICAL_ADAPTER_COMMIT = "edff50ab3adecb792c64a3fa98c906a73b0fb9d5"
CANONICAL_ADAPTER_BLOB = "7f73a479548c3417bdecf58bb9083e5226ba606a"
CERTIFICATE_SCHEMA = 32
SCIENTIFIC_ONTOLOGY_V32_SCHEMA = 1


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    prior.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    prior.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    prior.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    prior.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    return prior._remote(repo_row)


def _strict(report: dict[str, Any]) -> list[str]:
    errors = list(prior._strict(report))
    if report.get("scientific_ontology_v32_schema") != SCIENTIFIC_ONTOLOGY_V32_SCHEMA:
        errors.append(
            "scientific_ontology_v32_schema="
            f"{report.get('scientific_ontology_v32_schema')!r}"
        )
    if report.get("full_scientific_choice_cartesian_products_invented") is not False:
        errors.append("v32 full scientific-choice Cartesian-product guard missing")
    controls = report.get("strict_controls") or {}
    if not isinstance(controls, dict) or not bool(
        controls.get("require_full_scientific_choice_accounting")
    ):
        errors.append(
            "strict control disabled or absent: require_full_scientific_choice_accounting"
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
