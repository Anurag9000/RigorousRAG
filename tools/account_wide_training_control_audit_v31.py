#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v31.

Every owner repository except OPF_ADP must expose a root launcher wired to the
immutable v31 bootstrap or v31 preservation adapter. Local source reports must
pass every v30 contract and explicitly enable the extended v31 scientific
component ontology. This is static source auditing only: it does not download
assets, execute training, run inference, or substitute runtime validation.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v30 as prior

CANONICAL_BOOTSTRAP_COMMIT = "6bbb504893699aac7ab461767f8170b4ee1797fc"
CANONICAL_BOOTSTRAP_BLOB = "8ee1a31f2368f3fa0fad7b0b2896fe6da4613bd3"
CANONICAL_ADAPTER_COMMIT = "04e875505372d70711709042e4322384f6ded9d6"
CANONICAL_ADAPTER_BLOB = "ec384d937fb75f5bf1ff257d135c3f28001c29cd"
CERTIFICATE_SCHEMA = 31
SCIENTIFIC_ONTOLOGY_V31_SCHEMA = 1


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    prior.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    prior.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    prior.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    prior.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    return prior._remote(repo_row)


def _strict(report: dict[str, Any]) -> list[str]:
    errors = list(prior._strict(report))
    if report.get("scientific_ontology_v31_schema") != SCIENTIFIC_ONTOLOGY_V31_SCHEMA:
        errors.append(
            "scientific_ontology_v31_schema="
            f"{report.get('scientific_ontology_v31_schema')!r}"
        )
    if report.get("extended_scientific_component_cartesian_products_invented") is not False:
        errors.append("v31 extended scientific component Cartesian-product guard missing")
    controls = report.get("strict_controls") or {}
    if not isinstance(controls, dict) or not bool(
        controls.get("require_extended_scientific_component_accounting")
    ):
        errors.append(
            "strict control disabled or absent: "
            "require_extended_scientific_component_accounting"
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
