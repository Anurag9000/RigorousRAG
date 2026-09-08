#!/usr/bin/env python3
"""Account-wide static source certificate for exhaustive training-control v33.

Every live owner repository except OPF_ADP must expose a root launcher wired to
the immutable v33 bootstrap or v33 preservation adapter. Local source reports
must pass every v32 contract and explicitly enable the role/paradigm/protocol
scientific-choice accounting contract. This performs source auditing only; it
does not download datasets, execute training/inference, or substitute runtime
validation.
"""
from __future__ import annotations

from typing import Any

import account_wide_training_control_audit as base
import account_wide_training_control_audit_v32 as prior

CANONICAL_BOOTSTRAP_COMMIT = "985379f540edc92517955f07c3f81ef942364917"
CANONICAL_BOOTSTRAP_BLOB = "f100cf8363b9ebb45a4ae53e7260a3924e20e134"
CANONICAL_ADAPTER_COMMIT = "c1fca624be1e6a79fe65f8f651dc05a2ca6ef1cb"
CANONICAL_ADAPTER_BLOB = "3b38b47e2deceef0ff1f52beb67cee3b2a4156f9"
CERTIFICATE_SCHEMA = 33
SCIENTIFIC_ONTOLOGY_V33_SCHEMA = 1


def _remote(repo_row: dict[str, Any]) -> dict[str, Any]:
    prior.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    prior.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    prior.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    prior.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    return prior._remote(repo_row)


def _strict(report: dict[str, Any]) -> list[str]:
    errors = list(prior._strict(report))
    if report.get("scientific_ontology_v33_schema") != SCIENTIFIC_ONTOLOGY_V33_SCHEMA:
        errors.append(
            "scientific_ontology_v33_schema="
            f"{report.get('scientific_ontology_v33_schema')!r}"
        )
    if report.get("role_paradigm_protocol_cartesian_products_invented") is not False:
        errors.append("v33 role/paradigm/protocol Cartesian-product guard missing")
    controls = report.get("strict_controls") or {}
    if not isinstance(controls, dict) or not bool(
        controls.get("require_role_paradigm_protocol_accounting")
    ):
        errors.append(
            "strict control disabled or absent: require_role_paradigm_protocol_accounting"
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
