#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v32."""
from __future__ import annotations

import account_wide_training_control_audit_v32 as audit

CANONICAL_BOOTSTRAP_COMMIT = "e16946686d09a2b2c3afdba5a18162d6f0aaeaf9"
CANONICAL_BOOTSTRAP_BLOB = "e794a90b6d681a7b275f92deadce82861f303bbf"
CANONICAL_ADAPTER_COMMIT = "edff50ab3adecb792c64a3fa98c906a73b0fb9d5"
CANONICAL_ADAPTER_BLOB = "7f73a479548c3417bdecf58bb9083e5226ba606a"
SCIENTIFIC_COMBINATION_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V31_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V32_SCHEMA = 1
DECLARATION_CLOSURE_SCHEMA = 1
CONTROLLER_GENERATION = 32


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
