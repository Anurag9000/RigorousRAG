#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v34."""
from __future__ import annotations

import account_wide_training_control_audit_v34 as audit

CANONICAL_BOOTSTRAP_COMMIT = "142587b31ea4097e237cbe4ad756bb595c1591f7"
CANONICAL_BOOTSTRAP_BLOB = "0c7c6307ca97fbc502b511f845059f345712fc7d"
CANONICAL_ADAPTER_COMMIT = "a709cf605b818a61fdcce0e57c11a98459f6a9a1"
CANONICAL_ADAPTER_BLOB = "f174e72eebf00efb7f9de6e785210a9418b47450"
SCIENTIFIC_COMBINATION_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V31_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V32_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V33_SCHEMA = 1
RETAINED_TRAINING_CLOSURE_SCHEMA = 1
DECLARATION_CLOSURE_SCHEMA = 1
CONTROLLER_GENERATION = 34


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
