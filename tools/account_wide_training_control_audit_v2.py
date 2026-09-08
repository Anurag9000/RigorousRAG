#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v33."""
from __future__ import annotations

import account_wide_training_control_audit_v33 as audit

CANONICAL_BOOTSTRAP_COMMIT = "985379f540edc92517955f07c3f81ef942364917"
CANONICAL_BOOTSTRAP_BLOB = "f100cf8363b9ebb45a4ae53e7260a3924e20e134"
CANONICAL_ADAPTER_COMMIT = "c1fca624be1e6a79fe65f8f651dc05a2ca6ef1cb"
CANONICAL_ADAPTER_BLOB = "3b38b47e2deceef0ff1f52beb67cee3b2a4156f9"
SCIENTIFIC_COMBINATION_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V31_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V32_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V33_SCHEMA = 1
DECLARATION_CLOSURE_SCHEMA = 1
CONTROLLER_GENERATION = 33


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
