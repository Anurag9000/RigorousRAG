#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v30."""
from __future__ import annotations

import account_wide_training_control_audit_v30 as audit

CANONICAL_BOOTSTRAP_COMMIT = "df5cd8e3b78451c3e9a134cb685d2f90902d8925"
CANONICAL_BOOTSTRAP_BLOB = "ecf809be32092aac6e585e15edb6b5a6f90798bc"
CANONICAL_ADAPTER_COMMIT = "d9c350e3c2d6b9d1ee418de3a9b4786ed5246a4c"
CANONICAL_ADAPTER_BLOB = "c96f5f2f6a754f4ea4125bd38334c2ee6e76ad4b"
SCIENTIFIC_COMBINATION_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_SCHEMA = 1
DECLARATION_CLOSURE_SCHEMA = 1
CONTROLLER_GENERATION = 30


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
