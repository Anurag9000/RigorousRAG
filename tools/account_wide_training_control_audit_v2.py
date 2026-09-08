#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v31."""
from __future__ import annotations

import account_wide_training_control_audit_v31 as audit

CANONICAL_BOOTSTRAP_COMMIT = "6bbb504893699aac7ab461767f8170b4ee1797fc"
CANONICAL_BOOTSTRAP_BLOB = "8ee1a31f2368f3fa0fad7b0b2896fe6da4613bd3"
CANONICAL_ADAPTER_COMMIT = "04e875505372d70711709042e4322384f6ded9d6"
CANONICAL_ADAPTER_BLOB = "ec384d937fb75f5bf1ff257d135c3f28001c29cd"
SCIENTIFIC_COMBINATION_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_V31_SCHEMA = 1
DECLARATION_CLOSURE_SCHEMA = 1
CONTROLLER_GENERATION = 31


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
