#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v28."""
from __future__ import annotations

import account_wide_training_control_audit_v28 as audit

CANONICAL_BOOTSTRAP_COMMIT = "1239f040b7b96dba77f056d16e1b4da9723a42c9"
CANONICAL_BOOTSTRAP_BLOB = "4ba01878e41c736c09503040c5498f66beffffa0"
CANONICAL_ADAPTER_COMMIT = "5ff5db05815f2ef9dff9616952156de3aae93338"
CANONICAL_ADAPTER_BLOB = "3909e826c4054932fc188914a983dac823c0171c"
SCIENTIFIC_COMBINATION_SCHEMA = 1
CONTROLLER_GENERATION = 28


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
