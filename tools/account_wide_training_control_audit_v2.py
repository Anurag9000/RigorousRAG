#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the semantic source-closure v20 bootstrap."""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "8a9e9d5d042225c10477c3c7a5fe9fa64c59adbc"
CANONICAL_BOOTSTRAP_BLOB = "9504ca6b3d2c7bba60d47e4c209e8bd7da107a57"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
