#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the latest OPF contract bootstrap."""
from __future__ import annotations

import account_wide_training_control_audit as audit

# tools/universal_training_controller_entry.py at this immutable host commit.
CANONICAL_BOOTSTRAP_COMMIT = "a3d25bd282d8e2e0d9f2cb3730d4d036703a39ad"
CANONICAL_BOOTSTRAP_BLOB = "2515a8a2fc2dbed7334a4240438dc095f19636d2"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
