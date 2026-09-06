#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to workload-closure v22."""
from __future__ import annotations

import account_wide_training_control_audit as audit

# tools/universal_training_controller_entry.py at this immutable host commit.
CANONICAL_BOOTSTRAP_COMMIT = "257c82a9686d5aeeee765c1ca5d8df168f80129e"
CANONICAL_BOOTSTRAP_BLOB = "a164fd06fc2800cd92243a394f2582d0617f814d"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
