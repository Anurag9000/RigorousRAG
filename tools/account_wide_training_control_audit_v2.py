#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to scientific-surface v25."""
from __future__ import annotations

import account_wide_training_control_audit as audit

# tools/universal_training_controller_entry.py at this immutable host commit.
CANONICAL_BOOTSTRAP_COMMIT = "5e76048950a8c5531f213a5a7cb3ed71fceda48a"
CANONICAL_BOOTSTRAP_BLOB = "e538519e846ed4bb9af10fd1b3ad9afa1f40825c"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
