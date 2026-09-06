#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to exhaustive lifecycle v21."""
from __future__ import annotations

import account_wide_training_control_audit as audit

# tools/universal_training_controller_entry.py at this immutable host commit.
CANONICAL_BOOTSTRAP_COMMIT = "471388dc1d9fa50eeae6c116a1508503b516c92d"
CANONICAL_BOOTSTRAP_BLOB = "14d2fee4e0d505578e97a295dfb662ece82dbee7"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
