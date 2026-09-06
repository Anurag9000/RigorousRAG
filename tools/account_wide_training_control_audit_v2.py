#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the full lifecycle bootstrap."""
from __future__ import annotations

import account_wide_training_control_audit as audit

# tools/universal_training_controller_entry.py at this immutable host commit.
CANONICAL_BOOTSTRAP_COMMIT = "7b9ceb12d6c5fdef33eefd73eaea4c027b941737"
CANONICAL_BOOTSTRAP_BLOB = "4ecb86674c3baa91c88ff57a8699decce26c528d"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())