#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the semantic source-closure v20 bootstrap."""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "056f649a4a24236d46808a26eeac905ee9ac479d"
CANONICAL_BOOTSTRAP_BLOB = "749abc8bf0b441f3aa2f33d8454d0bb07349e6c5"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
