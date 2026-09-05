#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the semantic source-closure v20 bootstrap."""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "ac27b17c8cb9551a463b04164949d8dfcfa060d4"
CANONICAL_BOOTSTRAP_BLOB = "e91ff436662b93927a4c34fbaf78700876cd34a6"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
