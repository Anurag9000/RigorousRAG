#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the source-closure v20 bootstrap.

The v1 auditor contains the full remote/local certification logic.  This thin
versioned entrypoint updates only the immutable canonical bootstrap identity so
estate certification and per-repository launchers enforce the exact same bytes.
"""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "b8af2de3b35cd31f351b5d211d2384c7e96b7ff7"
CANONICAL_BOOTSTRAP_BLOB = "16db0d4c6fe886fc204d23b54b61bb7e6590dc13"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
