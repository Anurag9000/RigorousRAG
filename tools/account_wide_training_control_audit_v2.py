#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to the semantic source-closure v20 bootstrap.

The v1 auditor contains the full remote/local certification logic. This thin
versioned entrypoint updates only the immutable canonical bootstrap identity so
estate certification and per-repository launchers enforce the exact same bytes.
"""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "0ca36b9dec059eb6fd762f303158d118fae0b12b"
CANONICAL_BOOTSTRAP_BLOB = "31318081eda84ea27bcf10eb18e6467da0ec8dd1"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
