#!/usr/bin/env python3
"""Account-wide source-only auditor pinned to source-proven controller v24.

This wrapper evaluates every repository against the immutable v24 bootstrap:
member-complete workload closure, existing local command/config targets,
source-proven interruption-exact training resume, source-proven semantic early
stopping (or a well-formed non-applicable exemption), DAG enforcement, metrics
indexing, and literal OPF_ADP scheduler parity.  It executes no model training or
dataset campaign itself.
"""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "8124ebdda8380263923e439fcce79b3060b9d087"
CANONICAL_BOOTSTRAP_BLOB = "9dd4ea79dd9f4abfb9adabefd96925b66f2fe99e"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
