#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to member-complete controller v23.

The underlying auditor remains diagnostic/source-accounting only.  This wrapper
updates its immutable bootstrap identity so every repository is evaluated against
the same v23 controller used by the current estate launchers: exhaustive
lifecycle/config closure, exact-resume/early-stopping contracts, DAG slicing,
metrics indexing, workload closure and fail-closed registry-member accounting.
Resource scheduling itself remains the literal byte-pinned OPF_ADP implementation.
"""
from __future__ import annotations

import account_wide_training_control_audit as audit

CANONICAL_BOOTSTRAP_COMMIT = "d7d4677e494a4d8500ad497e87dfde373793a02f"
CANONICAL_BOOTSTRAP_BLOB = "351f4135c72e3de75e4241bc5579ae443b2cac1f"


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
