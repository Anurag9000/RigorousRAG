#!/usr/bin/env python3
"""Account-wide training-control auditor pinned to scientific combination v27."""
from __future__ import annotations

import account_wide_training_control_audit as audit

# tools/universal_training_controller_entry.py at this immutable host commit.
CANONICAL_BOOTSTRAP_COMMIT = "a5caedea0618ecce1cef38e4e9b7d4fe14fe76ab"
CANONICAL_BOOTSTRAP_BLOB = "3acfd37ae22e887b5a7bcd42b04c9a288593abae"
CANONICAL_ADAPTER_COMMIT = "ab919968985f72d68863a2647427736eb9175a22"
CANONICAL_ADAPTER_BLOB = "9e7d15b7cb801fa73e9281de1829ff386772c0db"
SCIENTIFIC_COMBINATION_SCHEMA = 1


def main() -> int:
    audit.CANONICAL_BOOTSTRAP_COMMIT = CANONICAL_BOOTSTRAP_COMMIT
    audit.CANONICAL_BOOTSTRAP_BLOB = CANONICAL_BOOTSTRAP_BLOB
    # Newer base auditors may expose adapter identities; assigning conditionally
    # keeps this wrapper backwards-compatible while making the intended account
    # contract explicit in source.
    if hasattr(audit, "CANONICAL_ADAPTER_COMMIT"):
        audit.CANONICAL_ADAPTER_COMMIT = CANONICAL_ADAPTER_COMMIT
    if hasattr(audit, "CANONICAL_ADAPTER_BLOB"):
        audit.CANONICAL_ADAPTER_BLOB = CANONICAL_ADAPTER_BLOB
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
