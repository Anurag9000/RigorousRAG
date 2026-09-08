#!/usr/bin/env python3
"""Canonical account-wide source certificate entrypoint for training control v29."""
from __future__ import annotations

import account_wide_training_control_audit_v29 as audit

CANONICAL_BOOTSTRAP_COMMIT = "5fecd0732a6f57741a1ea61b94f196965af5411f"
CANONICAL_BOOTSTRAP_BLOB = "2c76bc3c4b70c511fc4e94aaa1bd0dd145bbadb4"
CANONICAL_ADAPTER_COMMIT = "f899ac190085228cc95f47725a36bfd105822374"
CANONICAL_ADAPTER_BLOB = "82a10e0a6e221dbf90de92d56a8b87d0f270935f"
SCIENTIFIC_COMBINATION_SCHEMA = 1
SCIENTIFIC_ONTOLOGY_SCHEMA = 1
CONTROLLER_GENERATION = 29


def main() -> int:
    return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
