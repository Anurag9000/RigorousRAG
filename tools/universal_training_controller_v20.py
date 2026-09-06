#!/usr/bin/env python3
"""Universal controller v20: full lifecycle orchestration on literal OPF_ADP.

This controller never reimplements the OPF resource scheduler.  It synchronizes
every controller/audit layer to one byte-pinned OPF_ADP reference, expands each
repository's concrete dataset -> train -> validation -> testing -> metrics DAG,
and then passes every dependency-ready wave to the unchanged literal OPF_ADP
scheduler.  All pressure, concurrency, pause/resume, retry, GPU-selection,
logging and persistent-state mechanisms therefore remain OPF-owned.
"""
from __future__ import annotations

import universal_training_controller_audit_infrastructure as audit_infrastructure
import universal_training_controller_console as console
import universal_training_controller_console_defaults as console_defaults
import universal_training_controller_deferred_v4 as deferred
import universal_training_controller_entrypoint_markers as entrypoint_markers
import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_inventory_scope as inventory_scope
import universal_training_controller_job_catalog_v2 as job_catalog
import universal_training_controller_large_catalog as large_catalog
import universal_training_controller_lifecycle as lifecycle
import universal_training_controller_metrics as metrics
import universal_training_controller_opf_grace as opf_grace
import universal_training_controller_opf_mechanism_audit as mechanism_audit
import universal_training_controller_opf_reference_v2 as opf_reference
import universal_training_controller_profile_file as profile_file
import universal_training_controller_registry_scheduling as registry_scheduling
import universal_training_controller_restart_exact as restart_exact
import universal_training_controller_semantic_inventory as semantic_inventory
import universal_training_controller_subcommands as subcommands
import universal_training_controller_training_contracts as training_contracts


def main() -> int:
    # Must precede every install that can invoke current._configure_reference or
    # build a mechanism certificate.
    opf_reference.install()
    profile_file.install()
    job_catalog.install()
    # Expand all repository lifecycle phases before metadata/recovery audits are
    # installed so every generated job receives the same strict contracts.
    lifecycle.install()
    exact_resume.install()
    large_catalog.install()
    console.install()
    console_defaults.install()
    subcommands.install()
    entrypoint_markers.install()
    inventory_scope.install()
    audit_infrastructure.install()
    semantic_inventory.install()
    restart_exact.install()
    opf_grace.install()
    registry_scheduling.install()
    training_contracts.install()
    mechanism_audit.install()
    # Metrics/manifest indexing wraps the dependency executor only; it never
    # participates in resource admission or process control.
    metrics.install()
    return deferred.main()


if __name__ == "__main__":
    raise SystemExit(main())
