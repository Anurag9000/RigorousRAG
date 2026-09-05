#!/usr/bin/env python3
"""Universal controller v20: v19 orchestration on current literal OPF_ADP.

v20 changes no resource-scheduling algorithm.  It first synchronizes every
controller/audit layer to one byte-pinned OPF_ADP reference, then installs the
same repository-job discovery, exact-resume, DAG, semantic-inventory,
mechanism-audit and deferred fan-out layers used by v19.  Concrete jobs are
still executed by the unchanged literal OPF_ADP scheduler.
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
    return deferred.main()


if __name__ == "__main__":
    raise SystemExit(main())
