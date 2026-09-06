#!/usr/bin/env python3
"""Universal controller v21: exhaustive lifecycle on literal OPF_ADP.

v21 retains the byte-pinned OPF_ADP scheduler as the sole resource/admission
engine and adds exhaustive surrounding lifecycle discovery even for repositories
with authoritative explicit training catalogs.  Explicit catalog training jobs
remain authoritative; unrepresented dataset/preprocess, validation/evaluation,
testing/inference and metrics/aggregation executables are centrally enrolled.
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
import universal_training_controller_lifecycle_exhaustive as lifecycle_exhaustive
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
    opf_reference.install()
    profile_file.install()
    job_catalog.install()
    lifecycle.install()
    # Preserve explicit training matrices while making every surrounding
    # executable lifecycle phase discoverable and centrally schedulable.
    lifecycle_exhaustive.install()
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
    metrics.install()
    return deferred.main()


if __name__ == "__main__":
    raise SystemExit(main())
