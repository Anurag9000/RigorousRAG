#!/usr/bin/env python3
"""Universal controller v27: source-proven scientific combination closure.

v27 retains v26's literal OPF_ADP scheduler, exhaustive lifecycle orchestration,
source/exact-resume/early-stopping contracts, registry/member/config/selector
closure, and adds fail-closed accounting for repository-authored compatible
multi-component experiment tuples.  It never invents Cartesian products.

All resource scheduling remains exclusively in the unchanged byte-pinned OPF_ADP
massive-suite scheduler.
"""
from __future__ import annotations

import universal_training_controller_audit_infrastructure as audit_infrastructure
import universal_training_controller_combination_closure_v27 as combination_closure
import universal_training_controller_config_matrix as config_matrix
import universal_training_controller_console as console
import universal_training_controller_console_defaults as console_defaults
import universal_training_controller_dag_slicing as dag_slicing
import universal_training_controller_deferred_v4 as deferred
import universal_training_controller_early_stopping_wiring as early_stopping_wiring
import universal_training_controller_entrypoint_markers as entrypoint_markers
import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_inventory_scope as inventory_scope
import universal_training_controller_job_catalog_v2 as job_catalog
import universal_training_controller_large_catalog as large_catalog
import universal_training_controller_lifecycle as lifecycle
import universal_training_controller_lifecycle_affinity as lifecycle_affinity
import universal_training_controller_lifecycle_exhaustive as lifecycle_exhaustive
import universal_training_controller_metrics as metrics
import universal_training_controller_metrics_v2 as metrics_v2
import universal_training_controller_opf_grace as opf_grace
import universal_training_controller_opf_mechanism_audit as mechanism_audit
import universal_training_controller_opf_reference_v2 as opf_reference
import universal_training_controller_profile_file as profile_file
import universal_training_controller_registry_member_closure as registry_member_closure
import universal_training_controller_registry_scheduling as registry_scheduling
import universal_training_controller_restart_exact as restart_exact
import universal_training_controller_scientific_surface_v25 as scientific_surface
import universal_training_controller_selector_closure_v26 as selector_closure
import universal_training_controller_semantic_inventory as semantic_inventory
import universal_training_controller_source_contracts_v24 as source_contracts
import universal_training_controller_subcommands as subcommands
import universal_training_controller_training_contracts as training_contracts
import universal_training_controller_workload_closure as workload_closure


def main() -> int:
    # Establish one common static scientific vocabulary before any inventory is
    # built.  Every subsequent closure layer therefore audits the same universe.
    scientific_surface.install_primitives()
    selector_closure.install_primitives()

    opf_reference.install()
    profile_file.install()
    job_catalog.install()
    lifecycle.install()
    lifecycle_exhaustive.install()
    config_matrix.install()
    early_stopping_wiring.install()
    lifecycle_affinity.install()

    exact_resume.install()
    large_catalog.install()
    console.install()
    console_defaults.install()
    subcommands.install()
    entrypoint_markers.install()
    inventory_scope.install()
    audit_infrastructure.install()
    semantic_inventory.install()
    workload_closure.install()
    registry_member_closure.install()
    scientific_surface.install_contract()
    selector_closure.install_contract()
    combination_closure.install()
    restart_exact.install()
    opf_grace.install()
    registry_scheduling.install()
    training_contracts.install()
    source_contracts.install()
    mechanism_audit.install()

    dag_slicing.install()
    metrics.install()
    metrics_v2.install()
    return deferred.main()


if __name__ == "__main__":
    raise SystemExit(main())
