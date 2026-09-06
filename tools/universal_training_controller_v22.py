#!/usr/bin/env python3
"""Universal controller v22: exhaustive workload closure on literal OPF_ADP.

v22 keeps the exact byte-pinned OPF_ADP scheduler as the only resource/admission
engine.  It strengthens repository orchestration around that scheduler with:

* exhaustive surrounding lifecycle discovery while preserving authoritative
  training catalogs;
* conservative auto-enrolment of unambiguous repo-authored config experiments;
* activation of trainer-native early-stopping CLI controls when exposed;
* model/dataset/task/config-aware dependency affinity;
* fail-closed workload-surface closure in addition to learner-source closure;
* dependency-safe global start-index/job-limit slicing;
* global and per-job metric artifact ledgers.

No RAM/VRAM/concurrency/pause/resume/retry/device scheduling implementation lives
in this file or the added layers; ready jobs still execute through the unchanged
literal OPF scheduler.
"""
from __future__ import annotations

import universal_training_controller_audit_infrastructure as audit_infrastructure
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
import universal_training_controller_registry_scheduling as registry_scheduling
import universal_training_controller_restart_exact as restart_exact
import universal_training_controller_semantic_inventory as semantic_inventory
import universal_training_controller_subcommands as subcommands
import universal_training_controller_training_contracts as training_contracts
import universal_training_controller_workload_closure as workload_closure


def main() -> int:
    opf_reference.install()
    profile_file.install()
    job_catalog.install()
    lifecycle.install()
    lifecycle_exhaustive.install()
    # Expand only repository-authored, unambiguous config families; then wire
    # native early-stop controls and build the most specific safe dependencies.
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
    restart_exact.install()
    opf_grace.install()
    registry_scheduling.install()
    training_contracts.install()
    mechanism_audit.install()

    # Slicing is an orchestration concern. It selects the global DAG target plus
    # prerequisite closure, then removes slice flags before literal OPF waves.
    dag_slicing.install()
    metrics.install()
    metrics_v2.install()
    return deferred.main()


if __name__ == "__main__":
    raise SystemExit(main())
