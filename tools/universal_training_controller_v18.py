#!/usr/bin/env python3
"""Universal controller v18: v17 plus fail-closed post-producer fan-out.

All resource scheduling remains the literal pinned OPF_ADP implementation.  v18
adds only repository-job orchestration above that scheduler: deterministic
non-training producers may materialize an experiment dimension, after which
concrete jobs are frozen, fully audited and passed individually to the ordinary
DAG/OPF execution path.
"""
from __future__ import annotations

import universal_training_controller_console as console
import universal_training_controller_console_defaults as console_defaults
import universal_training_controller_deferred_v2 as deferred
import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_job_catalog_v2 as job_catalog
import universal_training_controller_large_catalog as large_catalog
import universal_training_controller_opf_grace as opf_grace
import universal_training_controller_opf_mechanism_audit as mechanism_audit
import universal_training_controller_profile_file as profile_file
import universal_training_controller_registry_scheduling as registry_scheduling
import universal_training_controller_restart_exact as restart_exact
import universal_training_controller_subcommands as subcommands
import universal_training_controller_training_contracts as training_contracts


def main() -> int:
    profile_file.install()
    job_catalog.install()
    exact_resume.install()
    large_catalog.install()
    console.install()
    console_defaults.install()
    subcommands.install()
    restart_exact.install()
    opf_grace.install()
    registry_scheduling.install()
    training_contracts.install()
    mechanism_audit.install()
    return deferred.main()


if __name__ == "__main__":
    raise SystemExit(main())
