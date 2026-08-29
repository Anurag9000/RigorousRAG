#!/usr/bin/env python3
"""Universal controller v16: v15 plus scalable strict audit execution."""
from __future__ import annotations

import universal_training_controller_console as console
import universal_training_controller_console_defaults as console_defaults
import universal_training_controller_dag as dag
import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_job_catalog_v2 as job_catalog
import universal_training_controller_large_catalog as large_catalog
import universal_training_controller_opf_grace as opf_grace
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
    return dag.main()


if __name__ == "__main__":
    raise SystemExit(main())
