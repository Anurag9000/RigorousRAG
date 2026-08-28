#!/usr/bin/env python3
"""Universal controller v11: exact recovery + registries + curated scheduling + DAG."""
from __future__ import annotations

import universal_training_controller_console as console
import universal_training_controller_console_defaults as console_defaults
import universal_training_controller_dag as dag
import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_opf_grace as opf_grace
import universal_training_controller_registry_scheduling as registry_scheduling
import universal_training_controller_restart_exact as restart_exact
import universal_training_controller_subcommands as subcommands


def main() -> int:
    exact_resume.install()
    console.install()
    console_defaults.install()
    subcommands.install()
    restart_exact.install()
    opf_grace.install()
    registry_scheduling.install()
    return dag.main()


if __name__ == "__main__":
    raise SystemExit(main())
