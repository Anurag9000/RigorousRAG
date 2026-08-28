#!/usr/bin/env python3
"""Universal training controller v7: graph + package registry + exact resume + DAG."""
from __future__ import annotations

import universal_training_controller_console as console
import universal_training_controller_dag as dag
import universal_training_controller_exact_resume as exact_resume


def main() -> int:
    # Exact-resume patches the checkpoint contract used by the graph audit;
    # package-registry discovery then patches the catalog/report symbols that
    # DAG installs into the base adapter. Scheduling itself remains the literal
    # pinned OPF_ADP implementation.
    exact_resume.install()
    console.install()
    return dag.main()


if __name__ == "__main__":
    raise SystemExit(main())
