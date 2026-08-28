#!/usr/bin/env python3
"""Universal controller v8: graph + package/CLI registries + exact resume + DAG."""
from __future__ import annotations

import universal_training_controller_console as console
import universal_training_controller_dag as dag
import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_subcommands as subcommands


def main() -> int:
    exact_resume.install()
    console.install()
    subcommands.install()
    return dag.main()


if __name__ == "__main__":
    raise SystemExit(main())
