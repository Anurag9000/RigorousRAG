#!/usr/bin/env python3
"""Universal training controller v6: graph audit + exact resume + DAG waves."""
from __future__ import annotations

import universal_training_controller_exact_resume as exact_resume
import universal_training_controller_dag as dag


def main() -> int:
    exact_resume.install()
    return dag.main()


if __name__ == "__main__":
    raise SystemExit(main())
