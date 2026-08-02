#!/usr/bin/env python3
"""Compatibility entrypoint for derived evidence-graph reconciliation jobs."""

from tools.evidence_graph_jobs_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
