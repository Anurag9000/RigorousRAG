#!/usr/bin/env python3
"""Entrypoint for read-only signed publication transition auditing."""

from tools.evidence_graph_set_signed_transition_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
