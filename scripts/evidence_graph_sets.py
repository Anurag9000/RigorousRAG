#!/usr/bin/env python3
"""Compatibility entrypoint for read-only cross-document graph sets."""

from tools.evidence_graph_set_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
