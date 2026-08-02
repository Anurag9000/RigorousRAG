#!/usr/bin/env python3
"""Compatibility entrypoint for relation-review authorization auditing."""

from tools.evidence_graph_relation_authorization_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
