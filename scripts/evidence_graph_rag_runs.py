#!/usr/bin/env python3
"""Compatibility entrypoint for resumable GraphRAG run storage."""

from tools.evidence_graph_rag_run_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
