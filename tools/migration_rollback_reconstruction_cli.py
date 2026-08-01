"""Verify encrypted rollback artifacts through typed in-memory reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.migration_cutover_preflight_runtime import (
    get_migration_cutover_preflight_store,
)
from tools.migration_rollback_artifact import rollback_key_from_environment
from tools.migration_rollback_reconstruction import reconstruct_rollback_snapshots
from tools.migration_rollback_runtime import get_migration_rollback_store
from tools.migration_types import digest, identifier


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_rollback_reconstruction_cli",
        description=(
            "Decrypt, validate and reconstruct public rollback snapshot types in "
            "memory. This command cannot write or restore authoritative state."
        ),
    )
    parser.add_argument("task_id")
    parser.add_argument("--preflight-digest")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        task_id = identifier(args.task_id, "task_id", 64)
        try:
            preflight = get_migration_cutover_preflight_store().read(
                task_id,
                preflight_digest=(
                    digest(args.preflight_digest, "preflight_digest")
                    if args.preflight_digest is not None
                    else None
                ),
            )
            payload, manifest = get_migration_rollback_store().load(
                preflight=preflight,
                key=rollback_key_from_environment(),
            )
        except FileNotFoundError:
            _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
            return 1
        reconstructed = reconstruct_rollback_snapshots(preflight, payload)
        _print(
            {
                "task_id": task_id,
                "preflight_digest": preflight.preflight_digest,
                "artifact_digest": manifest.artifact_digest,
                "source_sequence": reconstructed.generation.sequence,
                "source_profile_fingerprint": (
                    reconstructed.generation.profile_fingerprint
                ),
                "source_content_sha256": reconstructed.generation.content_sha256,
                "vector_rows": len(reconstructed.vector.ids),
                "sparse_generation": reconstructed.sparse.generation,
                "sparse_fields": len(reconstructed.sparse.fields),
                "typed_reconstruction_verified": True,
                "restore_performed": False,
                "mutation_performed": False,
            }
        )
        return 0
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
