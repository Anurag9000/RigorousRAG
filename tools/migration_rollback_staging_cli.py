"""Verify encrypted rollback snapshots in isolated non-authoritative staging."""

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
from tools.migration_rollback_staging import verify_in_isolated_staging
from tools.migration_types import digest, identifier


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_rollback_staging_cli",
        description=(
            "Decrypt and reconstruct a rollback artifact, stage it only in an "
            "isolated process-local store, and re-snapshot exact identities. "
            "This command cannot write authoritative state."
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
        verification = verify_in_isolated_staging(preflight, reconstructed)
        _print(
            {
                "task_id": task_id,
                "preflight_digest": preflight.preflight_digest,
                "artifact_digest": manifest.artifact_digest,
                "staging_id": verification.staging_id,
                "verification_digest": verification.verification_digest,
                "source_sequence": verification.source_sequence,
                "source_profile_fingerprint": verification.source_profile_fingerprint,
                "source_content_sha256": verification.source_content_sha256,
                "vector_snapshot_digest": verification.vector_snapshot_digest,
                "sparse_snapshot_digest": verification.sparse_snapshot_digest,
                "vector_rows": verification.vector_rows,
                "sparse_generation": verification.sparse_generation,
                "sparse_fields": verification.sparse_fields,
                "staging_verified": True,
                "staging_scope": "process_local_non_authoritative",
                "staging_mutation_performed": True,
                "authoritative_mutation_performed": False,
                "restore_performed": False,
                "cutover_performed": False,
            }
        )
        return 0
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
