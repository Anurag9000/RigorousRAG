"""Capture/status/verify-only CLI for encrypted migration rollback artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.migration_cutover_preflight_runtime import (
    get_migration_cutover_preflight_store,
)
from tools.migration_rollback_artifact import (
    capture_rollback_payload,
    rollback_key_from_environment,
)
from tools.migration_rollback_runtime import get_migration_rollback_store
from tools.migration_runtime import get_migration_journal
from tools.migration_types import digest, identifier


def _print(payload: Any, *, stream: Any = None) -> None:
    destination = stream if stream is not None else sys.stdout
    destination.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _capture_snapshot(task: Any) -> Any:
    from tools.authoritative_document_index import capture_authoritative_document
    from tools.rag import get_rag_layer

    return capture_authoritative_document(
        owner_id=task.owner_id,
        doc_id=task.doc_id,
        rag=get_rag_layer(),
    )


def _summary(manifest: Any, *, verified: bool) -> dict[str, Any]:
    return {
        "task_id": manifest.task_id,
        "owner_id": manifest.owner_id,
        "doc_id": manifest.doc_id,
        "preflight_digest": manifest.preflight_digest,
        "rollback_identity_digest": manifest.rollback_identity_digest,
        "source_sequence": manifest.source_sequence,
        "source_profile_fingerprint": manifest.source_profile_fingerprint,
        "source_content_sha256": manifest.source_content_sha256,
        "vector_snapshot_digest": manifest.vector_snapshot_digest,
        "sparse_snapshot_digest": manifest.sparse_snapshot_digest,
        "plaintext_sha256": manifest.plaintext_sha256,
        "ciphertext_sha256": manifest.ciphertext_sha256,
        "plaintext_bytes": manifest.plaintext_bytes,
        "ciphertext_bytes": manifest.ciphertext_bytes,
        "algorithm": manifest.algorithm,
        "key_id": manifest.key_id,
        "aad_sha256": manifest.aad_sha256,
        "artifact_digest": manifest.artifact_digest,
        "created_at": manifest.created_at,
        "verified": verified,
        "restore_performed": False,
        "mutation_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.migration_rollback_cli",
        description=(
            "Capture and verify AES-GCM encrypted rollback artifacts. This CLI "
            "cannot restore snapshots or cut over live generations."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Capture one encrypted rollback artifact.")
    capture.add_argument("task_id")
    capture.add_argument("--preflight-digest")
    status = commands.add_parser("status", help="Read one rollback manifest without decrypting.")
    status.add_argument("task_id")
    status.add_argument("--preflight-digest")
    verify = commands.add_parser("verify", help="Decrypt and validate one rollback artifact.")
    verify.add_argument("task_id")
    verify.add_argument("--preflight-digest")
    remove = commands.add_parser(
        "remove",
        help="Remove one artifact only for failed or cancelled migration tasks.",
    )
    remove.add_argument("task_id")
    remove.add_argument("--preflight-digest", required=True)
    remove.add_argument("--confirm-task-id", required=True)
    remove.add_argument("--confirm-preflight-digest", required=True)
    return parser


def _preflight(task_id: str, selected: str | None) -> Any | None:
    try:
        return get_migration_cutover_preflight_store().read(
            task_id,
            preflight_digest=(
                digest(selected, "preflight_digest") if selected is not None else None
            ),
        )
    except FileNotFoundError:
        return None


def _capture(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    task = get_migration_journal().get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    preflight = _preflight(task_id, args.preflight_digest)
    if preflight is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    snapshot = _capture_snapshot(task)
    payload = capture_rollback_payload(preflight, snapshot)
    manifest = get_migration_rollback_store().write(
        preflight=preflight,
        payload=payload,
        key=rollback_key_from_environment(),
    )
    _print(_summary(manifest, verified=True))
    return 0


def _status(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    preflight = _preflight(task_id, args.preflight_digest)
    if preflight is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    try:
        manifest = get_migration_rollback_store().read_manifest(
            task_id,
            preflight.preflight_digest,
        )
    except FileNotFoundError:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    _print(_summary(manifest, verified=False))
    return 0


def _verify(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    preflight = _preflight(task_id, args.preflight_digest)
    if preflight is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    try:
        _payload, manifest = get_migration_rollback_store().load(
            preflight=preflight,
            key=rollback_key_from_environment(),
        )
    except FileNotFoundError:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    _print(_summary(manifest, verified=True))
    return 0


def _remove(args: argparse.Namespace) -> int:
    task_id = identifier(args.task_id, "task_id", 64)
    preflight_digest = digest(args.preflight_digest, "preflight_digest")
    confirmation_task = identifier(args.confirm_task_id, "confirm_task_id", 64)
    confirmation_preflight = digest(
        args.confirm_preflight_digest,
        "confirm_preflight_digest",
    )
    if confirmation_task != task_id or confirmation_preflight != preflight_digest:
        raise ValueError("rollback removal confirmation is not exact.")
    task = get_migration_journal().get(task_id)
    if task is None:
        _print({"error": "not_found", "task_id": task_id}, stream=sys.stderr)
        return 1
    if task.state not in {"failed", "cancelled"}:
        raise ValueError("rollback artifacts may be removed only for failed or cancelled tasks.")
    removed = get_migration_rollback_store().remove(task_id, preflight_digest)
    _print(
        {
            "task_id": task_id,
            "preflight_digest": preflight_digest,
            "removed": removed,
            "restore_performed": False,
            "mutation_performed": removed,
        }
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "capture":
            return _capture(args)
        if args.command == "status":
            return _status(args)
        if args.command == "verify":
            return _verify(args)
        if args.command == "remove":
            return _remove(args)
        raise ValueError("unsupported rollback artifact command.")
    except (OSError, ValueError, RuntimeError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
