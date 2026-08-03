"""Operator CLI for external restore chain-of-custody manifests."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_artifact_readonly import (
    ReadOnlyRestoreCustodyArtifactJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_export import (
    authenticate_restore_chain_of_custody,
    export_restore_chain_of_custody,
    verify_authenticated_restore_chain_of_custody,
    verify_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_readonly import (
    ReadOnlySignedRetirementRestoreCustodyStore,
)
from tools.evidence_graph_set_signed_retirement_restore_hold_readonly import (
    ReadOnlySignedRetirementRestoreHoldStore,
)
from tools.evidence_graph_set_signed_retirement_restore_intent_readonly import (
    ReadOnlySignedRetirementRestoreIntentJournal,
)

_DEFAULT_RESTORE_DB = "data/evidence_graph_set_signed_retirement_restores.sqlite3"
_DEFAULT_CUSTODY_DB = "data/evidence_graph_set_signed_retirement_custody.sqlite3"
_DEFAULT_ARTIFACT_DB = (
    "data/evidence_graph_set_signed_retirement_custody_artifacts.sqlite3"
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _manifest_summary(value: Any, *, authenticated: bool = False) -> dict[str, Any]:
    manifest = value.manifest if authenticated else value
    return {
        "owner_id": manifest.owner_id,
        "restore_id": manifest.restore_id,
        "snapshot_digest": manifest.snapshot_digest,
        "target_path_digest": manifest.target_path_digest,
        "custody_id": manifest.custody_id,
        "chain_digest": manifest.chain_digest,
        "artifact_pair_count": len(manifest.artifacts),
        "legal_hold_status": manifest.legal_hold_status,
        "generated_at": manifest.generated_at,
        "authenticated": authenticated,
        "algorithm": value.algorithm if authenticated else None,
        "key_id": value.key_id if authenticated else None,
        "contains_source_text": False,
        "contains_assertion_secrets": False,
        "contains_raw_paths": False,
        "mutation_performed": False,
        "restore_performed": False,
        "import_performed": False,
        "deletion_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_export_cli"
        ),
        description=(
            "Export or verify complete external restore chain-of-custody manifests. "
            "No command imports, restores, overwrites, or deletes state."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export")
    export.add_argument("--restore-id", required=True)
    export.add_argument("--snapshot", required=True)
    export.add_argument("--target-db-path", required=True)
    export.add_argument("--backup-path", required=True)
    export.add_argument("--pre-receipt-path", required=True)
    export.add_argument("--post-receipt-path", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--restore-db-path")
    export.add_argument("--custody-db-path")
    export.add_argument("--artifact-db-path")
    export.add_argument("--hold-db-path")
    export.add_argument("--limit", type=int, default=10_000)

    verify = commands.add_parser("verify")
    verify.add_argument("manifest_path")

    authenticate = commands.add_parser("authenticate")
    authenticate.add_argument("manifest_path")
    authenticate.add_argument("--output", required=True)
    authenticate.add_argument("--key-id", required=True)
    authenticate.add_argument("--key-path", required=True)

    verify_authenticated = commands.add_parser("verify-authenticated")
    verify_authenticated.add_argument("envelope_path")
    verify_authenticated.add_argument("--key-path", required=True)
    verify_authenticated.add_argument("--expected-key-id")
    return parser


def _environment(value: str | None, variable: str, default: str) -> str:
    return value or os.getenv(variable, default)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "export":
            restore_store = ReadOnlySignedRetirementRestoreIntentJournal(
                _environment(
                    args.restore_db_path,
                    "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_RESTORE_DB_PATH",
                    _DEFAULT_RESTORE_DB,
                )
            )
            custody_store = ReadOnlySignedRetirementRestoreCustodyStore(
                _environment(
                    args.custody_db_path,
                    "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_DB_PATH",
                    _DEFAULT_CUSTODY_DB,
                )
            )
            artifact_store = ReadOnlyRestoreCustodyArtifactJournal(
                _environment(
                    args.artifact_db_path,
                    "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_CUSTODY_ARTIFACT_DB_PATH",
                    _DEFAULT_ARTIFACT_DB,
                )
            )
            hold_path = args.hold_db_path or os.getenv(
                "EVIDENCE_GRAPH_SET_SIGNED_RETIREMENT_HOLD_DB_PATH"
            )
            hold_store = (
                None
                if not hold_path
                else ReadOnlySignedRetirementRestoreHoldStore(hold_path)
            )
            manifest = export_restore_chain_of_custody(
                restore_id=args.restore_id,
                snapshot_path=args.snapshot,
                target_db_path=args.target_db_path,
                backup_path=args.backup_path,
                pre_receipt_path=args.pre_receipt_path,
                post_receipt_path=args.post_receipt_path,
                restore_journal=restore_store,
                custody_store=custody_store,
                artifact_journal=artifact_store,
                hold_store=hold_store,
                output_path=args.output,
                limit=args.limit,
            )
            payload = _manifest_summary(manifest)
            payload["output_created"] = True
            _print(payload)
            return 0
        if args.command == "verify":
            manifest = verify_restore_chain_of_custody(args.manifest_path)
            _print(_manifest_summary(manifest))
            return 0
        if args.command == "authenticate":
            envelope = authenticate_restore_chain_of_custody(
                manifest_path=args.manifest_path,
                output_path=args.output,
                key_id=args.key_id,
                key_path=args.key_path,
            )
            payload = _manifest_summary(envelope, authenticated=True)
            payload["output_created"] = True
            _print(payload)
            return 0
        if args.command == "verify-authenticated":
            envelope = verify_authenticated_restore_chain_of_custody(
                envelope_path=args.envelope_path,
                key_path=args.key_path,
                expected_key_id=args.expected_key_id,
            )
            _print(_manifest_summary(envelope, authenticated=True))
            return 0
        raise ValueError("unsupported custody export command.")
    except PermissionError:
        _print({"error": "authentication_failed"}, stream=sys.stderr)
        return 1
    except (
        FileExistsError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
