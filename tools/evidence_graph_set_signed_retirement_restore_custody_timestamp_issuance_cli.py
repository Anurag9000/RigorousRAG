"""Operator CLI for crash-recoverable custody timestamp issuance."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority_readonly import (
    ReadOnlyCustodyTimestampAuthorityRegistry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_contracts import (
    timestamp_output_path_digest,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_readonly import (
    ReadOnlyCustodyTimestampIssuanceJournal,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_reconcile import (
    CustodyTimestampIssuanceRecoveryError,
    execute_custody_timestamp_issuance,
    seed_custody_timestamp_issuance,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_runtime import (
    get_custody_timestamp_issuance_journal,
)

_DEFAULT_ISSUANCE_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_issuances.sqlite3"
)
_DEFAULT_AUTHORITY_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_authorities.sqlite3"
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _issuance_path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_ISSUANCE_DB_PATH",
        _DEFAULT_ISSUANCE_PATH,
    )


def _authority_path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH",
        _DEFAULT_AUTHORITY_PATH,
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "issuance_id": value.issuance_id,
        "owner_id": value.owner_id,
        "authority_id": value.authority_id,
        "key_id": value.key_id,
        "serial": value.serial,
        "attestation_digest": value.attestation_digest,
        "output_path_digest": value.output_path_digest,
        "state": value.state,
        "phase": value.phase,
        "attempt_count": value.attempt_count,
        "max_attempts": value.max_attempts,
        "lease_owner_present": value.lease_owner is not None,
        "lease_expires_at": value.lease_expires_at,
        "verification_digest": value.verification_digest,
        "failure_type": value.failure_type,
        "created_at": value.created_at,
        "updated_at": value.updated_at,
        "completed_at": value.completed_at,
        "contains_attestation_signature": False,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_timestamp_issuance_cli"
        ),
        description=(
            "Prepare, persist, and publish one custody timestamp serial with crash "
            "recovery. Private keys are used only during seed."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed")
    seed.add_argument("--owner-id", required=True)
    seed.add_argument("--authority-id", required=True)
    seed.add_argument("--key-id", required=True)
    seed.add_argument("--authority-private-key-path", required=True)
    seed.add_argument("--signed-envelope-path", required=True)
    seed.add_argument("--custody-signer-public-key-path", required=True)
    seed.add_argument("--output-path", required=True)
    seed.add_argument("--confirm-output-path-digest", required=True)
    seed.add_argument("--max-attempts", type=int, default=3)
    seed.add_argument("--issuance-db-path")
    seed.add_argument("--authority-db-path")

    execute = commands.add_parser("execute")
    execute.add_argument("issuance_id")
    execute.add_argument("--output-path", required=True)
    execute.add_argument("--worker-id", required=True)
    execute.add_argument("--lease-seconds", type=int, default=60)
    execute.add_argument("--issuance-db-path")
    execute.add_argument("--authority-db-path")

    status = commands.add_parser("status")
    status.add_argument("issuance_id")
    status.add_argument("--issuance-db-path")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument(
        "--state",
        choices=("planned", "running", "completed", "failed", "cancelled"),
    )
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--issuance-db-path")

    retry = commands.add_parser("retry")
    retry.add_argument("issuance_id")
    retry.add_argument("--owner-id", required=True)
    retry.add_argument("--confirm-issuance-id", required=True)
    retry.add_argument("--issuance-db-path")

    cancel = commands.add_parser("cancel")
    cancel.add_argument("issuance_id")
    cancel.add_argument("--owner-id", required=True)
    cancel.add_argument("--confirm-issuance-id", required=True)
    cancel.add_argument("--issuance-db-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "seed":
            computed = timestamp_output_path_digest(args.output_path)
            if computed != args.confirm_output_path_digest:
                raise ValueError("timestamp output path confirmation differs.")
            authority = ReadOnlyCustodyTimestampAuthorityRegistry(
                _authority_path(args.authority_db_path)
            )
            journal = get_custody_timestamp_issuance_journal(
                _issuance_path(args.issuance_db_path)
            )
            attempt, _attestation = seed_custody_timestamp_issuance(
                journal=journal,
                registry=authority,
                owner_id=args.owner_id,
                authority_id=args.authority_id,
                key_id=args.key_id,
                authority_private_key_path=args.authority_private_key_path,
                signed_envelope_path=args.signed_envelope_path,
                custody_signer_public_key_path=args.custody_signer_public_key_path,
                output_path=args.output_path,
                confirm_output_path_digest=args.confirm_output_path_digest,
                max_attempts=args.max_attempts,
            )
            _print(
                {
                    **_summary(attempt),
                    "issuance_journal_mutation_performed": True,
                    "attestation_output_created": False,
                    "private_key_material_stored": False,
                }
            )
            return 0
        if args.command == "execute":
            result = execute_custody_timestamp_issuance(
                args.issuance_id,
                worker_id=args.worker_id,
                lease_seconds=args.lease_seconds,
                output_path=args.output_path,
                journal=get_custody_timestamp_issuance_journal(
                    _issuance_path(args.issuance_db_path)
                ),
                registry=ReadOnlyCustodyTimestampAuthorityRegistry(
                    _authority_path(args.authority_db_path)
                ),
            )
            _print(
                {
                    "issuance_id": result.issuance_id,
                    "serial": result.serial,
                    "state": result.state,
                    "phase": result.phase,
                    "attestation_digest": result.attestation_digest,
                    "output_path_digest": result.output_path_digest,
                    "verification_digest": result.verification_digest,
                    "attempt_count": result.attempt_count,
                    "output_created": result.output_created,
                    "existing_exact_output_reused": (
                        result.existing_exact_output_reused
                    ),
                    "issuance_journal_mutation_performed": True,
                    "contains_attestation_signature": False,
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                }
            )
            return 0
        issuance_path = _issuance_path(getattr(args, "issuance_db_path", None))
        if args.command == "status":
            value = ReadOnlyCustodyTimestampIssuanceJournal(issuance_path).get(
                args.issuance_id
            )
            _print(
                {
                    **_summary(value),
                    "issuance_journal_mutation_performed": False,
                }
            )
            return 0
        if args.command == "list":
            values = ReadOnlyCustodyTimestampIssuanceJournal(issuance_path).list(
                owner_id=args.owner_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "state": args.state,
                    "count": len(values),
                    "items": [_summary(value) for value in values],
                    "issuance_journal_mutation_performed": False,
                    "contains_attestation_signatures": False,
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                }
            )
            return 0
        if args.confirm_issuance_id != args.issuance_id:
            raise ValueError("timestamp issuance confirmation differs.")
        journal = get_custody_timestamp_issuance_journal(issuance_path)
        if args.command == "retry":
            value = journal.retry(
                args.issuance_id,
                owner_id=args.owner_id,
                confirm_issuance_id=args.confirm_issuance_id,
                now=time.time(),
            )
        else:
            value = journal.cancel(
                args.issuance_id,
                owner_id=args.owner_id,
                confirm_issuance_id=args.confirm_issuance_id,
                now=time.time(),
            )
        _print({**_summary(value), "issuance_journal_mutation_performed": True})
        return 0
    except CustodyTimestampIssuanceRecoveryError as exc:
        _print(
            {
                "error": "timestamp_issuance_failed",
                "issuance_id": exc.issuance_id,
                "state": exc.state,
                "phase": exc.phase,
                "contains_private_key_material": False,
                "contains_raw_paths": False,
            },
            stream=sys.stderr,
        )
        return 1
    except PermissionError:
        _print({"error": "not_authorized_or_untrusted"}, stream=sys.stderr)
        return 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
