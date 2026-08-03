"""Governed CLI for custody timestamp authority registration and attestations."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import (
    load_relation_review_actor,
    require_relation_review_actor,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    _load_public,
    _public_fingerprint,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority import (
    issue_governed_custody_timestamp,
    verify_governed_custody_timestamp,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_authority_readonly import (
    ReadOnlyCustodyTimestampAuthorityRegistry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_timestamp_runtime import (
    get_custody_timestamp_authority_registry,
)

_DEFAULT_PATH = (
    "data/evidence_graph_set_signed_retirement_custody_timestamp_authorities.sqlite3"
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_TIMESTAMP_AUTHORITY_DB_PATH",
        _DEFAULT_PATH,
    )


def _record_summary(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "authority_id": value.authority_id,
        "key_id": value.key_id,
        "algorithm": value.algorithm,
        "public_key_sha256": value.public_key_sha256,
        "state": value.state,
        "registered_binding_method": value.registered_binding_method,
        "registered_binding_digest": value.registered_binding_digest,
        "registered_at": value.registered_at,
        "retired_binding_method": value.retired_binding_method,
        "retired_binding_digest": value.retired_binding_digest,
        "retired_at": value.retired_at,
        "record_digest": value.record_digest,
        "eligible_for_new_attestations": value.state == "active",
        "contains_actor_id": False,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def _attestation_summary(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "authority_id": value.authority_id,
        "key_id": value.key_id,
        "public_key_sha256": value.public_key_sha256,
        "custody_envelope_sha256": value.custody_envelope_sha256,
        "custody_manifest_digest": value.custody_manifest_digest,
        "custody_chain_digest": value.custody_chain_digest,
        "asserted_at": value.asserted_at,
        "nonce_sha256": value.nonce_sha256,
        "serial": value.serial,
        "timestamp_attestation_valid": True,
        "rfc3161_token": False,
        "hardware_clock_proven": False,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_timestamp_cli"
        ),
        description=(
            "Govern public custody timestamp-authority keys and issue or verify "
            "Ed25519 authority attestations. These are not RFC 3161 tokens."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register")
    register.add_argument("--owner-id", required=True)
    register.add_argument("--authority-id", required=True)
    register.add_argument("--key-id", required=True)
    register.add_argument("--public-key-path", required=True)
    register.add_argument("--confirm-public-key-sha256", required=True)
    register.add_argument("--actor-id")
    register.add_argument("--registry-db-path")

    retire = commands.add_parser("retire")
    retire.add_argument("--owner-id", required=True)
    retire.add_argument("--authority-id", required=True)
    retire.add_argument("--key-id", required=True)
    retire.add_argument("--confirm-key-id", required=True)
    retire.add_argument("--actor-id")
    retire.add_argument("--registry-db-path")

    status = commands.add_parser("status")
    status.add_argument("--owner-id", required=True)
    status.add_argument("--authority-id", required=True)
    status.add_argument("--key-id", required=True)
    status.add_argument("--registry-db-path")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state", choices=("active", "retired"))
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--registry-db-path")

    issue = commands.add_parser("issue-governed")
    issue.add_argument("signed_envelope_path")
    issue.add_argument("--custody-signer-public-key-path", required=True)
    issue.add_argument("--owner-id", required=True)
    issue.add_argument("--authority-id", required=True)
    issue.add_argument("--key-id", required=True)
    issue.add_argument("--authority-private-key-path", required=True)
    issue.add_argument("--output", required=True)
    issue.add_argument("--registry-db-path")

    verify = commands.add_parser("verify-governed")
    verify.add_argument("attestation_path")
    verify.add_argument("--signed-envelope-path", required=True)
    verify.add_argument("--custody-signer-public-key-path", required=True)
    verify.add_argument("--authority-public-key-path", required=True)
    verify.add_argument("--owner-id", required=True)
    verify.add_argument("--authority-id", required=True)
    verify.add_argument("--key-id", required=True)
    verify.add_argument("--maximum-future-seconds", type=float, default=300.0)
    verify.add_argument("--registry-db-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        registry_path = _path(getattr(args, "registry_db_path", None))
        if args.command == "register":
            fingerprint = _public_fingerprint(_load_public(args.public_key_path))
            if fingerprint != args.confirm_public_key_sha256:
                raise ValueError("public-key fingerprint confirmation differs.")
            actor = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            value = get_custody_timestamp_authority_registry(registry_path).register(
                owner_id=args.owner_id,
                authority_id=args.authority_id,
                key_id=args.key_id,
                public_key_path=args.public_key_path,
                actor=actor,
            )
            _print(
                {
                    **_record_summary(value),
                    "registry_mutation_performed": True,
                    "key_material_mutation_performed": False,
                    "key_deletion_performed": False,
                }
            )
            return 0
        if args.command == "retire":
            if args.confirm_key_id != args.key_id:
                raise ValueError("timestamp authority retirement confirmation differs.")
            actor = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            value = get_custody_timestamp_authority_registry(registry_path).retire(
                owner_id=args.owner_id,
                authority_id=args.authority_id,
                key_id=args.key_id,
                confirm_key_id=args.confirm_key_id,
                actor=actor,
            )
            _print(
                {
                    **_record_summary(value),
                    "registry_mutation_performed": True,
                    "key_material_mutation_performed": False,
                    "key_deletion_performed": False,
                }
            )
            return 0

        read_only = ReadOnlyCustodyTimestampAuthorityRegistry(registry_path)
        if args.command == "status":
            value = read_only.get(
                owner_id=args.owner_id,
                authority_id=args.authority_id,
                key_id=args.key_id,
            )
            _print({**_record_summary(value), "registry_mutation_performed": False})
            return 0
        if args.command == "list":
            values = read_only.list(
                owner_id=args.owner_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "state": args.state,
                    "count": len(values),
                    "items": [_record_summary(value) for value in values],
                    "registry_mutation_performed": False,
                    "key_material_mutation_performed": False,
                    "key_deletion_performed": False,
                    "contains_raw_paths": False,
                }
            )
            return 0
        if args.command == "issue-governed":
            attestation = issue_governed_custody_timestamp(
                registry=read_only,
                owner_id=args.owner_id,
                authority_id=args.authority_id,
                key_id=args.key_id,
                authority_private_key_path=args.authority_private_key_path,
                signed_envelope_path=args.signed_envelope_path,
                custody_signer_public_key_path=args.custody_signer_public_key_path,
                output_path=args.output,
            )
            _print(
                {
                    **_attestation_summary(attestation),
                    "attestation_created": True,
                    "registry_mutation_performed": False,
                    "key_material_mutation_performed": False,
                }
            )
            return 0
        attestation = verify_governed_custody_timestamp(
            registry=read_only,
            owner_id=args.owner_id,
            authority_id=args.authority_id,
            key_id=args.key_id,
            attestation_path=args.attestation_path,
            signed_envelope_path=args.signed_envelope_path,
            custody_signer_public_key_path=args.custody_signer_public_key_path,
            authority_public_key_path=args.authority_public_key_path,
            maximum_future_seconds=args.maximum_future_seconds,
        )
        _print(
            {
                **_attestation_summary(attestation),
                "historical_governance_window_valid": True,
                "registry_mutation_performed": False,
                "key_material_mutation_performed": False,
            }
        )
        return 0
    except PermissionError:
        _print({"error": "not_authorized_or_untrusted"}, stream=sys.stderr)
        return 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
