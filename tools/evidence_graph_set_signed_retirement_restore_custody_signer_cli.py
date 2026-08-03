"""Governed operator CLI for custody signer registration and use."""

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
from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    verify_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    _load_private,
    _load_public,
    _public_fingerprint,
    sign_restore_chain_of_custody,
    verify_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_readonly import (
    ReadOnlyCustodySignerKeyRegistry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_runtime import (
    get_custody_signer_key_registry,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_custody_signers.sqlite3"


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _path(value: str | None) -> str:
    return value or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
        _DEFAULT_PATH,
    )


def _summary(value: Any) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "key_id": value.key_id,
        "issuer": value.issuer,
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
        "eligible_for_new_signatures": value.state == "active",
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_signer_cli"
        ),
        description=(
            "Register, retire, inspect, and use public Ed25519 custody signer "
            "identities. Private keys are never stored in the registry."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register")
    register.add_argument("--owner-id", required=True)
    register.add_argument("--key-id", required=True)
    register.add_argument("--issuer", required=True)
    register.add_argument("--public-key-path", required=True)
    register.add_argument("--confirm-public-key-sha256", required=True)
    register.add_argument("--actor-id")
    register.add_argument("--registry-db-path")

    retire = commands.add_parser("retire")
    retire.add_argument("--owner-id", required=True)
    retire.add_argument("--key-id", required=True)
    retire.add_argument("--confirm-key-id", required=True)
    retire.add_argument("--actor-id")
    retire.add_argument("--registry-db-path")

    status = commands.add_parser("status")
    status.add_argument("--owner-id", required=True)
    status.add_argument("--key-id", required=True)
    status.add_argument("--registry-db-path")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state", choices=("active", "retired"))
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--registry-db-path")

    sign = commands.add_parser("sign-governed")
    sign.add_argument("manifest_path")
    sign.add_argument("--owner-id", required=True)
    sign.add_argument("--key-id", required=True)
    sign.add_argument("--private-key-path", required=True)
    sign.add_argument("--output", required=True)
    sign.add_argument("--registry-db-path")

    verify = commands.add_parser("verify-registered")
    verify.add_argument("envelope_path")
    verify.add_argument("--owner-id", required=True)
    verify.add_argument("--key-id", required=True)
    verify.add_argument("--public-key-path", required=True)
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
            value = get_custody_signer_key_registry(registry_path).register(
                owner_id=args.owner_id,
                key_id=args.key_id,
                issuer=args.issuer,
                public_key_path=args.public_key_path,
                actor=actor,
            )
            payload = _summary(value)
            payload.update(
                {
                    "registry_mutation_performed": True,
                    "key_material_mutation_performed": False,
                    "key_deletion_performed": False,
                }
            )
            _print(payload)
            return 0
        if args.command == "retire":
            if args.confirm_key_id != args.key_id:
                raise ValueError("signer retirement confirmation differs.")
            actor = require_relation_review_actor(
                args.actor_id,
                binding=load_relation_review_actor(),
            )
            value = get_custody_signer_key_registry(registry_path).retire(
                owner_id=args.owner_id,
                key_id=args.key_id,
                confirm_key_id=args.confirm_key_id,
                actor=actor,
            )
            payload = _summary(value)
            payload.update(
                {
                    "registry_mutation_performed": True,
                    "key_material_mutation_performed": False,
                    "key_deletion_performed": False,
                }
            )
            _print(payload)
            return 0
        read_only = ReadOnlyCustodySignerKeyRegistry(registry_path)
        if args.command == "status":
            payload = _summary(
                read_only.get(owner_id=args.owner_id, key_id=args.key_id)
            )
            payload["registry_mutation_performed"] = False
            _print(payload)
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
                    "items": [_summary(value) for value in values],
                    "registry_mutation_performed": False,
                    "key_material_mutation_performed": False,
                    "key_deletion_performed": False,
                    "contains_raw_paths": False,
                }
            )
            return 0
        record = read_only.get(owner_id=args.owner_id, key_id=args.key_id)
        if args.command == "sign-governed":
            if record.state != "active":
                raise PermissionError("signer key is not active.")
            manifest = verify_restore_chain_of_custody(args.manifest_path)
            if manifest.owner_id != record.owner_id:
                raise PermissionError("manifest owner differs from signer registry.")
            private_key = _load_private(args.private_key_path)
            if _public_fingerprint(private_key.public_key()) != record.public_key_sha256:
                raise PermissionError("private key differs from signer registry.")
            envelope = sign_restore_chain_of_custody(
                manifest_path=args.manifest_path,
                output_path=args.output,
                key_id=record.key_id,
                private_key_path=args.private_key_path,
            )
            _print(
                {
                    **_summary(record),
                    "restore_id": envelope.manifest.restore_id,
                    "chain_digest": envelope.manifest.chain_digest,
                    "signature_created": True,
                    "registry_mutation_performed": False,
                    "key_material_mutation_performed": False,
                    "contains_source_text": False,
                }
            )
            return 0
        envelope = verify_signed_restore_chain_of_custody(
            envelope_path=args.envelope_path,
            public_key_path=args.public_key_path,
            expected_key_id=record.key_id,
            expected_public_key_sha256=record.public_key_sha256,
        )
        if envelope.manifest.owner_id != record.owner_id:
            raise PermissionError("signed manifest owner differs from registry.")
        _print(
            {
                **_summary(record),
                "restore_id": envelope.manifest.restore_id,
                "chain_digest": envelope.manifest.chain_digest,
                "signature_valid": True,
                "historical_verification_allowed": True,
                "registry_mutation_performed": False,
                "key_material_mutation_performed": False,
                "contains_source_text": False,
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
