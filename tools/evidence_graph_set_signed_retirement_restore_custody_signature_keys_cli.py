"""Operator CLI for governed Ed25519 custody signer public keys."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import load_relation_review_actor
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys import (
    register_custody_signer_key,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys_runtime import (
    get_custody_signer_key_registry,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _summary(value: Any, *, mutation: bool) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "key_id": value.key_id,
        "algorithm": value.algorithm,
        "public_key_sha256": value.public_key_sha256,
        "valid_from": value.valid_from,
        "valid_until": value.valid_until,
        "state": value.state,
        "registered_actor_id_digest": value.registered_actor_id_digest,
        "registered_binding_method": value.registered_binding_method,
        "registered_binding_digest": value.registered_binding_digest,
        "registered_at": value.registered_at,
        "retired_actor_id_digest": value.retired_actor_id_digest,
        "retired_binding_method": value.retired_binding_method,
        "retired_binding_digest": value.retired_binding_digest,
        "retired_at": value.retired_at,
        "record_digest": value.record_digest,
        "contains_public_key_material": True,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
        "mutation_performed": mutation,
        "deletion_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_signature_keys_cli"
        ),
        description=(
            "Register, inspect, and retire governed Ed25519 custody signer public keys. "
            "The registry never stores or reads private signing keys."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register")
    register.add_argument("--owner-id", required=True)
    register.add_argument("--key-id", required=True)
    register.add_argument("--public-key-path", required=True)
    register.add_argument("--valid-from", type=float)
    register.add_argument("--valid-until", type=float)
    register.add_argument("--confirm-key-id", required=True)
    register.add_argument("--registry-path")

    status = commands.add_parser("status")
    status.add_argument("--owner-id", required=True)
    status.add_argument("--key-id", required=True)
    status.add_argument("--registry-path")

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state", choices=("active", "retired"))
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--registry-path")

    retire = commands.add_parser("retire")
    retire.add_argument("--owner-id", required=True)
    retire.add_argument("--key-id", required=True)
    retire.add_argument("--confirm-key-id", required=True)
    retire.add_argument("--registry-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command in {"register", "retire"} and args.confirm_key_id != args.key_id:
            raise ValueError("key confirmation differs from requested key ID.")
        if args.command == "register":
            actor = load_relation_review_actor()
            now = time.time()
            value = register_custody_signer_key(
                registry=get_custody_signer_key_registry(args.registry_path),
                owner_id=args.owner_id,
                key_id=args.key_id,
                public_key_path=args.public_key_path,
                actor=actor,
                valid_from=now if args.valid_from is None else args.valid_from,
                valid_until=args.valid_until,
                now=now,
            )
            _print(_summary(value, mutation=True))
            return 0
        if args.command == "status":
            value = get_custody_signer_key_registry(args.registry_path).get(
                owner_id=args.owner_id,
                key_id=args.key_id,
            )
            _print(_summary(value, mutation=False))
            return 0
        if args.command == "list":
            values = get_custody_signer_key_registry(args.registry_path).list(
                owner_id=args.owner_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "count": len(values),
                    "items": [_summary(value, mutation=False) for value in values],
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                    "mutation_performed": False,
                    "deletion_performed": False,
                }
            )
            return 0
        if args.command == "retire":
            actor = load_relation_review_actor()
            value = get_custody_signer_key_registry(args.registry_path).retire(
                owner_id=args.owner_id,
                key_id=args.key_id,
                actor=actor,
            )
            _print(_summary(value, mutation=True))
            return 0
        raise ValueError("unsupported signer key command.")
    except PermissionError:
        _print({"error": "authorization_failed"}, stream=sys.stderr)
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
