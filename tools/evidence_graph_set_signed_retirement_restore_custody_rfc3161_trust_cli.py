"""Operator CLI for governed external RFC 3161 TSA trust profiles."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_relation_actor import load_relation_review_actor
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust import (
    register_rfc3161_trust_profile,
    verify_rfc3161_timestamp_response_with_profile,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_runtime import (
    get_rfc3161_trust_registry,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        + "\n"
    )


def _profile(value: Any, *, mutation: bool) -> dict[str, Any]:
    return {
        "owner_id": value.owner_id,
        "profile_id": value.profile_id,
        "policy_oid": value.policy_oid,
        "trust_anchor_bundle_sha256": value.trust_anchor_bundle_sha256,
        "untrusted_bundle_sha256": value.untrusted_bundle_sha256,
        "crl_bundle_sha256": value.crl_bundle_sha256,
        "allowed_signer_certificate_sha256": list(
            value.allowed_signer_certificate_sha256
        ),
        "valid_from": value.valid_from,
        "valid_until": value.valid_until,
        "state": value.state,
        "registered_at": value.registered_at,
        "retired_at": value.retired_at,
        "record_digest": value.record_digest,
        "registry_mutation_performed": mutation,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_cli"
        ),
        description=(
            "Register, retire, inspect and enforce owner-scoped external TSA "
            "trust profiles for RFC 3161 custody verification."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser("register")
    register.add_argument("--owner-id", required=True)
    register.add_argument("--profile-id", required=True)
    register.add_argument("--confirm-profile-id", required=True)
    register.add_argument("--policy-oid", required=True)
    register.add_argument("--trust-anchor-bundle", required=True)
    register.add_argument("--untrusted-bundle")
    register.add_argument("--crl-bundle")
    register.add_argument(
        "--allowed-signer-sha256",
        action="append",
        default=[],
    )
    register.add_argument("--valid-from", type=float, required=True)
    register.add_argument("--valid-until", type=float)

    retire = commands.add_parser("retire")
    retire.add_argument("--owner-id", required=True)
    retire.add_argument("--profile-id", required=True)
    retire.add_argument("--confirm-profile-id", required=True)

    status = commands.add_parser("status")
    status.add_argument("--owner-id", required=True)
    status.add_argument("--profile-id", required=True)

    listing = commands.add_parser("list")
    listing.add_argument("--owner-id", required=True)
    listing.add_argument("--state", choices=("active", "retired"))
    listing.add_argument("--limit", type=int, default=100)

    verify = commands.add_parser("verify-response")
    verify.add_argument("--owner-id", required=True)
    verify.add_argument("--profile-id", required=True)
    verify.add_argument("--request-bundle", required=True)
    verify.add_argument("--response", required=True)
    verify.add_argument("--trust-anchor-bundle", required=True)
    verify.add_argument("--untrusted-bundle")
    verify.add_argument("--crl-bundle")
    verify.add_argument("--output-receipt", required=True)
    verify.add_argument("--openssl-binary", default="openssl")
    verify.add_argument("--timeout-seconds", type=int, default=30)
    verify.add_argument(
        "--maximum-future-seconds",
        type=float,
        default=300.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command in {"register", "retire"} and (
            args.profile_id != args.confirm_profile_id
        ):
            raise ValueError("profile confirmation differs.")
        registry = get_rfc3161_trust_registry()
        if args.command == "register":
            value = register_rfc3161_trust_profile(
                registry=registry,
                owner_id=args.owner_id,
                profile_id=args.profile_id,
                policy_oid=args.policy_oid,
                trust_anchor_bundle_path=args.trust_anchor_bundle,
                untrusted_bundle_path=args.untrusted_bundle,
                crl_bundle_path=args.crl_bundle,
                allowed_signer_certificate_sha256=(
                    args.allowed_signer_sha256
                ),
                valid_from=args.valid_from,
                valid_until=args.valid_until,
                actor=load_relation_review_actor(),
            )
            _print(_profile(value, mutation=True))
            return 0
        if args.command == "retire":
            value = registry.retire(
                owner_id=args.owner_id,
                profile_id=args.profile_id,
                actor=load_relation_review_actor(),
            )
            _print(_profile(value, mutation=True))
            return 0
        if args.command == "status":
            value = registry.get(
                owner_id=args.owner_id,
                profile_id=args.profile_id,
            )
            _print(_profile(value, mutation=False))
            return 0
        if args.command == "list":
            values = registry.list(
                owner_id=args.owner_id,
                state=args.state,
                limit=args.limit,
            )
            _print(
                {
                    "owner_id": args.owner_id,
                    "record_count": len(values),
                    "records": [
                        _profile(value, mutation=False) for value in values
                    ],
                    "registry_mutation_performed": False,
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                }
            )
            return 0
        if args.command == "verify-response":
            receipt, profile = verify_rfc3161_timestamp_response_with_profile(
                registry=registry,
                owner_id=args.owner_id,
                profile_id=args.profile_id,
                request_bundle_path=args.request_bundle,
                response_path=args.response,
                trust_anchor_bundle_path=args.trust_anchor_bundle,
                untrusted_bundle_path=args.untrusted_bundle,
                crl_bundle_path=args.crl_bundle,
                output_receipt_path=args.output_receipt,
                openssl_binary=args.openssl_binary,
                timeout_seconds=args.timeout_seconds,
                maximum_future_seconds=args.maximum_future_seconds,
            )
            _print(
                {
                    "owner_id": receipt.owner_id,
                    "profile_id": profile.profile_id,
                    "profile_record_digest": profile.record_digest,
                    "receipt_digest": receipt.receipt_digest,
                    "policy_oid": receipt.policy_oid,
                    "signer_certificate_sha256": (
                        receipt.signer_certificate_sha256
                    ),
                    "generated_at_rfc3339": receipt.generated_at_rfc3339,
                    "governed_profile_verified": True,
                    "rfc3161_token": True,
                    "independently_trusted_clock_proven": False,
                    "hardware_clock_proven": False,
                    "registry_mutation_performed": False,
                    "contains_private_key_material": False,
                    "contains_raw_paths": False,
                }
            )
            return 0
        raise ValueError("unsupported RFC 3161 trust command.")
    except (
        FileExistsError,
        KeyError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
