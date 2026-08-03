"""Operator CLI for offline RFC 3161 custody timestamp interoperability."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161 import (
    create_rfc3161_timestamp_request_bundle,
    emit_rfc3161_timestamp_request_der,
    verify_rfc3161_timestamp_receipt,
    verify_rfc3161_timestamp_request_bundle,
    verify_rfc3161_timestamp_response,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _request_summary(value: Any, *, output_path: str | None = None) -> dict[str, Any]:
    result = {
        "owner_id": value.owner_id,
        "subject_sha256": value.subject_sha256,
        "subject_size_bytes": value.subject_size_bytes,
        "request_sha256": value.request_sha256,
        "request_bundle_digest": value.bundle_digest,
        "requested_policy_oid": value.requested_policy_oid,
        "hash_algorithm": value.hash_algorithm,
        "cert_req": value.cert_req,
        "rfc3161_request": True,
        "trusted_time_obtained": False,
        "network_request_performed": False,
        "contains_private_key_material": False,
        "contains_raw_subject_content": False,
    }
    if output_path is not None:
        result["output_path"] = output_path
    return result


def _receipt_summary(value: Any, *, output_path: str | None = None) -> dict[str, Any]:
    result = {
        "owner_id": value.owner_id,
        "request_bundle_digest": value.request_bundle_digest,
        "subject_sha256": value.subject_sha256,
        "response_sha256": value.response_sha256,
        "token_sha256": value.token_sha256,
        "status": value.status,
        "policy_oid": value.policy_oid,
        "serial_decimal": value.serial_decimal,
        "generated_at_rfc3339": value.generated_at_rfc3339,
        "signer_certificate_sha256": value.signer_certificate_sha256,
        "trust_anchor_bundle_sha256": value.trust_anchor_bundle_sha256,
        "crl_bundle_sha256": value.crl_bundle_sha256,
        "receipt_digest": value.receipt_digest,
        "rfc3161_token": True,
        "message_imprint_verified": True,
        "nonce_verified": True,
        "policy_verified": True,
        "cms_signature_verified": True,
        "certificate_chain_verified": True,
        "revocation_checked": value.revocation_checked,
        "tsa_eku_verified": True,
        "ess_signer_binding_verified": True,
        "independently_trusted_clock_proven": False,
        "hardware_clock_proven": False,
        "network_request_performed": False,
        "contains_private_key_material": False,
        "contains_raw_paths": False,
        "mutation_performed": False,
    }
    if output_path is not None:
        result["output_path"] = output_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_cli"
        ),
        description=(
            "Create offline RFC 3161 request bundles, emit DER requests, and verify "
            "returned TimeStampResp files against pinned certificate evidence. "
            "No command contacts a timestamp authority."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create-request")
    create.add_argument("--owner-id", required=True)
    create.add_argument("--custody-envelope", required=True)
    create.add_argument("--output-bundle", required=True)
    create.add_argument("--policy-oid")

    emit = commands.add_parser("emit-request")
    emit.add_argument("--request-bundle", required=True)
    emit.add_argument("--output-der", required=True)

    inspect = commands.add_parser("inspect-request")
    inspect.add_argument("--request-bundle", required=True)

    verify = commands.add_parser("verify-response")
    verify.add_argument("--request-bundle", required=True)
    verify.add_argument("--response", required=True)
    verify.add_argument("--trust-anchor-bundle", required=True)
    verify.add_argument("--output-receipt", required=True)
    verify.add_argument("--untrusted-bundle")
    verify.add_argument("--crl-bundle")
    verify.add_argument("--expected-policy-oid")
    verify.add_argument("--openssl-binary", default="openssl")
    verify.add_argument("--timeout-seconds", type=int, default=30)
    verify.add_argument("--maximum-future-seconds", type=float, default=300.0)

    receipt = commands.add_parser("verify-receipt")
    receipt.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "create-request":
            value = create_rfc3161_timestamp_request_bundle(
                owner_id=args.owner_id,
                custody_envelope_path=args.custody_envelope,
                output_bundle_path=args.output_bundle,
                requested_policy_oid=args.policy_oid,
            )
            _print(_request_summary(value, output_path=args.output_bundle))
            return 0
        if args.command == "emit-request":
            value = emit_rfc3161_timestamp_request_der(
                request_bundle_path=args.request_bundle,
                output_der_path=args.output_der,
            )
            _print(_request_summary(value, output_path=args.output_der))
            return 0
        if args.command == "inspect-request":
            value = verify_rfc3161_timestamp_request_bundle(args.request_bundle)
            _print(_request_summary(value))
            return 0
        if args.command == "verify-response":
            value = verify_rfc3161_timestamp_response(
                request_bundle_path=args.request_bundle,
                response_path=args.response,
                trust_anchor_bundle_path=args.trust_anchor_bundle,
                output_receipt_path=args.output_receipt,
                untrusted_bundle_path=args.untrusted_bundle,
                crl_bundle_path=args.crl_bundle,
                expected_policy_oid=args.expected_policy_oid,
                openssl_binary=args.openssl_binary,
                timeout_seconds=args.timeout_seconds,
                maximum_future_seconds=args.maximum_future_seconds,
            )
            _print(_receipt_summary(value, output_path=args.output_receipt))
            return 0
        if args.command == "verify-receipt":
            value = verify_rfc3161_timestamp_receipt(args.receipt)
            _print(_receipt_summary(value))
            return 0
        raise ValueError("unsupported RFC 3161 command.")
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
