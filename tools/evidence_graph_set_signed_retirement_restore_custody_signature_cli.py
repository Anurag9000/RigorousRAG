"""Operator CLI for Ed25519-signed external restore custody evidence."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_rfc3161_trust_runtime import (
    get_rfc3161_trust_registry,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    bind_rfc3161_timestamp_to_signed_custody,
    load_signed_custody_envelope,
    sign_governed_restore_chain_of_custody,
    verify_governed_signed_restore_chain_of_custody,
    verify_governed_timestamped_signed_restore_chain_of_custody,
    verify_signed_restore_chain_of_custody,
    verify_timestamped_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature_keys_runtime import (
    get_custody_signer_key_registry,
)


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _envelope_summary(value: Any, *, timestamped: bool) -> dict[str, Any]:
    envelope = value.signed_envelope if timestamped else value
    return {
        "owner_id": envelope.owner_id,
        "key_id": envelope.key_id,
        "algorithm": envelope.algorithm,
        "public_key_sha256": envelope.public_key_sha256,
        "manifest_chain_digest": envelope.manifest.chain_digest,
        "envelope_digest": envelope.envelope_digest,
        "created_at": envelope.created_at,
        "timestamped": timestamped,
        "timestamp_receipt_digest": (
            value.timestamp_receipt.receipt_digest if timestamped else None
        ),
        "timestamped_subject_sha256": (
            value.timestamped_subject_sha256 if timestamped else None
        ),
        "contains_private_key_material": False,
        "contains_raw_paths": False,
        "mutation_performed": False,
        "restore_performed": False,
        "import_performed": False,
        "deletion_performed": False,
    }


def _verification_summary(value: Any) -> dict[str, Any]:
    payload = value.public_payload()
    payload.update(
        contains_private_key_material=False,
        contains_raw_paths=False,
        mutation_performed=False,
        restore_performed=False,
        import_performed=False,
        deletion_performed=False,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m tools."
            "evidence_graph_set_signed_retirement_restore_custody_signature_cli"
        ),
        description=(
            "Create and verify Ed25519 custody signatures. Signing requires one "
            "governed active public-key record; no command stores private keys."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sign = commands.add_parser("sign")
    sign.add_argument("--owner-id", required=True)
    sign.add_argument("--key-id", required=True)
    sign.add_argument("--manifest", required=True)
    sign.add_argument("--private-key-path", required=True)
    sign.add_argument("--output", required=True)
    sign.add_argument("--registry-path")

    verify = commands.add_parser("verify")
    verify.add_argument("envelope_path")
    verify.add_argument("--public-key-path", required=True)
    verify.add_argument("--expected-key-id")
    verify.add_argument("--expected-owner-id")

    bind = commands.add_parser("bind-timestamp")
    bind.add_argument("--signed-envelope", required=True)
    bind.add_argument("--receipt", required=True)
    bind.add_argument("--public-key-path", required=True)
    bind.add_argument("--output", required=True)
    bind.add_argument("--expected-key-id")

    verify_timestamped = commands.add_parser("verify-timestamped")
    verify_timestamped.add_argument("envelope_path")
    verify_timestamped.add_argument("--public-key-path", required=True)
    verify_timestamped.add_argument("--expected-key-id")
    verify_timestamped.add_argument("--expected-owner-id")

    governed = commands.add_parser("verify-governed")
    governed.add_argument("envelope_path")
    governed.add_argument("--owner-id", required=True)
    governed.add_argument("--registry-path")

    governed_timestamped = commands.add_parser("verify-governed-timestamped")
    governed_timestamped.add_argument("envelope_path")
    governed_timestamped.add_argument("--owner-id", required=True)
    governed_timestamped.add_argument("--tsa-profile-id", required=True)
    governed_timestamped.add_argument("--request-bundle", required=True)
    governed_timestamped.add_argument("--response", required=True)
    governed_timestamped.add_argument("--trust-anchor-bundle", required=True)
    governed_timestamped.add_argument("--untrusted-bundle")
    governed_timestamped.add_argument("--crl-bundle")
    governed_timestamped.add_argument("--registry-path")
    governed_timestamped.add_argument("--tsa-registry-path")
    governed_timestamped.add_argument("--openssl-binary", default="openssl")
    governed_timestamped.add_argument("--timeout-seconds", type=int, default=30)
    governed_timestamped.add_argument(
        "--maximum-future-seconds",
        type=float,
        default=300.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "sign":
            envelope, record = sign_governed_restore_chain_of_custody(
                registry=get_custody_signer_key_registry(args.registry_path),
                owner_id=args.owner_id,
                key_id=args.key_id,
                manifest_path=args.manifest,
                private_key_path=args.private_key_path,
                output_path=args.output,
            )
            payload = _envelope_summary(envelope, timestamped=False)
            payload.update(
                registry_record_digest=record.record_digest,
                signer_key_state=record.state,
                output_created=True,
                mutation_performed=True,
            )
            _print(payload)
            return 0
        if args.command == "verify":
            envelope = verify_signed_restore_chain_of_custody(
                envelope_path=args.envelope_path,
                public_key_path=args.public_key_path,
                expected_key_id=args.expected_key_id,
                expected_owner_id=args.expected_owner_id,
            )
            _print(_envelope_summary(envelope, timestamped=False))
            return 0
        if args.command == "bind-timestamp":
            value = bind_rfc3161_timestamp_to_signed_custody(
                signed_envelope_path=args.signed_envelope,
                receipt_path=args.receipt,
                public_key_path=args.public_key_path,
                output_path=args.output,
                expected_key_id=args.expected_key_id,
            )
            payload = _envelope_summary(value, timestamped=True)
            payload.update(output_created=True, mutation_performed=True)
            _print(payload)
            return 0
        if args.command == "verify-timestamped":
            value = verify_timestamped_signed_restore_chain_of_custody(
                envelope_path=args.envelope_path,
                public_key_path=args.public_key_path,
                expected_key_id=args.expected_key_id,
                expected_owner_id=args.expected_owner_id,
            )
            _print(_envelope_summary(value, timestamped=True))
            return 0
        if args.command == "verify-governed":
            envelope = load_signed_custody_envelope(args.envelope_path)
            receipt = verify_governed_signed_restore_chain_of_custody(
                registry=get_custody_signer_key_registry(args.registry_path),
                owner_id=args.owner_id,
                signed_envelope=envelope,
            )
            _print(_verification_summary(receipt))
            return 0
        if args.command == "verify-governed-timestamped":
            receipt = verify_governed_timestamped_signed_restore_chain_of_custody(
                registry=get_custody_signer_key_registry(args.registry_path),
                tsa_registry=get_rfc3161_trust_registry(args.tsa_registry_path),
                owner_id=args.owner_id,
                profile_id=args.tsa_profile_id,
                timestamped_envelope_path=args.envelope_path,
                request_bundle_path=args.request_bundle,
                response_path=args.response,
                trust_anchor_bundle_path=args.trust_anchor_bundle,
                untrusted_bundle_path=args.untrusted_bundle,
                crl_bundle_path=args.crl_bundle,
                openssl_binary=args.openssl_binary,
                timeout_seconds=args.timeout_seconds,
                maximum_future_seconds=args.maximum_future_seconds,
            )
            _print(_verification_summary(receipt))
            return 0
        raise ValueError("unsupported custody signature command.")
    except PermissionError:
        _print({"error": "signature_verification_failed"}, stream=sys.stderr)
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
