"""Compliance-enforced signing and governance-aware historical verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_export_boundary import (
    verify_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signature import (
    _load_private,
    _public_fingerprint,
    sign_restore_chain_of_custody,
    verify_signed_restore_chain_of_custody,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_admin_use_audit_readonly import (
    ReadOnlyCustodySignerAdminUseAuditStore,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_compliance import (
    audit_custody_signer_compliance,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_readonly import (
    ReadOnlyCustodySignerKeyRegistry,
)

_DEFAULT_REGISTRY = "data/evidence_graph_set_signed_retirement_custody_signers.sqlite3"


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def _stores(args):
    registry_path = args.registry_db_path or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
        _DEFAULT_REGISTRY,
    )
    admin_path = args.admin_use_db_path or os.getenv(
        "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_ADMIN_USE_DB_PATH"
    )
    registry = ReadOnlyCustodySignerKeyRegistry(registry_path)
    admin = (
        None
        if not admin_path
        else ReadOnlyCustodySignerAdminUseAuditStore(admin_path)
    )
    return registry, admin


def _compliance(*, owner_id: str, key_id: str, registry, admin, limit: int):
    report = audit_custody_signer_compliance(
        owner_id=owner_id,
        registry=registry,
        admin_use_store=admin,
        limit=limit,
    )
    matches = tuple(item for item in report.items if item.key_id == key_id)
    if len(matches) != 1:
        raise KeyError((owner_id, key_id))
    return report, matches[0]


def _compliance_payload(report, item):
    return {
        "registration_classification": item.registration_classification,
        "registration_use_id": item.registration_use_id,
        "retirement_classification": item.retirement_classification,
        "retirement_use_id": item.retirement_use_id,
        "eligible_for_new_signatures": item.eligible_for_new_signatures,
        "governance_compliant_for_historical_verification": (
            item.governance_compliant_for_historical_verification
        ),
        "compliance_report_digest": report.report_digest,
        "registry_mutation_performed": False,
        "admin_use_mutation_performed": False,
        "key_material_mutation_performed": False,
        "source_text_returned": False,
        "raw_path_returned": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_signer_enforced_cli"
        ),
        description=(
            "Create signatures only with governance-compliant active signer records, "
            "or verify historical signatures with separate compliance reporting."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    sign = commands.add_parser("sign-compliant")
    sign.add_argument("manifest_path")
    sign.add_argument("--owner-id", required=True)
    sign.add_argument("--key-id", required=True)
    sign.add_argument("--private-key-path", required=True)
    sign.add_argument("--output", required=True)
    sign.add_argument("--registry-db-path")
    sign.add_argument("--admin-use-db-path")
    sign.add_argument("--limit", type=int, default=1_000)

    verify = commands.add_parser("verify-compliance")
    verify.add_argument("envelope_path")
    verify.add_argument("--owner-id", required=True)
    verify.add_argument("--key-id", required=True)
    verify.add_argument("--public-key-path", required=True)
    verify.add_argument("--registry-db-path")
    verify.add_argument("--admin-use-db-path")
    verify.add_argument("--limit", type=int, default=1_000)
    verify.add_argument("--require-governance-compliance", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        registry, admin = _stores(args)
        report, item = _compliance(
            owner_id=args.owner_id,
            key_id=args.key_id,
            registry=registry,
            admin=admin,
            limit=args.limit,
        )
        record = registry.get(owner_id=args.owner_id, key_id=args.key_id)
        if args.command == "sign-compliant":
            if not item.eligible_for_new_signatures:
                raise PermissionError("signer record is not governance-compliant.")
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
                    "owner_id": record.owner_id,
                    "key_id": record.key_id,
                    "issuer": record.issuer,
                    "public_key_sha256": record.public_key_sha256,
                    "restore_id": envelope.manifest.restore_id,
                    "chain_digest": envelope.manifest.chain_digest,
                    "signature_created": True,
                    **_compliance_payload(report, item),
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
        governance_ok = item.governance_compliant_for_historical_verification
        if args.require_governance_compliance and not governance_ok:
            raise PermissionError("historical signer governance is noncompliant.")
        _print(
            {
                "owner_id": record.owner_id,
                "key_id": record.key_id,
                "issuer": record.issuer,
                "public_key_sha256": record.public_key_sha256,
                "restore_id": envelope.manifest.restore_id,
                "chain_digest": envelope.manifest.chain_digest,
                "signature_valid": True,
                "governance_requirement_enforced": (
                    args.require_governance_compliance
                ),
                **_compliance_payload(report, item),
            }
        )
        return 0
    except PermissionError:
        _print({"error": "not_authorized_or_noncompliant"}, stream=sys.stderr)
        return 1
    except (FileExistsError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
