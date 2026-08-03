"""Read-only custody signer rotation audit CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, Sequence

from tools.evidence_graph_set_signed_retirement_restore_custody_signer_operations import (
    CustodySignerRotationPolicy,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_operations_boundary import (
    assess_custody_signer_rotation,
)
from tools.evidence_graph_set_signed_retirement_restore_custody_signer_readonly import (
    ReadOnlyCustodySignerKeyRegistry,
)

_DEFAULT_PATH = "data/evidence_graph_set_signed_retirement_custody_signers.sqlite3"


def _print(value: Any, *, stream: Any = None) -> None:
    destination = sys.stdout if stream is None else stream
    destination.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "python -m "
            "tools.evidence_graph_set_signed_retirement_restore_custody_signer_operations_cli"
        ),
        description=(
            "Audit public custody signer records against an explicit rotation policy. "
            "The command never registers or retires keys."
        ),
    )
    parser.add_argument("--owner-id", required=True)
    parser.add_argument("--registry-db-path")
    parser.add_argument("--allowed-issuer", action="append", required=True)
    parser.add_argument("--maximum-active-keys", type=int, default=2)
    parser.add_argument(
        "--maximum-key-age-seconds",
        type=float,
        default=365 * 24 * 60 * 60,
    )
    parser.add_argument(
        "--rotation-warning-seconds",
        type=float,
        default=30 * 24 * 60 * 60,
    )
    parser.add_argument(
        "--minimum-overlap-seconds",
        type=float,
        default=7 * 24 * 60 * 60,
    )
    parser.add_argument("--limit", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        path = args.registry_db_path or os.getenv(
            "EVIDENCE_GRAPH_RESTORE_CUSTODY_SIGNER_DB_PATH",
            _DEFAULT_PATH,
        )
        policy = CustodySignerRotationPolicy.create(
            maximum_active_keys=args.maximum_active_keys,
            maximum_key_age_seconds=args.maximum_key_age_seconds,
            rotation_warning_seconds=args.rotation_warning_seconds,
            minimum_overlap_seconds=args.minimum_overlap_seconds,
            allowed_issuers=args.allowed_issuer,
        )
        report = assess_custody_signer_rotation(
            owner_id=args.owner_id,
            registry=ReadOnlyCustodySignerKeyRegistry(path),
            policy=policy,
            limit=args.limit,
        )
        payload = asdict(report)
        payload.update(
            {
                "policy": {
                    "maximum_active_keys": policy.maximum_active_keys,
                    "maximum_key_age_seconds": policy.maximum_key_age_seconds,
                    "rotation_warning_seconds": policy.rotation_warning_seconds,
                    "minimum_overlap_seconds": policy.minimum_overlap_seconds,
                    "allowed_issuers": list(policy.allowed_issuers),
                    "policy_digest": policy.policy_digest,
                },
                "registry_mutation_performed": False,
                "key_material_mutation_performed": False,
                "key_deletion_performed": False,
                "source_text_returned": False,
                "raw_path_returned": False,
            }
        )
        _print(payload)
        return 0
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _print({"error": "invalid_or_unavailable"}, stream=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
