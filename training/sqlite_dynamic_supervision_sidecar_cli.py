"""CLI for publishing and verifying SQLite-backed dynamic supervision sidecars."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from training.sqlite_dynamic_supervision_sidecars import (
    publish_sqlite_dynamic_supervision_sidecar,
    verify_sqlite_dynamic_supervision_sidecar,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-dynamic-sidecar",
        description="Publish or verify a sealed SQLite dynamic-RAG supervision sidecar authority",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--source", required=True)
    publish.add_argument("--source-sha256", required=True)
    publish.add_argument(
        "--kind",
        required=True,
        choices=("information_need", "realized_gain", "logged_value", "counterfactual"),
    )
    publish.add_argument("--semantic-contract-sha256")
    publish.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        receipt = publish_sqlite_dynamic_supervision_sidecar(
            args.source,
            expected_sha256=args.source_sha256,
            kind=args.kind,
            semantic_contract_sha256=args.semantic_contract_sha256,
            output_dir=args.output_dir,
        )
    else:
        receipt = verify_sqlite_dynamic_supervision_sidecar(args.receipt)
    print(
        json.dumps(
            {
                "kind": receipt.kind,
                "record_count": receipt.record_count,
                "source_sha256": receipt.source_sha256,
                "row_digest_sha256": receipt.row_digest_sha256,
                "provider_contract_sha256": receipt.provider_contract_sha256,
                "receipt_sha256": receipt.receipt_sha256,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
