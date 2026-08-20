"""CLI for publishing/verifying snapshot-keyed dynamic feature observations."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from orchestration.dynamic_feature_observation_sidecar import (
    publish_dynamic_feature_observations,
    verify_dynamic_feature_observations,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rigorousrag-dynamic-feature-sidecar")
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--source", required=True)
    publish.add_argument("--sha256", required=True)
    publish.add_argument("--semantic-contract-sha256", required=True)
    publish.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        receipt = publish_dynamic_feature_observations(
            args.source,
            expected_sha256=args.sha256,
            semantic_contract_sha256=args.semantic_contract_sha256,
            output_dir=args.output_dir,
        )
    else:
        receipt = verify_dynamic_feature_observations(args.receipt)
    print(json.dumps({
        "source_sha256": receipt.source_sha256,
        "semantic_contract_sha256": receipt.semantic_contract_sha256,
        "record_count": receipt.record_count,
        "provider_contract_sha256": receipt.provider_contract_sha256,
        "receipt_sha256": receipt.receipt_sha256,
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
