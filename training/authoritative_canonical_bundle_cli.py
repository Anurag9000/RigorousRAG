"""CLI for emitting neutral training bundles from restart-verified canonical v2 data."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from training.authoritative_canonical_bundle_bridge import (
    write_authoritative_dynamic_canonical_bundle,
    write_authoritative_grounded_canonical_bundle,
)
from training.authoritative_canonical_recipe_bridge import (
    read_authoritative_canonical_training_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-canonical-bundle",
        description="Build or verify a training bundle backed by canonical v2 data authority",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    grounded = sub.add_parser("grounded")
    grounded.add_argument("--canonical-receipt", required=True)
    grounded.add_argument("--output", required=True)
    dynamic = sub.add_parser("dynamic")
    dynamic.add_argument("--canonical-receipt", required=True)
    dynamic.add_argument("--output", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "grounded":
        bundle = write_authoritative_grounded_canonical_bundle(
            args.output,
            args.canonical_receipt,
        )
    elif args.command == "dynamic":
        bundle = write_authoritative_dynamic_canonical_bundle(
            args.output,
            args.canonical_receipt,
        )
    else:
        bundle = read_authoritative_canonical_training_bundle(args.bundle)
    print(
        json.dumps(
            {
                "kind": bundle.kind,
                "dataset_manifest_sha256": bundle.dataset_manifest_sha256,
                "canonical_receipt_sha256": bundle.canonical_receipt_sha256,
                "bundle_sha256": bundle.bundle_sha256,
                "split_names": [item.name for item in bundle.splits],
                "cache_roles": [item.role for item in bundle.caches],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
