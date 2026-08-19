"""Strict CLI for content-bound evaluator contract publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from evaluation.authoritative_evaluator_contract import (
    build_authoritative_evaluator_contract,
    verify_authoritative_evaluator_contract,
    write_authoritative_evaluator_contract,
)
from training.advanced_path_authority import safe_advanced_path


def publish_evaluator_contract(
    config_path: str | Path,
    *,
    output_path: str | Path,
) -> Mapping[str, object]:
    output = safe_advanced_path(
        output_path,
        label="authoritative evaluator contract output",
        must_exist=False,
    )
    if output.exists():
        raise ValueError("authoritative evaluator contract output must not already exist")
    contract = build_authoritative_evaluator_contract(config_path)
    write_authoritative_evaluator_contract(output, contract)
    verified = verify_authoritative_evaluator_contract(output)
    if verified.contract_sha256 != contract.contract_sha256:
        raise RuntimeError("evaluator contract changed during publication/reconstruction")
    return {
        "evaluator_id": verified.evaluator_id,
        "implementation_id": verified.implementation_id,
        "source_commit": verified.source_commit,
        "config_sha256": verified.config_sha256,
        "metric_names": [item.name for item in verified.metrics],
        "contract_sha256": verified.contract_sha256,
        "output": str(
            safe_advanced_path(
                output,
                label="authoritative evaluator contract receipt",
                must_exist=True,
                require_file=True,
            )
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-evaluator-contract",
        description="Publish a content-bound local evaluator contract receipt",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = publish_evaluator_contract(args.config, output_path=args.output)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "publish_evaluator_contract"]
