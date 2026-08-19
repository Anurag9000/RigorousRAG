"""Restricted installed CLI for advanced RAG training/checkpoint/export operations.

The underlying ``advanced_rag_operator`` also contains historical research-grade evaluation
and qualification helpers.  Those remain importable for reproducibility, but the installed
training command deliberately excludes them so production evaluation/promotion has one
advertised authority: ``rigorousrag-advanced-release``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping, Sequence

from training.advanced_rag_operator import (
    export_from_config,
    train_from_config,
    validate_config,
    verify_artifact,
    verify_checkpoint_from_config,
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-advanced-training",
        description="Authoritative advanced RAG training/checkpoint/export operator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate")
    validate.add_argument("--config", required=True)
    validate.add_argument("--load-models", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)

    verify_checkpoint = sub.add_parser("verify-checkpoint")
    verify_checkpoint.add_argument("--config", required=True)
    verify_checkpoint.add_argument("--checkpoint-digest", required=True)

    export = sub.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--checkpoint-digest", required=True)
    export.add_argument("--destination", required=True)

    artifact = sub.add_parser("verify-artifact")
    artifact.add_argument("--directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = validate_config(args.config, load_models=args.load_models)
    elif args.command == "train":
        result = train_from_config(args.config)
    elif args.command == "verify-checkpoint":
        result = verify_checkpoint_from_config(
            args.config,
            checkpoint_digest=args.checkpoint_digest,
        )
    elif args.command == "export":
        # Production release evidence is created later by the dedicated release operator.
        # Export therefore does not accept a research-grade evaluation receipt here.
        result = export_from_config(
            args.config,
            checkpoint_digest=args.checkpoint_digest,
            destination=args.destination,
            evaluation_receipt_path=None,
        )
    elif args.command == "verify-artifact":
        result = verify_artifact(args.directory)
    else:  # pragma: no cover
        raise RuntimeError("unreachable command")
    print(json.dumps(_jsonable(result), sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
