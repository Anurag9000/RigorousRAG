"""Restricted installed CLI for advanced RAG training/checkpoint/export operations.

The underlying ``advanced_rag_operator`` also contains historical research-grade evaluation
and qualification helpers.  Those remain importable for reproducibility, but the installed
training command deliberately excludes them so production evaluation/promotion has one
advertised authority: ``rigorousrag-advanced-release``.

Training now also persists a content-bound ``training_result.json`` beside the authoritative
checkpoint root.  Classical and retrieval authorities already expose stable result manifests;
this closes the advanced-RAG orchestration gap so downstream validation/metrics jobs consume a
real artifact instead of scraping stdout.  The manifest contains no second training/evaluation
implementation: it records the return value of the existing authoritative operator and binds it
to the exact training-config bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from training.advanced_rag_config import load_advanced_run_config
from training.advanced_rag_operator import (
    export_from_config,
    train_from_config,
    validate_config,
    verify_artifact,
    verify_checkpoint_from_config,
)

RESULT_SCHEMA = "rigorousrag-authoritative-advanced-training-result/v1"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _jsonable(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except Exception:
            directory = None
        if directory is not None:
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _persist_training_result(config_path: str | Path, result: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    configured = load_advanced_run_config(selected)
    checkpoint_root = Path(configured.checkpoint_root).expanduser().resolve(strict=False)
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "complete": True,
        "kind": str(result.get("kind") or ""),
        "config_path": str(selected),
        "config_sha256": _sha_file(selected),
        "plan_sha256": str(result.get("plan_sha256") or ""),
        "checkpoint_root": str(checkpoint_root),
        "result": _jsonable(result),
    }
    payload["result_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _atomic_json(checkpoint_root / "training_result.json", payload)
    return payload


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
        result = _persist_training_result(args.config, train_from_config(args.config))
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


__all__ = ["RESULT_SCHEMA", "main"]
