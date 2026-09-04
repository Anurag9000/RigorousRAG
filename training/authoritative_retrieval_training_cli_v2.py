"""Fail-closed external-input contract for learned retrieval training.

The v1 retrieval authority already binds the complete admitted train/validation bytes and
local model/tokenizer trees into its exact-resume architecture identity.  This v2 entry
adds a lightweight preflight before importing any heavy training dependency:

* all-zero ``source_commit`` placeholders are rejected (``auto`` is the only automatic
  source-revision spelling);
* every required dataset/model/tokenizer input must exist, be regular, and be free of
  symlinks;
* an optional ``expected_inputs`` object can pin the SHA-256 identities supplied by an
  upstream producer/materializer.  When present it is closed-world and must cover every
  admitted input exactly.

The authoritative optimizer/checkpoint implementation remains v1; this module deliberately
contains no second training loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "rigorousrag-authoritative-retrieval-training/v1"
RESULT_SCHEMA = "rigorousrag-authoritative-retrieval-training-result/v1"
_HEX = frozenset("0123456789abcdef")
_EXPECTED_BASE_KEYS = frozenset(
    {
        "train_data_sha256",
        "validation_data_sha256",
        "model_tree_sha256",
        "tokenizer_tree_sha256",
    }
)
_UNTIED_KEY = "untied_document_model_tree_sha256"


def _identifier(value: Any, label: str, maximum: int = 4000) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    selected = value.strip()
    if not selected or len(selected) > maximum or any(ord(ch) < 32 or ord(ch) == 127 for ch in selected):
        raise ValueError(f"{label} is invalid")
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(base: Path, value: Any, label: str, *, directory: bool = False) -> Path:
    selected = Path(_identifier(value, label)).expanduser()
    candidate = selected if selected.is_absolute() else base / selected
    # Check the lexical path before resolve so a final symlink cannot disappear into its
    # target.  Resolve then also protects against symlinked parents by comparing each
    # lexical component below.
    absolute = candidate.absolute()
    if absolute.is_symlink():
        raise ValueError(f"{label} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if directory:
        if not resolved.is_dir():
            raise ValueError(f"{label} must be a directory")
    elif not resolved.is_file():
        raise ValueError(f"{label} must be a file")
    return resolved


def _tree_sha256(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact root must be a regular directory")
    digest = hashlib.sha256()
    files: list[Path] = []
    for entry in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        if entry.is_symlink():
            raise ValueError(f"artifact tree contains a symlink: {entry}")
        if entry.is_file():
            files.append(entry)
    if not files:
        raise ValueError(f"artifact directory is empty: {root}")
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _expected_sha256(value: Any, label: str) -> str:
    selected = _identifier(value, label, 64).lower()
    if len(selected) != 64 or any(ch not in _HEX for ch in selected):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    if selected == "0" * 64:
        raise ValueError(f"{label} may not use the all-zero placeholder digest")
    return selected


def _reject_placeholder_source_commit(config: Mapping[str, Any]) -> None:
    value = config.get("source_commit", "auto")
    selected = _identifier(value, "source_commit", 64).lower()
    if selected in {"0" * 40, "0" * 64}:
        raise ValueError("source_commit may not use an all-zero placeholder; use 'auto' for the checked-out Git revision")


def _preflight(config_path: str | Path) -> Mapping[str, Any]:
    selected = Path(config_path).expanduser().resolve(strict=True)
    raw = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("retrieval training config must contain an object")
    if raw.get("schema") != SCHEMA:
        raise ValueError(f"retrieval training config schema must be {SCHEMA!r}")
    _reject_placeholder_source_commit(raw)
    base = selected.parent

    train = _path(base, raw.get("train_data"), "train_data")
    validation = _path(base, raw.get("validation_data"), "validation_data")
    model_root = _path(base, raw.get("model_root"), "model_root", directory=True)
    tokenizer_root = _path(base, raw.get("tokenizer_root", raw.get("model_root")), "tokenizer_root", directory=True)

    actual: dict[str, str] = {
        "train_data_sha256": _sha256_file(train),
        "validation_data_sha256": _sha256_file(validation),
        "model_tree_sha256": _tree_sha256(model_root),
        "tokenizer_tree_sha256": _tree_sha256(tokenizer_root),
    }
    model = raw.get("model", {})
    if not isinstance(model, Mapping):
        raise ValueError("model must be an object")
    untied = model.get("untied_document_model_root")
    if untied is not None:
        untied_root = _path(base, untied, "untied_document_model_root", directory=True)
        actual[_UNTIED_KEY] = _tree_sha256(untied_root)

    expected_raw = raw.get("expected_inputs")
    if expected_raw is not None:
        if not isinstance(expected_raw, Mapping):
            raise ValueError("expected_inputs must be an object")
        required = set(_EXPECTED_BASE_KEYS)
        if _UNTIED_KEY in actual:
            required.add(_UNTIED_KEY)
        keys = {str(key) for key in expected_raw}
        if keys != required:
            raise ValueError(
                "expected_inputs must cover admitted retrieval inputs exactly; "
                f"missing={sorted(required - keys)}, unexpected={sorted(keys - required)}"
            )
        for key in sorted(required):
            expected = _expected_sha256(expected_raw[key], f"expected_inputs[{key!r}]")
            if expected != actual[key]:
                raise ValueError(
                    f"retrieval input digest mismatch for {key}: expected {expected}, actual {actual[key]}"
                )

    return {
        "schema": "rigorousrag-retrieval-input-preflight/v2",
        "config": str(selected),
        "actual_inputs": actual,
        "expected_inputs_verified": expected_raw is not None,
    }


def run_config(config_path: str | Path) -> Mapping[str, Any]:
    _preflight(config_path)
    # Heavy torch/transformers imports happen only after the fail-closed byte-identity
    # preflight.  v1 remains the sole optimizer/checkpoint authority.
    from training import authoritative_retrieval_training_cli as v1

    return v1.run_config(config_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="preflight and run/continue one retrieval recipe")
    train.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "train":
        print(json.dumps(run_config(args.config), sort_keys=True, separators=(",", ":")))
        return 0
    raise RuntimeError(f"unsupported command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["RESULT_SCHEMA", "SCHEMA", "_preflight", "main", "run_config"]
