"""Config-only CLI for restart-verifiable advanced runtime stack bundles."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from orchestration.authoritative_advanced_runtime_bundle import (
    build_authoritative_advanced_runtime_bundle,
    verify_authoritative_advanced_runtime_bundle,
)
from training.advanced_path_authority import safe_advanced_path

_MAX_CONFIG_BYTES = 16 * 1024 * 1024
_MAX_COMPONENTS = 1_000


def _read(path: str | Path) -> Mapping[str, Any]:
    source = safe_advanced_path(path, label="advanced runtime bundle config", must_exist=True, require_file=True)
    if source.stat().st_size <= 0 or source.stat().st_size > _MAX_CONFIG_BYTES:
        raise ValueError("advanced runtime bundle config exceeds byte safety bound")
    try:
        value = json.loads(source.read_text(encoding="utf-8", errors="strict"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except Exception as exc:
        raise ValueError("advanced runtime bundle config is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("advanced runtime bundle config must contain an object")
    return value


def _finite(value: Any, label: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    selected = float(value)
    if not math.isfinite(selected):
        raise ValueError(f"{label} must be finite")
    return selected


def build_runtime_bundle_from_config(path: str | Path) -> Mapping[str, object]:
    raw = _read(path)
    required = {
        "schema", "output_dir", "stack_id", "advanced_sources", "other_components",
        "retrieval_contract_sha256", "generation_contract_sha256", "compatibility_sha256",
        "source_revision", "valid_from", "expires_at",
    }
    if set(raw) != required or raw.get("schema") != "rigorousrag-authoritative-advanced-runtime-bundle-config/v1":
        raise ValueError("config must be rigorousrag-authoritative-advanced-runtime-bundle-config/v1")
    advanced = raw["advanced_sources"]
    others = raw["other_components"]
    if not isinstance(advanced, list) or not advanced or len(advanced) > _MAX_COMPONENTS:
        raise ValueError("advanced_sources must be a bounded non-empty array")
    if not isinstance(others, list) or len(others) > _MAX_COMPONENTS:
        raise ValueError("other_components must be a bounded array")
    valid_from = _finite(raw["valid_from"], "valid_from")
    expires_at = _finite(raw["expires_at"], "expires_at", allow_none=True)
    receipt = build_authoritative_advanced_runtime_bundle(
        advanced_sources=tuple(advanced),
        other_components=tuple(others),
        stack_id=raw["stack_id"],
        retrieval_contract_sha256=raw["retrieval_contract_sha256"],
        generation_contract_sha256=raw["generation_contract_sha256"],
        compatibility_sha256=raw["compatibility_sha256"],
        source_revision=raw["source_revision"],
        valid_from=valid_from,
        expires_at=expires_at,
        output_dir=raw["output_dir"],
    )
    verified = verify_authoritative_advanced_runtime_bundle(
        Path(raw["output_dir"]) / "bundle_receipt.json"
    )
    if verified.receipt_sha256 != receipt.receipt_sha256:
        raise RuntimeError("advanced runtime bundle changed during reconstruction")
    return {
        "stack_sha256": receipt.stack_sha256,
        "offline_quality_evidence_sha256": receipt.offline_quality_evidence_sha256,
        "source_set_sha256": receipt.source_set_sha256,
        "stack_config_sha256": receipt.stack_config_sha256,
        "bundle_receipt_sha256": receipt.receipt_sha256,
        "output_dir": str(
            safe_advanced_path(
                raw["output_dir"],
                label="advanced runtime bundle output",
                must_exist=True,
                require_directory=True,
            )
        ),
    }


def verify_runtime_bundle(path: str | Path) -> Mapping[str, object]:
    receipt = verify_authoritative_advanced_runtime_bundle(path)
    return {
        "stack_sha256": receipt.stack_sha256,
        "offline_quality_evidence_sha256": receipt.offline_quality_evidence_sha256,
        "source_set_sha256": receipt.source_set_sha256,
        "stack_config_sha256": receipt.stack_config_sha256,
        "bundle_receipt_sha256": receipt.receipt_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rigorousrag-runtime-bundle",
        description="Build or verify a restart-verifiable promoted advanced runtime stack bundle",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--config", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = (
        build_runtime_bundle_from_config(args.config)
        if args.command == "build"
        else verify_runtime_bundle(args.receipt)
    )
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_runtime_bundle_from_config", "main", "verify_runtime_bundle"]
