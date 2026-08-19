"""Filesystem-hardened verification for authoritative advanced runtime bundles."""
from __future__ import annotations

from pathlib import Path

from orchestration.authoritative_advanced_runtime_bundle import (
    AuthoritativeAdvancedRuntimeBundleReceipt,
    verify_authoritative_advanced_runtime_bundle,
)
from training.advanced_path_authority import safe_advanced_path

_EXPECTED = (
    "stack.json",
    "offline_quality.json",
    "bindings.json",
    "bundle_receipt.json",
)


def verify_strict_authoritative_advanced_runtime_bundle(
    receipt_path: str | Path,
) -> AuthoritativeAdvancedRuntimeBundleReceipt:
    receipt = safe_advanced_path(
        receipt_path,
        label="advanced runtime bundle receipt",
        must_exist=True,
        require_file=True,
    )
    root = receipt.parent
    for name in _EXPECTED:
        child = safe_advanced_path(
            root / name,
            label=f"advanced runtime bundle {name}",
            must_exist=True,
            require_file=True,
        )
        if child.parent != root or child.name != name:
            raise ValueError(f"advanced runtime bundle child {name} escapes canonical root")
    return verify_authoritative_advanced_runtime_bundle(receipt)


__all__ = ["verify_strict_authoritative_advanced_runtime_bundle"]
