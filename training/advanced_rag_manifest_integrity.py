"""Self-consistency checks for advanced RAG inference artifact manifests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from training.advanced_rag_artifacts import AdvancedArtifactManifest


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def assert_advanced_manifest_self_consistent(manifest: AdvancedArtifactManifest) -> str:
    if not isinstance(manifest, AdvancedArtifactManifest):
        raise ValueError("manifest must be AdvancedArtifactManifest")
    values = asdict(manifest)
    provided = values.pop("artifact_sha256")
    unsigned = {"schema": "rigorousrag-advanced-inference-artifact/v3", **values}
    expected = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if expected != provided:
        raise ValueError("advanced artifact manifest artifact_sha256 does not match its payload")
    return expected


__all__ = ["assert_advanced_manifest_self_consistent"]
