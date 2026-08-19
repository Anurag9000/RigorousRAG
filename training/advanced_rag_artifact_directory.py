"""Closed-directory byte authority for advanced-RAG inference artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path

_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _stream_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _strict_manifest(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("advanced artifact manifest must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > _MAX_MANIFEST_BYTES:
        raise ValueError("advanced artifact manifest exceeds byte safety bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)),
        )
    except Exception as exc:
        raise ValueError("advanced artifact manifest is not strict JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema") != "rigorousrag-advanced-inference-artifact/v3":
        raise ValueError("unsupported advanced artifact manifest schema")
    if "artifact_sha256" not in value:
        raise ValueError("advanced artifact manifest lacks artifact_sha256")
    return value


def assert_artifact_directory_matches_manifest(directory: str | Path, manifest: Any) -> Path:
    """Re-verify exact persisted manifest/weights against a supplied manifest object.

    This function intentionally avoids importing ``AdvancedArtifactManifest`` so it can be used
    by the artifact module itself without an import cycle. The supplied value must be a
    dataclass with the expected digest/weight attributes; its canonical payload must equal the
    persisted manifest exactly after JSON normalization.
    """
    if not is_dataclass(manifest):
        raise ValueError("advanced artifact manifest must be a dataclass instance")
    artifact_sha = str(getattr(manifest, "artifact_sha256", "")).strip().lower()
    weights_sha = str(getattr(manifest, "weights_sha256", "")).strip().lower()
    weights_bytes = getattr(manifest, "weights_bytes", None)
    for value, label in ((artifact_sha, "artifact_sha256"), (weights_sha, "weights_sha256")):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{label} must be SHA-256")
    if isinstance(weights_bytes, bool) or not isinstance(weights_bytes, int) or weights_bytes <= 0:
        raise ValueError("weights_bytes must be positive")

    root = safe_advanced_path(directory, label="advanced artifact directory", must_exist=True, require_directory=True)
    if root.name != artifact_sha:
        raise ValueError("advanced artifact directory name differs from artifact_sha256")
    children = {item.name: item for item in root.iterdir()}
    if set(children) != {"manifest.json", "model.safetensors"}:
        raise ValueError("advanced artifact directory must contain exactly manifest.json and model.safetensors")
    if any(item.is_symlink() or not item.is_file() for item in children.values()):
        raise ValueError("advanced artifact children must be regular non-symlink files")

    persisted = _strict_manifest(children["manifest.json"])
    unsigned = dict(persisted)
    persisted_sha = str(unsigned.pop("artifact_sha256")).strip().lower()
    computed_sha = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if persisted_sha != computed_sha or persisted_sha != artifact_sha:
        raise ValueError("persisted advanced artifact manifest self-digest mismatch")

    supplied = json.loads(_canonical({"schema": "rigorousrag-advanced-inference-artifact/v3", **asdict(manifest)}).decode("utf-8"))
    if supplied != dict(persisted):
        raise ValueError("supplied artifact manifest differs from persisted manifest")

    weights = children["model.safetensors"]
    if weights.stat().st_size != weights_bytes:
        raise ValueError("advanced artifact weights byte count changed")
    if _stream_sha(weights) != weights_sha:
        raise ValueError("advanced artifact weights digest changed")
    return root


__all__ = ["assert_artifact_directory_matches_manifest"]
