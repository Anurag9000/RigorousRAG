"""Fail-closed safetensor supervision cache for authoritative advanced RAG workflows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from training.advanced_path_authority import safe_advanced_path
from training.advanced_rag_supervision import SafetensorSupervisionCache, SupervisionCacheIdentity

_MAX_ENTRY_BYTES = 512 * 1024 * 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024


class AuthoritativeSafetensorSupervisionCache(SafetensorSupervisionCache):
    """Base cache plus path authority, strict JSON and bounded entry verification."""
    def __init__(self, root: str | Path, identity: SupervisionCacheIdentity) -> None:
        safe = safe_advanced_path(root, label="supervision cache root", must_exist=False)
        if safe.exists() and not safe.is_dir():
            raise ValueError("supervision cache root must be a directory when it exists")
        super().__init__(safe, identity)

    def get(self, key: str) -> Mapping[str, Any]:
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for supervision cache reads") from exc
        tensor_path, manifest_path = self._paths(key)
        for path, label, maximum in (
            (tensor_path, "supervision tensor entry", _MAX_ENTRY_BYTES),
            (manifest_path, "supervision manifest entry", _MAX_MANIFEST_BYTES),
        ):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"{label} must be a regular non-symlink file")
            size = path.stat().st_size
            if size <= 0 or size > maximum:
                raise ValueError(f"{label} exceeds byte safety bound")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"), parse_constant=lambda raw: (_ for _ in ()).throw(ValueError(raw)))
        except Exception as exc:
            raise ValueError("supervision cache manifest is not strict JSON") from exc
        required = {"schema", "key", "identity_sha256", "tensor_sha256", "tensor_names"}
        if not isinstance(manifest, Mapping) or set(manifest) != required:
            raise ValueError("supervision cache manifest has unexpected fields")
        if manifest.get("schema") != "rigorousrag-supervision-cache-entry/v1" or manifest.get("identity_sha256") != self.identity.digest or manifest.get("key") != key:
            raise ValueError("supervision cache manifest identity mismatch")
        tensor_sha = str(manifest.get("tensor_sha256", "")).strip().lower()
        if len(tensor_sha) != 64 or any(ch not in "0123456789abcdef" for ch in tensor_sha):
            raise ValueError("supervision cache manifest tensor_sha256 is invalid")
        actual = hashlib.sha256()
        with tensor_path.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                actual.update(block)
        if actual.hexdigest() != tensor_sha:
            raise ValueError("supervision cache tensor digest mismatch")
        names = manifest.get("tensor_names")
        if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names) or names != sorted(set(names)):
            raise ValueError("supervision cache tensor_names must be a sorted unique string list")
        tensors = load_file(str(tensor_path), device="cpu")
        if sorted(tensors) != names:
            raise ValueError("supervision cache tensor names differ from manifest")
        return tensors


__all__ = ["AuthoritativeSafetensorSupervisionCache"]
